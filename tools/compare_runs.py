#!/usr/bin/env python3
"""把两组实验的 trainer_state.json 摊在一起看趋势（只用标准库）。

    python tools/compare_runs.py outputs/dpo-3b-20pct outputs/ea-3b-20pct --labels DPO EA-DPO
    python tools/compare_runs.py outputs/dpo-3b-20pct outputs/ea-3b-20pct --csv logs/compare.csv

先说清楚哪些数字能横向比、哪些不能 —— 这是本工具存在的意义。

EA-DPO 的 loss / rewards 用的是**加权后**的逐 token log-prob（权重由熵和因果入度
决定），标准 DPO 用的是不加权求和。所以：

    能横向比（定义与加权无关）
        entropy              策略输出分布的熵
        token_acc            chosen 的 token 准确率
        logit_chosen         词表平均 logit

    只能各自看趋势（口径不同，不是同一个函数）
        loss / eval_loss     EA 是加权 DPO 损失
        win_rate*            EA 是"加权 margin 的胜率"
        margin*              EA 是加权 margin

    结论层面
        真正证明 EA-DPO 有没有用的，是固定 prompt 集上的生成评测，不是这里的数。
"""

import argparse
import csv
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

STATE_NAME = "trainer_state.json"
STEP_WIDTH = 8
COL_WIDTH = 13

# (配置里的键名候选, 显示名, 是否可横向比)
METRICS = [
    (("entropy",), "entropy", True),
    (("mean_token_accuracy",), "token_acc", True),
    (("logits/chosen",), "logit_chosen", True),
    (("rewards/accuracies",), "win_rate", False),
    (("rewards/margins",), "margin", False),
    (("loss",), "loss", False),
]

HEALTH = [
    (("ea/weight_mean_chosen",), "weight_mean_chosen"),
    (("ea/weight_mean_rejected",), "weight_mean_rejected"),
    (("ea/weight_max_rejected",), "weight_max_rejected"),
    (("ea/target_layer",), "target_layer"),
]


def find_state(path):
    """返回 trainer_state.json 的路径；找不到返回 None（与 tools/report.py 同样的定位规则）。"""
    if os.path.isfile(path):
        return path
    candidates = [os.path.join(path, STATE_NAME)]
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.startswith("checkpoint-"):
                candidates.append(os.path.join(path, name, STATE_NAME))
    return next((c for c in candidates if os.path.isfile(c)), None)


def load_run(path, label):
    state_path = find_state(path)
    if state_path is None:
        return None
    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    history = [row for row in state.get("log_history", []) if isinstance(row, dict)]
    train = [row for row in history if "loss" in row and "eval_loss" not in row]
    evals = [row for row in history if "eval_loss" in row]
    return {
        "label": label,
        "state": state_path,
        "train": train,
        "evals": evals,
        "eval_by_step": {row.get("step"): row for row in evals},
        "step": state.get("global_step"),
        "epoch": state.get("epoch"),
    }


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def pick(row, names):
    """行里取第一个可用的数；评测行带 eval_ 前缀，训练行不带。"""
    if row is None:
        return None
    for name in names:
        for key in (f"eval_{name}", name):
            if is_number(row.get(key)):
                return row[key]
    return None


def cell(value, width=COL_WIDTH):
    return "-".rjust(width) if value is None else f"{value:.4g}".rjust(width)


def table(title, headers, rows):
    if not rows:
        print(f"\n{title}\n  （没有数据）")
        return
    print(f"\n{title}")
    print("  " + "step".rjust(STEP_WIDTH) + "".join(header.rjust(COL_WIDTH) for header in headers))
    for step, cells in rows:
        print("  " + str(step).rjust(STEP_WIDTH) + "".join(cell(value) for value in cells))


def metric_rows(run_a, run_b, names):
    """按并集步数对齐评测点，任一侧缺失留 None。"""
    left = run_a["eval_by_step"]
    right = run_b["eval_by_step"]
    rows = []
    for step in sorted({s for s in left if isinstance(s, int)} | {s for s in right if isinstance(s, int)}):
        a, b = pick(left.get(step), names), pick(right.get(step), names)
        diff = None if a is None or b is None else a - b
        rows.append((step, [a, b, diff]))
    return rows


def health(run):
    train = run["train"]
    lines = []
    losses = [row["loss"] for row in train if is_number(row.get("loss"))]
    if losses:
        head = sum(losses[:5]) / len(losses[:5])
        tail = sum(losses[-5:]) / len(losses[-5:])
        lines.append(f"loss {head:.4f} -> {tail:.4f}")
    norms = [row["grad_norm"] for row in train if is_number(row.get("grad_norm"))]
    if norms:
        clipped = sum(1 for value in norms if value >= 1.0)
        line = f"grad_norm 均值 {sum(norms) / len(norms):.3f} 最大 {max(norms):.3f}"
        if clipped:
            line += f"，{clipped}/{len(norms)} 步撞到 1.0 被裁剪"
        lines.append(line)
    for names, display in HEALTH:
        values = [value for value in (pick(row, names) for row in train) if value is not None]
        if values:
            lines.append(f"{display} 最新 {values[-1]:.4g} 最大 {max(values):.4g}")
    return lines


def write_csv(path, run_a, run_b):
    rows = {}
    for run, prefix in ((run_a, ""), (run_b, "b_")):
        for row in run["train"] + run["evals"]:
            step = row.get("step")
            if not isinstance(step, int):
                continue
            target = rows.setdefault(step, {})
            for key, value in row.items():
                if key != "step" and is_number(value):
                    target[f"{prefix}{key}"] = value
    columns = sorted({key for row in rows.values() for key in row})
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step"] + columns)
        for step in sorted(rows):
            writer.writerow([step] + [rows[step].get(key, "") for key in columns])
    print(f"\nCSV 已写入 {path}（{len(rows)} 个点，{len(columns)} 列）")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a", help="第一组，一般是标准 DPO 对照组")
    parser.add_argument("run_b", help="第二组，一般是 EA-DPO")
    parser.add_argument("--labels", nargs=2, default=["A", "B"], metavar=("A", "B"))
    parser.add_argument("--csv", help="把合并后的逐点数据写成 CSV")
    args = parser.parse_args()

    run_a = load_run(args.run_a, args.labels[0])
    run_b = load_run(args.run_b, args.labels[1])
    missing = [name for name, run in zip(args.labels, (run_a, run_b), strict=True) if run is None]
    if missing:
        for path, label in zip((args.run_a, args.run_b), args.labels, strict=True):
            if label in missing:
                print(f"[fail] {path} 下没有 {STATE_NAME}", file=sys.stderr)
        print("       save_strategy: no 时不写这个文件（冒烟跑没有它是正常的）", file=sys.stderr)
        return 1

    print("=" * 78)
    for run in (run_a, run_b):
        print(f"{run['label']:<10} {run['state']}")
        print(f"{'':<10} 训练点 {len(run['train'])}  评测点 {len(run['evals'])}  "
              f"global_step={run['step']}  epoch={run['epoch']}")
    print("=" * 78)

    groups = [
        ("同口径，可以横向比", [metric for metric in METRICS if metric[2]]),
        ("不同口径，只能各自看趋势", [metric for metric in METRICS if not metric[2]]),
    ]
    for title, metrics in groups:
        print(f"\n{'=' * 30} {title} {'=' * 30}")
        if not any(run["evals"] for run in (run_a, run_b)):
            print("\n  两组都还没有评测点（eval_strategy 为 no，或还没到 eval_steps）")
            continue
        for names, display, _ in metrics:
            table(
                f"[{display}]",
                [f"{run_a['label']}", f"{run_b['label']}", "diff"],
                metric_rows(run_a, run_b, names),
            )
            if display == "win_rate":
                print("        * EA 的胜率是加权口径（权重随样本自适应），与基线不是同一个函数；"
                      "要横向比请用 tools/eval_implicit_reward.py")

    print(f"\n{'=' * 30} 各自看：训练健康度 {'=' * 30}")
    for run in (run_a, run_b):
        lines = health(run)
        print(f"\n  {run['label']}")
        if not lines:
            print("    没有训练日志点")
        for line in lines:
            print(f"    {line}")

    if args.csv:
        write_csv(args.csv, run_a, run_b)

    print("""
怎么用这张表
  1. 两组 loss 都在降、entropy 都不崩 -> 两个训练都健康，可以谈方法差异
  2. 同一组自己随步数的走向（尤其 win_rate 是否单调上升）是可信的：同一把尺子量同一个模型
  3. 跨组比 win_rate 要小心：EA 组是加权口径，与基线不是同一个函数。做法是统一尺子：
       python tools/eval_implicit_reward.py --adapter outputs/dpo-3b-20pct:DPO \\
                                            --adapter outputs/ea-3b-20pct:EA-DPO
     它会用不加权的定义把两组都重算一遍，并把每组的"口径差"打出来
  4. 想比较生成质量：两组 adapter 在同一批固定 prompt 上各生成一次，裁判模型 pairwise 判定
  5. EA 组 weight_max_rejected 长期 > 20 说明权重过度集中，
     下一轮把 ea_tau_rejected 从 0.7 调到 0.85 左右再试
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
