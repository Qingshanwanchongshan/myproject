#!/usr/bin/env python3
"""训练监控：把 loss / margin / accuracies / grad_norm / 权重 / entropy 的趋势打成表 + 走势图。

数据源二选一，自动判断：
  * 训练日志（`logs/ea.log`）—— 每 logging_steps 一条，训练中就有
  * `trainer_state.json` —— 每 save_steps 落一次，含 eval 指标

    python tools/monitor.py logs/ea.log
    python tools/monitor.py outputs/ea-3b-run1            # 自动找 trainer_state.json
    python tools/monitor.py logs/ea.log --watch 60        # 每 60 秒刷新
"""

import argparse
import ast
import glob
import json
import os
import re
import statistics as st
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SPARK = "▁▂▃▄▅▆▇█"

# (键名, 显示名, 是否"越小越好")
METRICS = [
    ("loss", "loss", True),
    ("rewards/margins", "margin", False),
    ("rewards/accuracies", "acc@8对", False),
    ("eval_rewards/accuracies", "eval_acc@512", False),
    ("grad_norm", "grad_norm", False),
    ("entropy", "entropy", None),
    ("mean_token_accuracy", "token_acc", False),
    ("ea/weight_max_rejected", "w_max_rej", None),
    ("ea/weight_max_chosen", "w_max_cho", None),
    ("ea/weight_mean_rejected", "w_mean_rej", None),
    ("learning_rate", "lr", None),
]


def sparkline(values, width=48):
    """把一列数压成一行走势图。"""
    if not values:
        return ""
    step = max(1, len(values) // width)
    buckets = [values[i : i + step] for i in range(0, len(values), step)][:width]
    points = [sum(b) / len(b) for b in buckets if b]
    low, high = min(points), max(points)
    if high - low < 1e-12:
        return SPARK[0] * len(points)
    return "".join(SPARK[int((p - low) / (high - low) * (len(SPARK) - 1))] for p in points)


def parse_log(path):
    """从训练日志里抽出所有指标字典（tqdm 的 \\r 会干扰，先归一化）。"""
    text = Path(path).read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    rows = []
    for match in re.finditer(r"\{'loss':.*?\}", text):
        try:
            data = ast.literal_eval(match.group(0))
        except (ValueError, SyntaxError):
            continue
        if isinstance(data, dict) and "loss" in data:
            rows.append(data)
    return rows


def parse_state(run_dir):
    """从 trainer_state.json 读 log_history（含 eval 行）。"""
    candidates = sorted(glob.glob(os.path.join(run_dir, "checkpoint-*", "trainer_state.json")))
    root = os.path.join(run_dir, "trainer_state.json")
    if os.path.isfile(root):
        candidates.append(root)
    if not candidates:
        return []
    data = json.loads(Path(candidates[-1]).read_text(encoding="utf-8"))
    return [row for row in data.get("log_history", []) if isinstance(row, dict)]


def num(row, key):
    value = row.get(key)
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return None
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def collect(rows):
    """{key: [值...]}，跳过缺值的行。"""
    series = {}
    for key, _, _ in METRICS:
        values = [v for v in (num(row, key) for row in rows) if v is not None]
        if values:
            series[key] = values
    return series


def progress(rows, total_steps):
    """用 epoch 推断当前 step（日志里没有 step 字段）。"""
    epochs = [v for v in (num(row, "epoch") for row in rows) if v is not None]
    if not epochs or not total_steps:
        return ""
    step = int(epochs[-1] * total_steps)
    return f"step {step}/{total_steps}（{step / total_steps * 100:.1f}%）epoch {epochs[-1]:.3f}"


def render(path, total_steps, tail):
    if os.path.isdir(path):
        rows = parse_state(path)
        source = "trainer_state.json"
    else:
        rows = parse_log(path)
        source = os.path.basename(path)
    if not rows:
        print(f"[fail] {path} 里没有解析到任何指标")
        return False

    series = collect(rows)
    head = f"数据源 {source} · {len(rows)} 个点"
    line = progress(rows, total_steps)
    print("=" * 78)
    print(f"  {head}" + (f" · {line}" if line else ""))
    print("=" * 78)
    print(f"  {'指标':<12}{'首':>9}{'末':>9}{'均值±σ':>17}   走势")
    for key, label, _ in METRICS:
        values = series.get(key)
        if not values:
            continue
        recent = values[-tail:]
        mean = st.mean(recent)
        sd = st.pstdev(recent) if len(recent) > 1 else 0.0
        print(f"  {label:<12}{values[0]:>9.4g}{values[-1]:>9.4g}"
              f"{mean:>10.4g}±{sd:<6.3g} {sparkline(recent)}")

    evals = [row for row in rows if num(row, "eval_loss") is not None]
    print()
    if evals:
        print(f"  评测点 {len(evals)} 个（512 对，噪声 σ≈2.2%）：")
        for row in evals[-6:]:
            step = row.get("step", "?")
            acc = num(row, "eval_rewards/accuracies")
            loss = num(row, "eval_loss")
            margin = num(row, "eval_rewards/margins")
            ent = num(row, "eval_entropy")
            print(f"    step {str(step):>6}  acc={acc if acc is None else round(acc, 4)}"
                  f"  loss={loss if loss is None else round(loss, 4)}"
                  f"  margin={margin if margin is None else round(margin, 3)}"
                  f"  entropy={ent if ent is None else round(ent, 3)}")
    else:
        print("  还没有评测点（第一次在 eval_steps，通常 500 步）")
    print()
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="训练日志，或输出目录（含 checkpoint-*/trainer_state.json）")
    parser.add_argument("--total-steps", type=int, default=3813, help="总步数（用于算进度）")
    parser.add_argument("--tail", type=int, default=200, help="只统计最近多少个点")
    parser.add_argument("--watch", type=int, default=0, help="每 N 秒刷新一次（0 = 只跑一次）")
    args = parser.parse_args()

    if args.watch:
        while True:
            os.system("clear" if os.name != "nt" else "cls")
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            render(args.path, args.total_steps, args.tail)
            sys.stdout.flush()
            time.sleep(args.watch)
    return 0 if render(args.path, args.total_steps, args.tail) else 1


if __name__ == "__main__":
    sys.exit(main())
