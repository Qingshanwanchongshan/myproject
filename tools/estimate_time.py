#!/usr/bin/env python3
"""从训练日志里读出实测速度，换算两轮实验的预计耗时（只用标准库）。

训练结束时 transformers 会打印一块 `***** Train metrics *****`，其中
`train_runtime` 是 `H:MM:SS.ff` 的时间串、`train_steps_per_second` 是浮点。
这两种格式在不同版本里不稳定，所以解析逻辑放在这里，方便单独测。

    python tools/estimate_time.py logs/timing-ea.log
    python tools/estimate_time.py logs/timing-ea.log --steps 1500 --measured-steps 40
"""

import argparse
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPOCH_STEPS = 7642  # 61135 对 / 有效 batch 8
EVAL_PAIRS = 512

PATTERNS = {
    "steps_per_second": r"train_steps_per_second'?\s*[=:]\s*'?([0-9.]+)",
    "runtime": r"train_runtime'?\s*[=:]\s*'?([0-9][0-9:.]*)",
}


def to_seconds(value):
    """'0:05:12.34' / '5:12' / '312.4' -> 秒。"""
    parts = value.split(":")
    total = 0.0
    for part in parts:
        total = total * 60 + float(part)
    return total


def parse(path):
    text = open(path, encoding="utf-8", errors="replace").read()
    found = {}
    for name, pattern in PATTERNS.items():
        matches = re.findall(pattern, text)
        if matches:
            found[name] = matches[-1]
    steps_per_second = None
    if "steps_per_second" in found:
        steps_per_second = float(found["steps_per_second"])
    runtime = to_seconds(found["runtime"]) if "runtime" in found else None
    return steps_per_second, runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", help="测速那一轮的训练日志")
    parser.add_argument("--steps", type=int, default=1500, help="第一轮每组多少步（默认 1500）")
    parser.add_argument("--measured-steps", type=int, default=40, help="测速那一轮实际跑了多少步")
    parser.add_argument("--epoch-steps", type=int, default=EPOCH_STEPS, help="全量 1 epoch 的步数")
    parser.add_argument("--eval-pairs", type=int, default=EVAL_PAIRS, help="固定评测集大小")
    parser.add_argument("--eval-every", type=int, default=250, help="每隔多少步评测一次")
    args = parser.parse_args()

    try:
        steps_per_second, runtime = parse(args.log)
    except FileNotFoundError:
        print(f"[fail] 找不到日志 {args.log}", file=sys.stderr)
        return 1

    if steps_per_second is None and runtime:
        steps_per_second = args.measured_steps / runtime
        source = f"由 train_runtime={runtime:.1f}s / {args.measured_steps} 步换算"
    elif steps_per_second is not None:
        source = "直接读 train_steps_per_second"
    else:
        print(f"[fail] {args.log} 里没有 train_runtime / train_steps_per_second", file=sys.stderr)
        print("       确认这轮跑完了（Ctrl-C 中断的日志里没有这块）", file=sys.stderr)
        return 1

    if steps_per_second <= 0:
        print(f"[fail] 解析出 {steps_per_second} 步/秒，日志内容不对", file=sys.stderr)
        return 1

    step_time = 1.0 / steps_per_second
    # 一步 = 8 个微批 × (策略前向 + 重算 + 反向 + 参考前向) ≈ 40 个序列前向
    forward_time = step_time / 40.0
    # 一次评测：每个偏好对 2 条序列 × (策略 + 参考) 两个前向；评测没有反向，通常比训练快 2-3 倍
    eval_seconds = args.eval_pairs * 4 * forward_time / 2.5
    evals = max(args.steps // args.eval_every, 1)

    print(f"来源        : {args.log}")
    print(f"解析方式    : {source}")
    print(f"实测步速    : {steps_per_second:.3f} 步/秒（{step_time:.2f} 秒/步）")
    if runtime:
        print(f"train_runtime: {runtime:.1f} 秒")
    print()
    print(f"第一轮每组 {args.steps} 步")
    print(f"  EA-DPO      ≈ {args.steps * step_time / 3600:.1f} 小时")
    print(f"  对照组      ≈ {args.steps * step_time * 0.95 / 3600:.1f} 小时（EA 的注意力统计省掉）")
    print(f"  评测        ≈ {evals} 次 × {eval_seconds / 60:.1f} 分钟（粗估，评测时没有反向）")
    print(f"  两组合计    ≈ {(args.steps * 2 * step_time + evals * 2 * eval_seconds) / 3600:.1f} 小时")
    print()
    print(f"第二轮全量 {args.epoch_steps} 步（1 epoch）")
    print(f"  一组        ≈ {args.epoch_steps * step_time / 3600:.1f} 小时")
    print()
    print("说明")
    print("  * 测速那一轮含首次前向/编译开销，真实长跑通常略快；以 trainer_state.json 的 train_runtime 为准")
    print("  * 评测时间按“每个前向耗时与训练时相同”粗估，实际会因 batch 与无反向而偏快")
    print("  * 换卡/换 max_length/换 batch 后重新测一次")
    return 0


if __name__ == "__main__":
    sys.exit(main())
