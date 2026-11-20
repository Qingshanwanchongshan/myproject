#!/usr/bin/env python3
"""Summarize an EA-DPO / DPO run from `trainer_state.json` (standard library only).

The file is written by the Trainer on every save (`save_strategy: steps`) and holds the full
`log_history`; with `save_strategy: no` it does not exist at all.

Usage:
    python tools/report.py outputs/ea-3b-epoch1
    python tools/report.py outputs/ea-3b-epoch1/checkpoint-500 --tail 20
    python tools/report.py outputs/ea-3b-epoch1 --csv /tmp/ea.csv
"""

import argparse
import csv
import json
import os
import sys

WATCHED = (
    "loss",
    "grad_norm",
    "entropy",
    "rewards/accuracies",
    "rewards/margins",
    "ea/weight_mean_chosen",
    "ea/weight_mean_rejected",
    "ea/weight_max_chosen",
    "ea/weight_max_rejected",
    "ea/target_layer",
    "eval_loss",
    "eval_rewards/accuracies",
)
NUMERIC_REQUIRED = ("loss", "grad_norm")


def find_state(path):
    candidates = [os.path.join(path, "trainer_state.json")]
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.startswith("checkpoint-"):
                candidates.append(os.path.join(path, name, "trainer_state.json"))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def main():
    parser = argparse.ArgumentParser(description="Summarize a training run from trainer_state.json")
    parser.add_argument("run_dir", help="output_dir or a checkpoint-* directory")
    parser.add_argument("--tail", type=int, default=8, help="how many recent log rows to print (default 8)")
    parser.add_argument("--csv", help="also dump the full log_history to this CSV file")
    args = parser.parse_args()

    state_path = find_state(args.run_dir)
    if state_path is None:
        print(f"no trainer_state.json under {args.run_dir!r}")
        print("hint: it is only written when save_strategy != 'no'; for a smoke run pass --save_strategy steps")
        return 1

    with open(state_path, encoding="utf-8") as handle:
        state = json.load(handle)
    history = [row for row in state.get("log_history", []) if isinstance(row, dict)]
    rows = [row for row in history if "loss" in row or "eval_loss" in row]
    if not rows:
        print(f"{state_path}: log_history has {len(history)} entries but no loss/eval_loss rows")
        return 1

    steps = [row["step"] for row in rows if number(row.get("step"))]
    print(f"file      : {state_path}")
    print(f"log rows  : {len(history)} (with loss: {len(rows)})")
    if steps:
        print(f"steps     : {min(steps)} .. {max(steps)}")
    print(f"epochs    : {state.get('epoch')}   global_step: {state.get('global_step')}")

    columns = [key for key in WATCHED if any(number(row.get(key)) for row in rows)]
    if columns:
        print("\nrecent log rows")
        width = max(len(key) for key in columns)
        print("  " + f"{'step':>7}  " + "  ".join(key.rjust(width) for key in columns))
        for row in rows[-max(args.tail, 1) :]:
            cells = []
            for key in columns:
                value = row.get(key)
                cells.append(("" if not number(value) else f"{value:.4g}").rjust(width))
            print(f"  {row.get('step', -1):>7}  " + "  ".join(cells))

    problems = []
    for row in rows:
        for key in NUMERIC_REQUIRED:
            value = row.get(key)
            if number(value) and (value != value or value in (float("inf"), float("-inf"))):
                problems.append(f"step {row.get('step')}: {key}={value}")
    for row in rows:
        for key in ("ea/weight_mean_chosen", "ea/weight_mean_rejected"):
            value = row.get(key)
            if number(value) and abs(value - 1.0) > 1e-6:
                problems.append(f"step {row.get('step')}: {key}={value:.6f} (should be 1)")
                break

    def series(key, wanted=None):
        return [(row["step"], row[key]) for row in rows if number(row.get(key)) and (wanted is None or wanted in row)]

    train_loss = series("loss")
    if len(train_loss) >= 2:
        head = sum(v for _, v in train_loss[:5]) / len(train_loss[:5])
        tail = sum(v for _, v in train_loss[-5:]) / len(train_loss[-5:])
        print(f"\ntrain loss: first5={head:.4f}  last5={tail:.4f}  delta={tail - head:+.4f}")
    for key in ("eval_loss", "rewards/accuracies", "eval_rewards/accuracies", "ea/weight_max_chosen"):
        points = series(key)
        if len(points) >= 2:
            print(f"{key}: {points[0][1]:.4g} (step {points[0][0]}) -> {points[-1][1]:.4g} (step {points[-1][0]})")

    print("\nhealth check")
    if problems:
        for line in problems[:10]:
            print(f"  FAIL {line}")
        if len(problems) > 10:
            print(f"  ... {len(problems) - 10} more")
    else:
        print("  OK: no NaN/inf in loss or grad_norm; EA-DPO weight_mean stays at 1")

    if args.csv:
        keys = sorted({key for row in rows for key in row if isinstance(row[key], (int, float))})
        with open(args.csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key) for key in keys})
        print(f"\ncsv       : {args.csv} ({len(rows)} rows, {len(keys)} columns)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
