"""Export current sweep progress, exact metrics, and same-step leaderboards."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def summarize_experiment(root):
    manifest = json.loads((root / "experiment.json").read_text())
    rows = []
    signature = None
    for arm in manifest["trials"]:
        config = manifest["trials"][arm]
        progress_path = root / f"{arm}-progress.json"
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
        evaluations = list((root / arm).glob("evaluation-*.json"))
        row = {"arm": arm, "completed_step": progress.get("step", 0), "changes": config}
        if evaluations:
            path = max(evaluations, key=lambda p: int(p.stem.split("-")[-1]))
            data = json.loads(path.read_text())
            pair_inputs = [(sample["pair_index"], sample["tokens_sha256"]) for sample in data["per_pair"]]
            current = hashlib.sha256(json.dumps(pair_inputs).encode()).hexdigest()
            if signature is None:
                signature = current
            if signature != current:
                raise ValueError("Trials evaluated different preference pairs or tokenized inputs.")
            row.update(
                evaluation_step=data["step"], pairs=data["pairs"],
                weighted_accuracy=data["weighted"]["accuracy"],
                unweighted_accuracy=data["unweighted"]["accuracy"],
                weighted_loss=data["weighted"]["loss"],
                unweighted_loss=data["unweighted"]["loss"],
                weighted_margin=data["weighted"]["margin"],
                unweighted_margin=data["unweighted"]["margin"],
                rejected_mean_max_weight=data["mean_max_weight_rejected"],
            )
        rows.append(row)
    stages = sorted({row["evaluation_step"] for row in rows if "evaluation_step" in row})
    leaderboards = {
        str(step): {
            "weighted": [row["arm"] for row in sorted(
                (row for row in rows if row.get("evaluation_step") == step),
                key=lambda row: row["weighted_accuracy"], reverse=True,
            )],
            "unweighted": [row["arm"] for row in sorted(
                (row for row in rows if row.get("evaluation_step") == step),
                key=lambda row: row["unweighted_accuracy"], reverse=True,
            )],
        }
        for step in stages
    }
    baseline_path = Path(manifest["baseline"]) / "checkpoint-3753/trainer_state.json"
    baseline = [
        row for row in json.loads(baseline_path.read_text())["log_history"] if "eval_loss" in row
    ]
    result = {
        "trials": rows, "leaderboards_by_evaluation_step": leaderboards,
        "tokenized_validation_sha256": signature,
        "historical_dpo": [
            {"step": row["step"], "accuracy": row["eval_rewards/accuracies"],
             "loss": row["eval_loss"], "margin": row["eval_rewards/margins"],
             "aggregation": "historical batch mean"}
            for row in baseline
        ],
    }
    pending = root / "results.tmp"
    pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pending.replace(root / "results.json")
    columns = [
        "arm", "completed_step", "evaluation_step", "pairs",
        "weighted_accuracy", "unweighted_accuracy", "weighted_loss", "unweighted_loss",
        "weighted_margin", "unweighted_margin", "rejected_mean_max_weight",
    ]
    with (root / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=Path)
    arguments = parser.parse_args()
    summarize_experiment(arguments.experiment)
