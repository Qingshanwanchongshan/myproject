"""Run one resumable sweep segment and export exact per-pair validation metrics."""

import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--experiment", type=Path, required=True)
parser.add_argument("--arm", required=True)
parser.add_argument("--segment-end", type=int, required=True)
parser.add_argument("--validation-limit", type=int, default=0)
segment, remaining = parser.parse_known_args()
project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project / "trl-1.18.1"))

import torch
import trl
import trl.trainer.dpo_trainer as dpo_module
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainerCallback
from transformers.integrations import TensorBoardCallback


def summarize(rows, name):
    count = len(rows)
    chosen = [row[name]["chosen"] for row in rows]
    rejected = [row[name]["rejected"] for row in rows]
    margins = [c - r for c, r in zip(chosen, rejected, strict=True)]
    return {
        "pairs": count,
        "wins": sum(value > 0 for value in margins),
        "ties": sum(value == 0 for value in margins),
        "accuracy": sum(value > 0 for value in margins) / count,
        "chosen": sum(chosen) / count,
        "rejected": sum(rejected) / count,
        "margin": sum(margins) / count,
        "loss": torch.nn.functional.softplus(-torch.tensor(margins, dtype=torch.float64)).mean().item(),
    }


class SegmentControl(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 100:
            control.should_evaluate = True
        if (state.global_step >= segment.segment_end
                or (segment.experiment / "STOP").exists()
                or (segment.experiment / f"STOP-{segment.arm}").exists()):
            control.should_training_stop = True
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        progress = {
            "step": state.global_step,
            "planned_steps": state.max_steps,
            "total_train_tokens": self.trainer._total_train_tokens,
            "arm": segment.arm,
        }
        checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        (checkpoint / "sweep_progress.json").write_text(json.dumps(progress))
        path = segment.experiment / f"{segment.arm}-progress.json"
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(progress))
        pending.replace(path)
        return control


class SweepTrainer(trl.DPOTrainer):
    def __init__(self, *args, **kwargs):
        dataset = kwargs["eval_dataset"]
        dataset = dataset.add_column("sweep_pair_index", list(range(len(dataset))))
        if segment.validation_limit:
            dataset = dataset.select(range(min(segment.validation_limit, len(dataset))))
        kwargs["eval_dataset"] = dataset
        super().__init__(*args, **kwargs)
        self.remove_callback(TensorBoardCallback)
        self.add_callback(TensorBoardCallback(tb_writer=SummaryWriter(
            str(segment.experiment / "tensorboard" / segment.arm)
        )))
        self.add_callback(SegmentControl(self))
        checkpoints = sorted(
            Path(self.args.output_dir).glob("checkpoint-*"),
            key=lambda path: int(path.name.split("-")[-1]),
        )
        if checkpoints:
            saved = json.loads((checkpoints[-1] / "sweep_progress.json").read_text())
            self._total_train_tokens = saved["total_train_tokens"]
        self._validation_rows = []

    def _compute_loss(self, model, inputs, return_outputs):
        mode = "train" if self.model.training else "eval"
        forwards = []
        captured_weights = []
        original_weights = dpo_module.token_weights

        def observe_weights(*args, **kwargs):
            weights = original_weights(*args, **kwargs)
            captured_weights.append(weights)
            return weights

        def observe_output(module, args, output):
            forwards.append(output.log_probs.detach())

        metrics = self._metrics[mode]
        hook = self.model.register_forward_hook(observe_output)
        dpo_module.token_weights = observe_weights
        try:
            loss = super()._compute_loss(model, inputs, return_outputs)
        finally:
            hook.remove()
            dpo_module.token_weights = original_weights
        if len(forwards) != 2 or len(captured_weights) != 1:
            raise RuntimeError("Expected one EA weight vector and policy/reference forwards.")

        with torch.no_grad():
            policy, reference = forwards
            weights = captured_weights[0]
            weighted_policy = (policy.float() * weights).sum(1)
            weighted_reference = (reference.float() * weights).sum(1)
            chosen, rejected = (self.beta * (weighted_policy - weighted_reference)).chunk(2)
            margin = chosen - rejected
            standard_chosen, standard_rejected = (
                self.beta * (policy.sum(1) - reference.sum(1))
            ).chunk(2)
            standard_margin = standard_chosen - standard_rejected
            metrics["unweighted_rewards/accuracies"].extend((standard_margin > 0).float().cpu().tolist())
            metrics["unweighted_rewards/margins"].extend(standard_margin.cpu().tolist())
            metrics["unweighted_loss"].extend(torch.nn.functional.softplus(-standard_margin).cpu().tolist())
            metrics["cumulative_num_tokens"] = [self._total_train_tokens]
            if mode == "train":
                metrics["gpu_peak_gib"].append(torch.cuda.max_memory_allocated() / 2**30)
                return loss

            half = chosen.numel()
            offset = len(self._validation_rows)
            pair_ids = self.eval_dataset["sweep_pair_index"]
            for index in range(half):
                sequences = []
                for position in (index, index + half):
                    valid = inputs["attention_mask"][position].bool()
                    sequences.append(inputs["input_ids"][position][valid].cpu().tolist())
                self._validation_rows.append({
                    "pair_index": int(pair_ids[offset + index]),
                    "tokens_sha256": hashlib.sha256(json.dumps(sequences).encode()).hexdigest(),
                    "weighted": {"chosen": chosen[index].item(), "rejected": rejected[index].item()},
                    "unweighted": {
                        "chosen": standard_chosen[index].item(),
                        "rejected": standard_rejected[index].item(),
                    },
                    "max_weight": {
                        "chosen": weights[index].max().item(),
                        "rejected": weights[index + half].max().item(),
                    },
                })
                if self.args.ea_top_fraction < 1:
                    row = self._validation_rows[-1]
                    completion_masks = inputs["completion_mask"][:, 1:].bool()
                    positions = {"chosen": index, "rejected": index + half}
                    row["completion_lengths"] = {
                        side: int(completion_masks[position].sum()) for side, position in positions.items()
                    }
                    row["selected_completion_indices"] = {
                        side: (weights[position][completion_masks[position]] > 0).nonzero(as_tuple=True)[0].cpu().tolist()
                        for side, position in positions.items()
                    }
        return loss

    def evaluate(self, *args, **kwargs):
        self._validation_rows = []
        metrics = super().evaluate(*args, **kwargs)
        rows = self._validation_rows
        if len(rows) != len(self.eval_dataset):
            raise RuntimeError("Validation did not score every effective pair exactly once.")
        payload = {
            "arm": segment.arm, "step": self.state.global_step,
            "ea_top_fraction": self.args.ea_top_fraction,
            "scoring": "trainer_fused_lm_head", "batch_size": self.args.per_device_eval_batch_size,
            "dataset_fingerprint": self.eval_dataset._fingerprint,
            "pairs": len(rows), "weighted": summarize(rows, "weighted"),
            "unweighted": summarize(rows, "unweighted"), "per_pair": rows,
        }
        for name in ("chosen", "rejected"):
            payload[f"mean_max_weight_{name}"] = sum(row["max_weight"][name] for row in rows) / len(rows)
        path = segment.experiment / segment.arm / f"evaluation-{self.state.global_step}.json"
        pending = path.with_suffix(".tmp")
        pending.write_text(json.dumps(payload, indent=2, allow_nan=False))
        pending.replace(path)
        return metrics


trl.DPOTrainer = SweepTrainer
sys.argv = ["train_dpo.py", *remaining]
runpy.run_path(str(project / "train_dpo.py"), run_name="__main__")
