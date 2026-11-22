#!/usr/bin/env python3
"""Evaluate adapters on a fixed preference set using the trainer's fused LM head.

EA-DPO keeps its native weighted training objective. This tool reports additional
unweighted implicit rewards. Historical trainer metrics are shown as context.
The output records sample IDs, tokenized-input hashes, and per-pair scores.
"""

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trl-1.18.1"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_adapter(value):
    """`路径:标签` -> (路径, 标签)；只给路径时用目录名。"""
    if ":" in value and not Path(value).exists():
        path, label = value.rsplit(":", 1)
        return path, label
    return value, Path(value).name or value


def logged_win_rate(adapter_dir):
    """Return the latest evaluation accuracy from historical trainer logs."""
    base = Path(adapter_dir)
    found = []
    root_state = base / "trainer_state.json"
    if root_state.is_file():
        found.append((float("inf"), root_state))
    if base.is_dir():
        for child in base.glob("checkpoint-*"):
            suffix = child.name[len("checkpoint-") :]
            state = child / "trainer_state.json"
            if suffix.isdigit() and state.is_file():
                found.append((int(suffix), state))
    if not found:
        return None
    state_path = max(found)[1]
    try:
        history = json.loads(state_path.read_text(encoding="utf-8")).get("log_history", [])
    except (OSError, ValueError):
        return None
    values = [float(row["eval_rewards/accuracies"]) for row in history
              if isinstance(row.get("eval_rewards/accuracies"), (int, float))
              and not isinstance(row["eval_rewards/accuracies"], bool)]
    return values[-1] if values else None


def as_messages(field, role):
    """把 prompt/chosen/rejected 统一成消息列表。"""
    if isinstance(field, str):
        return [{"role": role, "content": field}]
    if isinstance(field, list):
        out = []
        for item in field:
            if isinstance(item, dict):
                out.append({"role": item.get("role") or role, "content": str(item.get("content") or "")})
            elif isinstance(item, str):
                out.append({"role": role, "content": item})
        return out
    raise TypeError(f"无法识别的对话字段类型：{type(field)}")


def chat_ids(tokenizer, messages, add_generation_prompt=False):
    """拿到对话模板渲染后的 token id 列表。

    transformers 5.x 的 `apply_chat_template(tokenize=True)` 返回 `BatchEncoding`
    （旧版返回 `list[int]`），这里统一成扁平的一维 id 列表。
    """
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=True)
    ids = out["input_ids"] if isinstance(out, Mapping) else out
    if ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    return list(ids)


def build_pair(tokenizer, example, max_length):
    """返回 [(ids, first_completion_index)]，顺序是 [chosen, rejected]；超长返回 None。"""
    prompt = as_messages(example["prompt"], "user")
    prompt_ids = chat_ids(tokenizer, prompt, add_generation_prompt=True)
    full_sequences = []
    for key in ("chosen", "rejected"):
        full_ids = chat_ids(tokenizer, prompt + as_messages(example[key], "assistant"))
        full_sequences.append(full_ids)
    starts = []
    for full_ids in full_sequences:
        start = 0
        while start < min(len(full_ids), len(prompt_ids)) and full_ids[start] == prompt_ids[start]:
            start += 1
        starts.append(start)
    # The trainer uses one common boundary for both completions.
    start = min(starts)
    if start == 0 or start >= max_length:
        return None
    built = []
    for full_ids in full_sequences:
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]  # keep_start，与训练配置相同
        if len(full_ids) <= start:
            return None  # 截断后没有 completion Token
        built.append((full_ids, start))
    return built


def completion_rows(logits_len, length, start):
    """logits 的哪些行对应"预测 completion Token"。

    `logits_to_keep=k` 生效时只保留最后 k 个位置（k = length-start+1）；
    没生效时是整条序列。两种情况下要取的行数都正好是 length-start，
    单独抽出来是为了能在没有 torch 的机器上验证这个索引。
    """
    if logits_len == length - start + 1:
        return slice(0, -1)
    return slice(start - 1, -1)


def sequence_logprob(torch, model, ids, start, chunk_size=128):
    """completion 部分（从 start 开始）的逐 Token log-prob 之和。不加权，与 DPO 原始定义一致。"""
    input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    labels[:, :start] = -100
    with torch.no_grad(), torch.autocast(model.device.type, dtype=torch.bfloat16):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                    use_cache=False, fused_lm_head=True)
        return out.log_probs.sum().item()


def score(torch, model, pairs, batch_size=2):
    """对一批已构建好的样本打分，返回逐对的 dict 列表。"""
    from trl.trainer.dpo_trainer import DataCollatorForPreference

    collator = DataCollatorForPreference(pad_token_id=model.config.pad_token_id)
    rows = []
    for offset in range(0, len(pairs), batch_size):
        group = pairs[offset:offset + batch_size]
        examples = [{"prompt_ids": c[:start], "chosen_ids": c[start:], "rejected_ids": r[start:]}
                    for (c, start), (r, _) in group]
        batch = {key: value.to(model.device) for key, value in collator(examples).items()}
        mask = batch.pop("completion_mask")
        labels = batch["input_ids"].masked_fill(mask == 0, -100)
        with torch.no_grad(), torch.autocast(model.device.type, dtype=torch.bfloat16):
            values = model(**batch, labels=labels, use_cache=False, fused_lm_head=True).log_probs.sum(1)
        chosen_values, rejected_values = values.chunk(2)
        for i, ((c, start), (r, _)) in enumerate(group):
            rows.append({"chosen": chosen_values[i].item(), "rejected": rejected_values[i].item(),
                         "chosen_len": len(c) - start, "rejected_len": len(r) - start})
    return rows


def summarize(results, ref_rows, beta):
    """policy 打分 + 参考打分 -> 隐式奖励与胜率。"""
    chosen_rewards, rejected_rewards, margins, wins = [], [], [], []
    for row, ref in zip(results, ref_rows, strict=True):
        rc = beta * (row["chosen"] - ref["chosen"])
        rr = beta * (row["rejected"] - ref["rejected"])
        margin = rc - rr
        chosen_rewards.append(rc)
        rejected_rewards.append(rr)
        margins.append(margin)
        wins.append(1.0 if margin > 0 else 0.0)
    n = max(len(wins), 1)
    return {
        "pairs": len(wins),
        "win_rate": sum(wins) / n,
        "margin_mean": sum(margins) / n,
        "reward_chosen_mean": sum(chosen_rewards) / n,
        "reward_rejected_mean": sum(rejected_rewards) / n,
        "chosen_len_mean": sum(row["chosen_len"] for row in results) / n,
        "rejected_len_mean": sum(row["rejected_len"] for row in results) / n,
        "policy_chosen_logprob_mean": sum(row["chosen"] for row in results) / n,
        "ref_chosen_logprob_mean": sum(row["chosen"] for row in ref_rows) / n,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", action="append", required=True, metavar="路径[:标签]",
                        help="adapter 目录（trainer 保存的 output_dir），可重复")
    parser.add_argument("--base", default=None, help="基座模型，默认读取 adapter_config.json")
    parser.add_argument("--data", default="data/uf-subset", help="落盘的数据集目录")
    parser.add_argument("--split", default="test", help="用哪个划分（默认 test，512 对固定评测集）")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 对（0 = 全部）")
    parser.add_argument("--max-length", type=int, default=1024, help="与训练配置一致")
    parser.add_argument("--beta", type=float, default=0.1, help="与训练配置一致；只影响 margin 的量纲")
    parser.add_argument("--chunk", type=int, default=128, help="log_softmax 的分块大小")
    parser.add_argument("--batch-size", type=int, default=2, help="与训练验证 batch 一致")
    parser.add_argument("--out", default="", help="把结果写成 JSON")
    args = parser.parse_args()

    if args.limit < 0 or args.batch_size <= 0:
        parser.error("limit must be nonnegative and batch-size must be positive")
    arms = [parse_adapter(value) for value in args.adapter]
    for path, label in arms:
        if not Path(path).is_dir():
            print(f"[fail] adapter 目录不存在：{path}（{label}）", file=sys.stderr)
            return 1
    if not Path(args.data).is_dir():
        print(f"[fail] 数据目录不存在：{args.data}，先跑 python scripts/prepare_data.py", file=sys.stderr)
        return 1
    if args.base is None:
        adapter_config = json.loads((Path(arms[0][0]) / "adapter_config.json").read_text(encoding="utf-8"))
        args.base = adapter_config["base_model_name_or_path"]

    import torch
    from datasets import load_from_disk
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.trainer.utils import add_fused_lm_head

    if not torch.cuda.is_available():
        print("[warn] CUDA 不可用，将在 CPU 上跑（会非常慢）")

    print(f"[1/4] 载入 tokenizer 与基座 {args.base} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/4] 构建样本（{args.data} / {args.split}）...")
    dataset = load_from_disk(args.data)[args.split]
    dataset_size = len(dataset)
    dataset_fingerprint = dataset._fingerprint
    if args.limit:
        dataset = dataset.select(range(min(args.limit, dataset.num_rows)))
    pairs, pair_indices, skipped = [], [], 0
    for index, example in enumerate(dataset):
        built = build_pair(tokenizer, example, args.max_length)
        if built is None:
            skipped += 1
            continue
        pairs.append(built)
        pair_indices.append(index)
    print(f"      可用 {len(pairs)} 对，跳过 {skipped} 对（prompt 过长 / 截断后无 completion）")
    if not pairs:
        print("[fail] 没有可用样本", file=sys.stderr)
        return 1

    print(f"[3/4] 基座模型作为参考，先算一遍 ref log-prob ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.pad_token_id = tokenizer.pad_token_id
    add_fused_lm_head(model, outputs=("log_probs",))
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    ref_rows = score(torch, model, pairs, args.batch_size)
    print(f"      ref chosen log-prob 均值 {sum(r['chosen'] for r in ref_rows) / len(ref_rows):.2f}")

    print("[4/4] 逐个 adapter 打分 ...")
    table = []
    per_pair = {}
    for path, label in arms:
        print(f"  --- {label}  ({path}) ---")
        peft_model = PeftModel.from_pretrained(model, path)
        peft_model.eval()
        rows = score(torch, peft_model, pairs, args.batch_size)
        summary = summarize(rows, ref_rows, args.beta)
        summary["label"] = label
        summary["adapter"] = path
        summary["logged_win_rate"] = logged_win_rate(path)
        table.append(summary)
        per_pair[label] = [
            {"pair_index": index,
             "policy_chosen": row["chosen"], "policy_rejected": row["rejected"],
             "ref_chosen": ref["chosen"], "ref_rejected": ref["rejected"],
             "reward_chosen": args.beta * (row["chosen"] - ref["chosen"]),
             "reward_rejected": args.beta * (row["rejected"] - ref["rejected"]),
             "margin": args.beta * ((row["chosen"] - ref["chosen"]) - (row["rejected"] - ref["rejected"]))}
            for index, row, ref in zip(pair_indices, rows, ref_rows, strict=True)
        ]
        model = peft_model.unload()  # 还原成基座，供下一个 adapter 使用
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"      win_rate={summary['win_rate']:.6f} margin={summary['margin_mean']:+.6f}")

    print("\nAdditional unweighted preference evaluation")
    for row in table:
        print(f"{row['label']}: {row['win_rate']:.6f}, margin={row['margin_mean']:+.6f}, "
              f"pairs={row['pairs']}, historical_log_accuracy={row['logged_win_rate']}")
    print("Historical logs may use different samples, checkpoints, numerical paths, or weighting.")
    print("Chosen/rejected lengths describe the scored dataset responses.")

    if args.out:
        payload = {"base": args.base, "data": args.data, "split": args.split,
                   "max_length": args.max_length, "beta": args.beta,
                   "pairs": table[0]["pairs"], "skipped": skipped, "arms": table,
                   "dataset_size": dataset_size, "dataset_fingerprint": dataset_fingerprint,
                   "limit": args.limit, "batch_size": args.batch_size,
                   "pair_indices": pair_indices, "scoring": "trainer_fused_lm_head_bfloat16",
                   "tokenized_pairs_sha256": hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
                   "per_pair": per_pair}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
