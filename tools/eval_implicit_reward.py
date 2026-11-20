#!/usr/bin/env python3
"""给多个 adapter 用**同一把尺子**打分：标准 DPO 隐式奖励胜率，并量化口径差。

两者的形式其实一样
------------------
标准 DPO 和 EA-DPO 报的都是同一种量 —— 隐式奖励之差：

    r(y)      = β · Σ_t [ log π(y_t) − log π_ref(y_t) ]          （基线：权重全 1）
    r_EA(y)   = β · Σ_t w_t · [ log π(y_t) − log π_ref(y_t) ]    （EA：w_t 样本自适应）

差别只在权重向量，而且权重均值恒为 1（Σ w_t = T），所以两者的**量纲和尺度是同一个**，
数值高度相关，方向上也是一致的。这正是论文里能并排比较不同 DPO 变体的原因。

那到底差在哪里
--------------
w_t 由**样本自身的**熵与注意力因果入度算出来，所以它是 π 的函数。这意味着

    "EA 的 win_rate" = f_EA(π)      "基线的 win_rate" = f_dpo(π)

是两个不同的函数。同一个模型换一把尺子，数会变；两个模型各用各的尺子比，口径差就混进了
方法差里。口径差通常只有 1-3 个百分点 —— 但一篇方法论文要证明的效应量也常常就是 1-2 个
百分点，两者同量级，所以才值得fix：用第三把尺子（下面这个不加权的定义）同时量两个模型。

这个脚本做的事
--------------
在**同一批固定偏好对**上，对每个 adapter 重算一遍**不加权**的 DPO 隐式奖励：

    r_chosen   = β · Σ_t [ log π(y_t) − log π_ref(y_t) ]
    r_rejected = β · Σ_t [ log π(y_t) − log π_ref(y_t) ]
    win_rate   = P(r_chosen > r_rejected)

数据、β、参考模型、分词与截断规则对每个 adapter 完全相同 —— 这才是可以横向比的数，
也是 DPO 论文里报告的那个量。

更进一步：如果 adapter 目录里留着 `trainer_state.json`，脚本会把该组**训练日志里自己的
加权口径**一并列出来。两列相减，就是这一组的口径差有多大 —— 不用猜，直接量。

用法
----
    # 两个 adapter 一起评（推荐，省一次 ref 前向）
    python tools/eval_implicit_reward.py \
      --adapter outputs/dpo-3b-20pct:DPO \
      --adapter outputs/ea-3b-20pct:EA-DPO

    # 只评一个
    python tools/eval_implicit_reward.py --adapter outputs/ea-3b-20pct:EA-DPO --limit 256

    # 结果落盘，方便两台机器/两轮实验合起来看
    python tools/eval_implicit_reward.py --adapter a:DPO --adapter b:EA-DPO --out logs/ir.json

耗时：256 对、两个 adapter、4090 上约 6-8 分钟（每对每条序列一次前向）。

注意
----
* 这不是"生成质量"评测。它衡量的是模型对偏好对的分辨力，和人类/裁判的偏好一致率
  不是一回事。要下"EA-DPO 更好"的结论，还得做固定 prompt 的生成 + 裁判判定。
* 参考模型 = 基座模型（关掉 adapter），与训练时 `ref_model=None` + LoRA 的做法一致。
* 自检：DPO 那一行的 win_rate 应该和它训练日志里最后一个 `eval_rewards/accuracies` 很接近
  （同一把尺子）。差很多说明数据子集或截断规则和训练时不一致。
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_adapter(value):
    """`路径:标签` -> (路径, 标签)；只给路径时用目录名。"""
    if ":" in value and not Path(value).exists():
        path, label = value.rsplit(":", 1)
        return path, label
    return value, Path(value).name or value


def logged_win_rate(adapter_dir):
    """从 adapter 目录（或它的 checkpoint-*）里的 trainer_state.json 取该组自己记录的胜率。

    这是**该组自己的口径**（EA 组是加权后的），拿它和本脚本算的不加权数相减，
    就是这个 adapter 的口径差有多大。取不到返回 None。
    """
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
    values = []
    for row in history:
        for key in ("eval_rewards/accuracies", "rewards/accuracies"):
            value = row.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(float(value))
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


def build_pair(tokenizer, example, max_length):
    """返回 [(ids, first_completion_index)]，顺序是 [chosen, rejected]；超长返回 None。"""
    prompt = as_messages(example["prompt"], "user")
    prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True)
    if len(prompt_ids) >= max_length:
        return None  # 与训练器一致：prompt 就占满长度的样本直接丢弃

    built = []
    for key in ("chosen", "rejected"):
        full_ids = tokenizer.apply_chat_template(prompt + as_messages(example[key], "assistant"), tokenize=True)
        if full_ids[: len(prompt_ids)] != prompt_ids:
            raise SystemExit(
                "chat template 前后缀不一致：prompt 单独分词与拼接分词结果不同，无法定位 completion 起点。"
            )
        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]  # keep_start，与训练配置相同
        if len(full_ids) <= len(prompt_ids):
            return None  # 截断后没有 completion Token
        built.append((full_ids, len(prompt_ids)))
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
    length = len(ids)
    with torch.no_grad():
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False,
                        logits_to_keep=length - start + 1)
        except TypeError:  # 该模型/版本不支持 logits_to_keep
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits
        logits = logits[:, completion_rows(logits.shape[1], length, start)]
        targets = input_ids[:, start:]
        total = 0.0
        for offset in range(0, targets.shape[1], chunk_size):
            piece = logits[:, offset : offset + chunk_size].float().log_softmax(-1)
            target = targets[:, offset : offset + chunk_size].unsqueeze(-1)
            total += piece.gather(-1, target).squeeze(-1).sum().item()
        del out, logits
    return total


def score(torch, model, pairs):
    """对一批已构建好的样本打分，返回逐对的 dict 列表。"""
    rows = []
    for chosen, rejected in pairs:
        row = {}
        for name, (ids, start) in (("chosen", chosen), ("rejected", rejected)):
            row[name] = sequence_logprob(torch, model, ids, start)
            row[f"{name}_len"] = len(ids) - start
        rows.append(row)
    return rows


def summarize(results, ref_rows, beta):
    """policy 打分 + 参考打分 -> 隐式奖励与胜率。"""
    chosen_rewards, rejected_rewards, margins, wins = [], [], [], []
    for row, ref in zip(results, ref_rows):
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
    parser.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct", help="基座模型")
    parser.add_argument("--data", default="data/uf-20pct", help="落盘的数据集目录")
    parser.add_argument("--split", default="test", help="用哪个划分（默认 test，512 对固定评测集）")
    parser.add_argument("--limit", type=int, default=0, help="只用前 N 对（0 = 全部）")
    parser.add_argument("--max-length", type=int, default=1024, help="与训练配置一致")
    parser.add_argument("--beta", type=float, default=0.1, help="与训练配置一致；只影响 margin 的量纲")
    parser.add_argument("--chunk", type=int, default=128, help="log_softmax 的分块大小")
    parser.add_argument("--out", default="", help="把结果写成 JSON")
    args = parser.parse_args()

    arms = [parse_adapter(value) for value in args.adapter]
    for path, label in arms:
        if not Path(path).is_dir():
            print(f"[fail] adapter 目录不存在：{path}（{label}）", file=sys.stderr)
            return 1
    if not Path(args.data).is_dir():
        print(f"[fail] 数据目录不存在：{args.data}，先跑 python scripts/prepare_data.py", file=sys.stderr)
        return 1

    import torch
    from datasets import load_from_disk
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("[warn] CUDA 不可用，将在 CPU 上跑（会非常慢）")

    print(f"[1/4] 载入 tokenizer 与基座 {args.base} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[2/4] 构建样本（{args.data} / {args.split}）...")
    dataset = load_from_disk(args.data)[args.split]
    if args.limit:
        dataset = dataset.select(range(min(args.limit, dataset.num_rows)))
    pairs, skipped = [], 0
    for example in dataset:
        built = build_pair(tokenizer, example, args.max_length)
        if built is None:
            skipped += 1
            continue
        pairs.append(built)
    print(f"      可用 {len(pairs)} 对，跳过 {skipped} 对（prompt 过长 / 截断后无 completion）")
    if not pairs:
        print("[fail] 没有可用样本", file=sys.stderr)
        return 1

    print(f"[3/4] 基座模型作为参考，先算一遍 ref log-prob ...")
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    ref_rows = score(torch, model, pairs)
    print(f"      ref chosen log-prob 均值 {sum(r['chosen'] for r in ref_rows) / len(ref_rows):.2f}")

    print("[4/4] 逐个 adapter 打分 ...")
    table = []
    for path, label in arms:
        print(f"  --- {label}  ({path}) ---")
        peft_model = PeftModel.from_pretrained(model, path)
        peft_model.eval()
        rows = score(torch, peft_model, pairs)
        summary = summarize(rows, ref_rows, args.beta)
        summary["label"] = label
        summary["adapter"] = path
        summary["logged_win_rate"] = logged_win_rate(path)
        table.append(summary)
        model = peft_model.unload()  # 还原成基座，供下一个 adapter 使用
        model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        extra = ""
        if summary["logged_win_rate"] is not None:
            extra = f"  日志口径={summary['logged_win_rate']:.4f}（差 {summary['win_rate'] - summary['logged_win_rate']:+.4f}）"
        print(f"      win_rate={summary['win_rate']:.4f}  margin={summary['margin_mean']:+.4f}  "
              f"长度={summary['chosen_len_mean']:.0f}/{summary['rejected_len_mean']:.0f}{extra}")

    # ---------------- 汇总 ----------------
    width = max(len(row["label"]) for row in table) + 2
    has_logged = any(row.get("logged_win_rate") is not None for row in table)
    print()
    print("=" * 88)
    print("同口径对比（标准 DPO 隐式奖励，不加权 —— 这一列才是能横向比的）")
    print("=" * 88)
    header = (f"{'adapter'.ljust(width)}" + f"{'win_rate':>10}" + f"{'margin':>10}"
              + f"{'r_chosen':>10}" + f"{'r_rejected':>11}" + f"{'长度(chosen)':>13}")
    if has_logged:
        header += f"{'日志口径':>10}" + f"{'口径差':>9}"
    print(header)
    print("-" * len(header))
    for row in table:
        line = (f"{row['label'].ljust(width)}"
                f"{row['win_rate']:>10.4f}{row['margin_mean']:>+10.4f}"
                f"{row['reward_chosen_mean']:>10.4f}{row['reward_rejected_mean']:>+11.4f}"
                f"{row['chosen_len_mean']:>13.0f}")
        if has_logged:
            logged = row.get("logged_win_rate")
            if logged is None:
                line += f"{'-':>10}{'-':>9}"
            else:
                line += f"{logged:>10.4f}{row['win_rate'] - logged:>+9.4f}"
        print(line)
    print()
    print(f"样本 {table[0]['pairs']} 对，β={args.beta}，max_length={args.max_length}")
    if len(table) > 1:
        best = max(table, key=lambda row: row["win_rate"])
        spread = max(row["win_rate"] for row in table) - min(row["win_rate"] for row in table)
        print(f"最高 win_rate：{best['label']}（{best['win_rate']:.4f}），两组相差 {spread:+.4f}")
        print("样本量 %d 时胜率的抽样标准差约 %.4f；相差小于它就别急着下结论。"
              % (table[0]["pairs"], (0.25 / table[0]["pairs"]) ** 0.5))
        if has_logged:
            gaps = [abs(row["win_rate"] - row["logged_win_rate"]) for row in table
                    if row.get("logged_win_rate") is not None]
            if gaps:
                print(f"“口径差”列的绝对值最大 {max(gaps):.4f}：这就是各用各的尺子时，"
                      f"最多有多少是口径而不是模型。")
                print("差得比你要测的效应还大时，就该用左边那一列比；差得可忽略时，直接看日志也够。")
    print("""
这份数字怎么读
  1. 左半边（win_rate / margin / r_*）对每个 adapter 用的是同一个定义 —— 这是能横向比的前提
  2. "日志口径"是该组训练时自己记录的胜率；EA 组是加权后的，所以两列不相等
     "口径差"= 本脚本不加权数 − 该组日志加权数，量出每组的尺子差了多少
  3. 它衡量"模型对偏好对的分辨力"，不等于生成质量；要下结论还需要固定 prompt 的生成 + 裁判判定
  4. chosen 长度顺带看一眼：DPO 最常见的副作用就是回答变长（长度 reward hacking）
""")

    if args.out:
        payload = {"base": args.base, "data": args.data, "split": args.split,
                   "max_length": args.max_length, "beta": args.beta,
                   "pairs": table[0]["pairs"], "skipped": skipped, "arms": table}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"JSON 已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
