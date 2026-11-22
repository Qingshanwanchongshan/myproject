#!/usr/bin/env python3
"""把 ultrafeedback_binarized 固定成两个本地数据集，供两组实验共用。

产出（默认位于仓库根目录）：

    data/uf-subset/  train 30500 对（全量 61135 的 50%）/ test 512 对   ← 第一轮用
    data/uf-full/    train 61135 对（全量 1 epoch）      / test 512 对   ← 第二轮用

两个目录里的 test 是**同一批 512 条**（同一个 seed、同一个 shuffle），所以第一轮
50% 数据和第二轮全量数据的评测曲线可以直接对比。

**改数据量只改这一个参数**：`--train-size`。配置文件里是 `max_steps: -1` +
`num_train_epochs: 1`，步数会自动等于"子集大小 / 8"，不用同步改配置。

为什么要落盘而不是直接用 HF 上的原始数据：
  * 两组训练看到的数据顺序必须完全一致，否则差异分不清是方法还是数据；
  * 离线后不受网络影响，重启实例不会重新下载；
  * eval 从 2000 条降到 512 条，一次评测约 3-4 分钟（4090），十几分钟的浪费省掉。

用法：
    python scripts/prepare_data.py                      # 默认 30500 对（50%）
    python scripts/prepare_data.py --train-size 12000   # 换回 20%
    python scripts/prepare_data.py --force               # 重新生成
    python scripts/prepare_data.py --skip-full           # 只生成第一轮需要的数据

国内机器建议先设镜像（只需一次）：
    export HF_ENDPOINT=https://hf-mirror.com
"""

import argparse
import os
import sys
from pathlib import Path

# 中文输出在 Windows GBK 控制台上不至于直接崩掉；Linux 上是空操作。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def text_of(field) -> str:
    """把 prompt/chosen/rejected 统一成文本：可能是 str，也可能是 [{"role": ..., "content": ...}]。"""
    if isinstance(field, str):
        return field.strip()
    if isinstance(field, list):
        parts = []
        for item in field:
            if isinstance(item, dict):
                parts.append(str(item.get("content") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts).strip()
    return ""


def as_messages(field, role) -> list:
    """把字符串 / 消息列表统一成 [{"role": ..., "content": ...}]（对话格式）。"""
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
    return [{"role": role, "content": str(field)}]


def load_source(args, load_dataset):
    """返回含 train/test 的 DatasetDict，列统一成 prompt / chosen / rejected（对话格式）。

    三种来源：
      * 本地 train.parquet / test.parquet
      * 本地 train.json / test.json（魔搭上的 llamafactory 版是 instruction/chosen/rejected 纯字符串，
        需要转成对话格式，否则 TRL 会走非对话分支、不加 chat template，token 序列就和官方版不一致了）
      * HuggingFace Hub 上的数据集名
    """
    if not args.local_dir:
        return load_dataset(args.dataset)

    base = Path(args.local_dir)
    if (base / "train.parquet").is_file():
        return load_dataset("parquet", data_files={
            "train": str(base / "train.parquet"), "test": str(base / "test.parquet")})
    if not (base / "train.json").is_file():
        raise FileNotFoundError(f"{base} 里既没有 train.parquet 也没有 train.json")

    raw = load_dataset("json", data_files={"train": str(base / "train.json"), "test": str(base / "test.json")})
    columns = raw["train"].column_names
    prompt_key = "prompt" if "prompt" in columns else "instruction"

    def to_conversational(example):
        return {
            "prompt": [{"role": "user", "content": str(example[prompt_key])}],
            "chosen": as_messages(example["chosen"], "assistant"),
            "rejected": as_messages(example["rejected"], "assistant"),
        }

    drop = [name for name in columns if name not in ("chosen", "rejected")]
    return raw.map(to_conversational, remove_columns=drop)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="trl-lib/ultrafeedback_binarized", help="源数据集（走 HuggingFace Hub）")
    parser.add_argument("--local-dir", default="",
                        help="改用本地文件目录（train.parquet/test.parquet 或 train.json/test.json），绕过 Hub")
    parser.add_argument("--seed", type=int, default=42, help="子集采样种子（两组配置的 seed 也是 42）")
    parser.add_argument("--train-size", type=int, default=30500, help="第一轮训练集大小（默认 = 全量 50%%）")
    parser.add_argument("--eval-size", type=int, default=512, help="固定评测集大小")
    parser.add_argument("--out", default="data/uf-subset", help="第一轮数据目录")
    parser.add_argument("--full-out", default="data/uf-full", help="全量轮数据目录")
    parser.add_argument("--skip-full", action="store_true", help="不生成全量数据集（省磁盘）")
    parser.add_argument("--force", action="store_true", help="覆盖已存在的目录")
    parser.add_argument("--no-verify", action="store_true", help="跳过写盘后的回读校验")
    args = parser.parse_args()

    # 仓库/集群环境：家目录常在系统盘，HF 缓存默认丢到 ~/data（若存在）
    if not os.environ.get("HF_HOME") and (Path.home() / "data").is_dir():
        os.environ["HF_HOME"] = str(Path.home() / "data" / "hf")

    from datasets import DatasetDict, load_dataset, load_from_disk

    # 相对路径一律相对**仓库根目录**解析：配置里写的是 data/uf-subset，
    # 所以无论从哪个目录调用本脚本，数据都必须落在仓库根下，否则训练找不到。
    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out) if Path(args.out).is_absolute() else root / args.out
    full_dir = Path(args.full_out) if Path(args.full_out).is_absolute() else root / args.full_out
    need_small = args.force or not (out_dir / "dataset_dict.json").is_file()
    need_full = (not args.skip_full) and (args.force or not (full_dir / "dataset_dict.json").is_file())

    if not need_small and not need_full:
        print(f"[skip] 数据集已存在：{out_dir}" + ("" if args.skip_full else f"、{full_dir}"))
        print("       需要重建请加 --force")
        return 0

    source = args.local_dir or args.dataset
    print(f"[1/3] 读取 {source} ...")
    try:
        raw = load_source(args, load_dataset)
    except Exception as exc:  # 下载失败/镜像不可达/断网
        print(f"[fail] 读取数据集失败：{exc}", file=sys.stderr)
        print("       国内机器先设置镜像再重试：", file=sys.stderr)
        print("       export HF_ENDPOINT=https://hf-mirror.com", file=sys.stderr)
        print("       或者检查网络/磁盘（数据约 200 MB，缓存在 $HF_HOME）", file=sys.stderr)
        return 1
    if "train" not in raw:
        print(f"[fail] 源数据集没有 train 划分，实际有：{list(raw)}", file=sys.stderr)
        return 1
    print(f"      train={raw['train'].num_rows}  其他划分={ {k: v.num_rows for k, v in raw.items() if k != 'train'} }")

    if "test" not in raw:
        print("[fail] 源数据集没有 test 划分；评测集需要它。", file=sys.stderr)
        return 1

    # EA-DPO 遇到空 prompt / 空 completion 会直接报错（权重归一化没有有效 Token 可算），
    # 与其在第八小时才炸，不如在这里先剔掉。ultrafeedback 里正常情况下一条都没有。
    def has_text(example):
        return all(text_of(example.get(key)) for key in ("prompt", "chosen", "rejected"))

    kept = {}
    for split in ("train", "test"):
        before = raw[split].num_rows
        raw[split] = raw[split].filter(has_text, desc=f"丢弃空样本 [{split}]")
        kept[split] = before - raw[split].num_rows
        if kept[split]:
            print(f"      [warn] {split} 丢弃 {kept[split]} 条空 prompt/completion 样本")
    if not any(kept.values()):
        print("      （没有空样本需要丢弃）")

    eval_size = min(args.eval_size, raw["test"].num_rows)
    if eval_size != args.eval_size:
        print(f"[warn] test 只有 {raw['test'].num_rows} 条，评测集取 {eval_size} 条")

    print(f"[2/3] 固定子集（seed={args.seed}）：train {args.train_size} / test {eval_size}")
    # 先按同一行序打乱再切片：第一轮与全量轮的开头若干条完全相同。
    shuffled_train = raw["train"].shuffle(seed=args.seed)
    shuffled_test = raw["test"].shuffle(seed=args.seed)
    eval_subset = shuffled_test.select(range(eval_size))
    train_size = min(args.train_size, shuffled_train.num_rows)
    if train_size != args.train_size:
        print(f"[warn] train 只有 {shuffled_train.num_rows} 条，子集取 {train_size} 条")
    train_subset = shuffled_train.select(range(train_size))
    print(f"      第一轮：train {train_subset.num_rows} / test {eval_subset.num_rows}"
          f"（占全量 {100 * train_subset.num_rows / raw['train'].num_rows:.1f}%）")

    print("[3/3] 写盘 ...")
    if need_small:
        DatasetDict({"train": train_subset, "test": eval_subset}).save_to_disk(str(out_dir))
        print(f"      {out_dir}  train={train_subset.num_rows}  test={eval_subset.num_rows}")
    else:
        print(f"      [skip] {out_dir} 已存在")

    if not args.skip_full:
        if need_full:
            DatasetDict({"train": raw["train"], "test": eval_subset}).save_to_disk(str(full_dir))
            print(f"      {full_dir}  train={raw['train'].num_rows}  test={eval_subset.num_rows}")
        else:
            print(f"      [skip] {full_dir} 已存在")

    if args.no_verify:
        return 0

    # 回读校验：确认目录能被 load_dataset 直接读，且 test 与预期一致。
    print("回读校验 ...")
    ok = True
    for path, expect_train in [(out_dir, train_size)] + ([] if args.skip_full else [(full_dir, raw["train"].num_rows)]):
        loaded = load_from_disk(str(path))
        train_rows, test_rows = loaded["train"].num_rows, loaded["test"].num_rows
        same_eval = loaded["test"]["prompt"] == eval_subset["prompt"]
        status = "ok" if (train_rows == expect_train and test_rows == eval_size and same_eval) else "MISMATCH"
        ok = ok and status == "ok"
        print(f"  [{status}] {path}  train={train_rows}  test={test_rows}  评测集一致={same_eval}")

    print("\n完成。下一步：python tools/check_configs.py")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
