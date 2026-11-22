#!/usr/bin/env python3
"""从 HuggingFace 拉取数据集文件到本地（只用标准库，不需要装 datasets/huggingface_hub）。

    python tools/fetch_datasets.py --dest "D:/.../数据集" trl-lib/ultrafeedback_binarized
    python tools/fetch_datasets.py --dest "..." --list trl-lib/ultrafeedback_binarized Anthropic/hh-rlhf

特性：
  * 走 Hub 的 tree API 列出全部文件，再逐个下载（支持 hf-mirror 镜像回退）
  * 断点续传：大小一致就跳过
  * 文件按 path 还原目录结构，例如 data/train-00000-of-00001.parquet
"""

import argparse
import json
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

MIRRORS = ["https://hf-mirror.com", "https://huggingface.co"]


def human(n):
    return f"{n / 2**20:.1f} MB" if n >= 2**20 else f"{n / 2**10:.0f} KB"


def api_tree(base, dataset, revision="main", timeout=60):
    quoted = urllib.parse.quote(revision, safe="")
    url = f"{base}/api/datasets/{dataset}/tree/{quoted}?recursive=true"
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        data = json.load(resp)
    return [e for e in data if e.get("type") == "file"]


def download_file(url, dest, expect_size, retries=4):
    """带续传的下载；服务端返回 302 到 CDN 时 urllib 会自动跟随。"""
    for attempt in range(1, retries + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if expect_size and have == expect_size:
            return "skip"
        headers = {"User-Agent": "fetch-datasets/1.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=180) as resp, open(dest, "ab" if have else "wb") as handle:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    handle.write(chunk)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            print(f"      重试 {attempt}/{retries}：{exc}")
            time.sleep(2 * attempt)
            continue
        if not expect_size or dest.stat().st_size == expect_size:
            return "ok"
        print(f"      大小不符（{dest.stat().st_size}/{expect_size}），续传")
    return "fail"


def fetch(dataset, dest_root, dry_run=False, revision="main"):
    local = Path(dest_root) / dataset.replace("/", "__")
    print(f"\n=== {dataset} → {local}")
    for base in MIRRORS:
        try:
            files = api_tree(base, dataset, revision)
            print(f"    源 {base}：{len(files)} 个文件，共 {human(sum(f.get('size', 0) for f in files))}")
            break
        except Exception as exc:
            print(f"    {base} 列目录失败：{type(exc).__name__}")
    else:
        print("    [fail] 两个源都失败")
        return False

    ok = fail = 0
    for entry in files:
        path, size = entry["path"], entry.get("size", 0)
        target = local / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if dry_run:
            print(f"    [dry] {path}  {human(size)}")
            continue
        # 修订名里含斜杠（如 refs/convert/parquet）时必须整体转义，否则会被当成路径分隔
        quoted_rev = urllib.parse.quote(revision, safe="")
        status = download_file(f"{base}/datasets/{dataset}/resolve/{quoted_rev}/{path}", target, size)
        flag = "ok  " if status in ("ok", "skip") else "FAIL"
        print(f"    [{flag}] {path}  {human(target.stat().st_size) if target.exists() else '-'}")
        ok += status in ("ok", "skip")
        fail += status == "fail"
    print(f"    → 成功 {ok}，失败 {fail}")
    return fail == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("datasets", nargs="+", help="数据集名，如 trl-lib/ultrafeedback_binarized")
    parser.add_argument("--dest", default="datasets", help="本地根目录")
    parser.add_argument("--dry-run", action="store_true", help="只列文件不下载")
    parser.add_argument("--revision", default="main", help="分支/修订；脚本型数据集用 refs/convert/parquet")
    args = parser.parse_args()

    root = Path(args.dest)
    root.mkdir(parents=True, exist_ok=True)
    results = {name: fetch(name, root, args.dry_run, args.revision) for name in args.datasets}
    print("\n=== 汇总 ===")
    for name, good in results.items():
        print(f"  {'OK  ' if good else 'FAIL'} {name}")
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
