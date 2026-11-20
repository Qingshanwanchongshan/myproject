#!/usr/bin/env bash
# =============================================================================
# 一次性环境准备（国内 RTX 4090 服务器，Ubuntu 22.04 / 24.04，已有 NVIDIA 驱动）
#
#   bash scripts/setup_env.sh
#
# 幂等：可以重复执行。会做五件事
#   1. 检查/安装支持 CUDA 的 torch（>= 2.6，TRL v1.18.1 的要求）
#   2. 以可编辑方式安装本仓库的修改版 TRL、固定 transformers 5.23.0
#   3. 校验 `import trl` 指向本仓库，而不是 PyPI 上的官方包
#   4. 跑 EA-DPO 的单元测试（33 个，CPU 上几分钟，不下载权重）
#   5. 报告 GPU / bf16 / 磁盘情况
#
# 常用环境变量
#   PYTHON=python3.11        指定解释器（默认 python；必须 >= 3.11）
#   HF_ENDPOINT=...          默认 https://hf-mirror.com；
#                            想直连官方就设成 https://huggingface.co（不要置空）
#   TORCH_INDEX_URL=...      默认 cu124；40 系卡也可用 cu121/cu126
#   TORCH_VERSION=2.8.0      仅在需要新装 torch 时使用（镜像已装 >=2.6 就不动它）
#   SKIP_TORCH=1             镜像里已装好 torch，跳过 torch 检查与安装
#   SKIP_TESTS=1             没网/没时间时跳过第 4 步（不推荐）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
# 国内下载默认走镜像。想直连官方：export HF_ENDPOINT=https://huggingface.co
# 注意不能用"置空"来取消——空串会让下载地址变成空，直接失败。
if [ -z "${HF_ENDPOINT-}" ]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
# 集群/学校服务器：家目录常在系统盘，大文件要放数据盘。
# 本项目约定：仓库放在数据盘（如 ~/data/EA-DPO），HF 缓存也放过去，避免撑爆系统盘。
if [ -z "${HF_HOME-}" ] && [ -d "$HOME/data" ]; then
  export HF_HOME="$HOME/data/hf"
fi
export TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
export TORCH_VERSION="${TORCH_VERSION:-2.8.0}"

echo "工作目录      : $(pwd)"
echo "解释器        : $("$PY" -V 2>&1)  ($(command -v "$PY" 2>/dev/null || echo "$PY"))"
echo "HF_ENDPOINT   : $HF_ENDPOINT"
echo "HF_HOME       : ${HF_HOME:-(默认 ~/.cache/huggingface)}"
echo

# 无 root 的服务器上写不了系统 site-packages，必须在 venv 里跑（脚本内部逻辑不变）
"$PY" - <<'PY' || true
import site, sys, sysconfig
path = sysconfig.get_paths().get("purelib", "")
writable = False
try:
    import os
    writable = os.access(path, os.W_OK)
except Exception:
    pass
where = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
print(f"环境类型      : {'venv/conda' if where else '系统解释器'}；site-packages 可写={writable}")
if not where and not writable:
    print("[warn] 系统解释器且 site-packages 不可写：请先建虚拟环境再跑本脚本")
    print("       python -m venv ~/data/venvs/eadpo && source ~/data/venvs/eadpo/bin/activate")
    print("       PYTHON=$VIRTUAL_ENV/bin/python bash scripts/setup_env.sh")
PY
echo

# -----------------------------------------------------------------------------
echo "== [1/5] python 与 torch =="
"$PY" - <<'PY' || { echo "[fail] Python 版本不满足要求" >&2; exit 1; }
import sys

print(f"python {sys.version.split()[0]}  ({sys.executable})")
if sys.version_info < (3, 11):
    print("[fail] TRL v1.18.1 需要 Python >= 3.11。")
    print("       租来的镜像常用 3.10，请换镜像或改用 conda：")
    print("       conda create -n eadpo python=3.11 -y && conda activate eadpo")
    print("      然后：PYTHON=$(which python) bash scripts/setup_env.sh")
    sys.exit(1)
PY

check_torch() {
  "$PY" - <<'PY'
import sys
try:
    import torch
except Exception as exc:  # 未安装
    print(f"torch 未安装: {exc}")
    sys.exit(1)
if not torch.cuda.is_available():
    print(f"torch {torch.__version__} 已安装，但 CUDA 不可用（驱动或镜像的问题）")
    sys.exit(2)
major, minor = (int(part) for part in torch.__version__.split(".")[:2])
free, total = torch.cuda.mem_get_info()
print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
      f"{torch.cuda.get_device_name(0)}  显存 {total / 2**30:.1f} GiB")
if (major, minor) < (2, 6):
    print("TRL v1.18.1 要求 torch >= 2.6")
    sys.exit(3)
if not torch.cuda.is_bf16_supported():
    print("这张卡不支持 bf16 训练；配置里的 bf16 需要改成 fp16 并同步改 dtype")
    sys.exit(4)
sys.exit(0)
PY
}

torch_status=0
check_torch || torch_status=$?
if [ "$torch_status" -eq 0 ]; then
  echo "[ok] torch 可用"
elif [ "${SKIP_TORCH:-0}" = "1" ]; then
  echo "[fail] SKIP_TORCH=1 但 torch 检查未通过（退出码 $torch_status）" >&2
  exit 1
else
  echo "[..] 安装 torch $TORCH_VERSION（$TORCH_INDEX_URL）"
  echo "     注意：这会升级/替换镜像自带的 torch。如果镜像里是别的 CUDA 版本组合，"
  echo "     更稳的做法是换一个已经装好 torch>=2.6 的镜像，然后 SKIP_TORCH=1 再跑本脚本。"
  "$PY" -m pip install --upgrade pip
  "$PY" -m pip install --upgrade "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
  check_torch || { echo "[fail] torch 安装后自检仍未通过" >&2; exit 1; }
fi
echo

# -----------------------------------------------------------------------------
echo "== [2/5] 项目依赖 =="
# 不要用 `pip install -U trl`：那会装上官方未修改的包，覆盖这里的 EA-DPO 实现。
"$PY" -m pip install -e './trl-1.18.1[peft]' 'transformers==5.23.0' 'jinja2>=3.1.6' pytest tensorboard
echo

# -----------------------------------------------------------------------------
echo "== [3/5] 校验 trl 来源 =="
"$PY" - <<'PY'
import sys
from pathlib import Path

import transformers
import trl

trl_path = Path(trl.__file__).resolve()
root = Path.cwd().resolve()
print("trl        :", trl_path)
print("transformers:", transformers.__version__)
if root not in trl_path.parents:
    print(f"[fail] import trl 指向 {trl_path}，不是本仓库的修改版。"
          "请在本目录执行 pip install -e './trl-1.18.1[peft]'", file=sys.stderr)
    sys.exit(1)
if not transformers.__version__.startswith("5.23"):
    print(f"[warn] transformers 是 {transformers.__version__}，脚本按 5.23.0 验证过")
print("[ok] trl 来自本仓库")
PY
echo

# -----------------------------------------------------------------------------
echo "== [4/5] 自检与单元测试 =="
if [ "${SKIP_TESTS:-0}" = "1" ]; then
  echo "[skip] SKIP_TESTS=1"
else
  echo "-- 仓库自检（不需要 GPU/transformers，约 1 秒）--"
  "$PY" -m pytest tests/ -q
  echo "-- EA-DPO 单元测试（公式与训练链路，33 个）--"
  if [ ! -f trl-1.18.1/tests/test_ea_dpo.py ]; then
    echo "[warn] 没上传 trl-1.18.1/tests/test_ea_dpo.py，跳过"
  else
    "$PY" -m pytest trl-1.18.1/tests/test_ea_dpo.py -q
  fi
  echo "[ok] 测试通过"
fi
echo

# -----------------------------------------------------------------------------
echo "== [5/5] 机器情况 =="
df -h . | tail -n 1 | awk '{print "磁盘可用      : " $4 "（输出目录 + 6GB 模型 + 数据需要 30GB 以上）"}'
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | awk -F', ' '{print "GPU           : " $1 " / " $2 " / driver " $3}'
echo
echo "环境就绪。下一步："
echo "  python scripts/prepare_data.py      # 生成 20% 数据 + 固定评测集"
echo "  bash scripts/run_first_round.sh     # 冒烟 -> 测速 -> 两组训练 -> 对比"
