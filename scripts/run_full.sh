#!/usr/bin/env bash
# =============================================================================
# 第二轮：全量 1 epoch（7642 步），EA-DPO vs 标准 DPO
#
#   bash scripts/run_full.sh              # 依次跑对照组和 EA-DPO
#   bash scripts/run_full.sh dpo          # 只跑对照组
#   bash scripts/run_full.sh ea           # 只跑 EA-DPO
#   bash scripts/run_full.sh report       # 只汇总
#   STEPS=3000 bash scripts/run_full.sh   # 只跑约 40% 数据
#
# 4090 24G 上 7642 步约 18-22 小时；A100 80G 约 8-10 小时；H100 约 5-6 小时。
# 中途断了直接重跑本脚本，会用 --resume_from_checkpoint auto 从最新存档继续。
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
STAGE="${1:-all}"
STEPS="${STEPS:-}"
EA_CFG="configs/ea_dpo_full.yaml"
DPO_CFG="configs/dpo_baseline_full.yaml"
EA_OUT="outputs/ea-3b-full"
DPO_OUT="outputs/dpo-3b-full"
# 国内下载默认走镜像。想直连官方：export HF_ENDPOINT=https://huggingface.co
if [ -z "${HF_ENDPOINT-}" ]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
# 集群/学校服务器：大文件放数据盘，HF 缓存同理
if [ -z "${HF_HOME-}" ] && [ -d "$HOME/data" ]; then
  export HF_HOME="$HOME/data/hf"
fi

mkdir -p logs

banner() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }
die() { printf '\n[fail] %s\n' "$*" >&2; exit 1; }

preflight() {
  banner "环境自检"
  "$PY" - <<'PY' || die "环境不完整，先跑 bash scripts/setup_env.sh"
import sys
from pathlib import Path

import torch, transformers, trl

trl_path = Path(trl.__file__).resolve()
if Path.cwd().resolve() not in trl_path.parents:
    print(f"[fail] trl 来自 {trl_path}，不是本仓库的修改版")
    sys.exit(1)
if not torch.cuda.is_available():
    print("[fail] CUDA 不可用")
    sys.exit(1)
print(f"trl OK  transformers={transformers.__version__}  torch={torch.__version__}  "
      f"{torch.cuda.get_device_name(0)}")
PY
  [ -f data/uf-full/dataset_dict.json ] \
    || die "缺少 data/uf-full，先执行：python scripts/prepare_data.py"
  "$PY" tools/check_configs.py || die "配置检查未通过"
  local avail
  avail=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}' || echo 0)
  avail=${avail:-0}
  echo "磁盘可用 ${avail} GiB（全量跑建议 >30 GiB：模型 6GB + 2 个 checkpoint + 缓存）"
}

record_run_env() {
  # $1=输出目录 $2=配置文件
  mkdir -p "$1"
  cp "$2" "$1/run_config.yaml"
  {
    echo "date     : $(date -Is)"
    echo "host     : $(hostname)"
    echo "user     : $(whoami)"
    echo "git_rev  : $(git rev-parse --short HEAD 2>/dev/null || echo 'no-git')"
    echo "python   : $("$PY" -V 2>&1)"
    echo "torch    : $("$PY" -c 'import torch;print(torch.__version__, torch.version.cuda)' 2>/dev/null || echo 'n/a')"
    echo "gpu      : $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -n1)"
    echo "config   : $2"
    echo "steps    : ${STEPS:-from-config}"
    echo "script   : $0 $STAGE"
  } > "$1/run_env.txt"
}

train_one() {
  local name="$1" cfg="$2" out="$3" log="logs/$1-full.log"
  [ -f "$cfg" ] || die "缺少配置文件 $cfg"
  banner "训练：$name（全量）"
  if [ -f "$out/adapter_config.json" ] || [ -f "$out/config.json" ]; then
    echo "[skip] $out 已有训练完成的产物。要重跑：rm -rf $out"
    return 0
  fi
  record_run_env "$out" "$cfg"
  if [ -n "$STEPS" ]; then
    "$PY" train_dpo.py --config "$cfg" --max_steps "$STEPS" --resume_from_checkpoint auto 2>&1 | tee "$log"
  else
    "$PY" train_dpo.py --config "$cfg" --resume_from_checkpoint auto 2>&1 | tee "$log"
  fi
  echo "[ok] $name 完成，产物在 $out"
}

report() {
  banner "汇总"
  for out in "$DPO_OUT" "$EA_OUT"; do
    if [ -f "$out/trainer_state.json" ] || ls "$out"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
      echo "--- $out ---"
      "$PY" tools/report.py "$out" --tail 5 || true
    fi
  done
  "$PY" tools/compare_runs.py "$DPO_OUT" "$EA_OUT" --labels "DPO-full" "EA-DPO-full" || true
}

case "$STAGE" in
  all)    preflight; train_one ea "$EA_CFG" "$EA_OUT"; train_one dpo "$DPO_CFG" "$DPO_OUT"; report ;;
  dpo)    preflight; train_one dpo "$DPO_CFG" "$DPO_OUT" ;;
  ea)     preflight; train_one ea "$EA_CFG" "$EA_OUT" ;;
  report) report ;;
  *)      die "未知阶段 $STAGE（可用：all / dpo / ea / report）" ;;
esac

banner "完成"
