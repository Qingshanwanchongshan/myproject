#!/usr/bin/env bash
# =============================================================================
# 第一轮：20% 数据，EA-DPO vs 标准 DPO 趋势对比（单卡 4090 24G，无人值守）
#
#   bash scripts/run_first_round.sh              # 全流程：冒烟 -> 测速 -> 两组训练 -> 对比
#   bash scripts/run_first_round.sh smoke        # 只做冒烟
#   bash scripts/run_first_round.sh timing       # 只做测速
#   bash scripts/run_first_round.sh dpo          # 只跑对照组
#   bash scripts/run_first_round.sh ea           # 只跑 EA-DPO
#   bash scripts/run_first_round.sh report       # 只汇总已有结果
#
# 环境变量
#   STEPS=            训练步数；留空 = 用配置里的 max_steps（-1 → 跑满子集一个 epoch，约 3813 步 = 30500 对）
#   PYTHON=python   解释器
#   LR / TAG        学习率对照组：LR=5.0e-6 TAG=lr5e-6 bash scripts/run_first_round.sh
#                   （设了 LR 就必须设 TAG，否则会覆盖主实验的 output_dir）
#
# 特性
#   * 幂等：某组训练完成后重跑会自动跳过；训练中断后用 --resume_from_checkpoint auto 续跑
#   * 日志落在 logs/，训练产物落在 outputs/
#   * 任何一步失败立刻停下（set -e），不会带着坏状态继续烧机时
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PYTHON:-python}"
STEPS="${STEPS:-}"                                 # 空 = 交给配置（max_steps: -1 + num_train_epochs: 1）
EFF_STEPS="${STEPS:--1}"                          # 实际传给 --max_steps 的值
STAGE="${1:-all}"
LR="${LR:-}"
TAG="${TAG:-}"
if [ -n "$LR" ] && [ -z "$TAG" ]; then
  echo "[fail] 设了 LR 就必须同时设 TAG，否则产物会覆盖主实验目录。" >&2
  echo "       例如：LR=5.0e-6 TAG=lr5e-6 bash scripts/run_first_round.sh" >&2
  exit 1
fi
EA_CFG="configs/ea_dpo.yaml"
DPO_CFG="configs/dpo_baseline.yaml"
PLAN_STEPS="${PLAN_STEPS:-3813}"                    # 仅用于测速换算：30500 对 / 8
EA_OUT="outputs/ea-3b-run1${TAG:+-$TAG}"
DPO_OUT="outputs/dpo-3b-run1${TAG:+-$TAG}"
# 国内下载默认走镜像。想直连官方：export HF_ENDPOINT=https://huggingface.co
if [ -z "${HF_ENDPOINT-}" ]; then
  export HF_ENDPOINT="https://hf-mirror.com"
fi
# 集群/学校服务器：大文件放数据盘，HF 缓存同理（否则 6GB 模型会写进系统盘）
if [ -z "${HF_HOME-}" ] && [ -d "$HOME/data" ]; then
  export HF_HOME="$HOME/data/hf"
fi

mkdir -p logs

banner() { printf '\n\033[1m===== %s =====\033[0m\n' "$*"; }
die() { printf '\n[fail] %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
preflight() {
  banner "0. 环境自检"

  [ -f train_dpo.py ] || die "请在仓库根目录执行本脚本"
  [ -f "$EA_CFG" ] || die "缺少 $EA_CFG"
  [ -f "$DPO_CFG" ] || die "缺少 $DPO_CFG"

  "$PY" - <<'PY' || die "python 依赖不完整，先跑 bash scripts/setup_env.sh"
import sys
import torch, transformers, trl
from pathlib import Path

trl_path = Path(trl.__file__).resolve()
if Path.cwd().resolve() not in trl_path.parents:
    print(f"[fail] trl 来自 {trl_path}，不是本仓库的修改版")
    sys.exit(1)
if not torch.cuda.is_available():
    print("[fail] CUDA 不可用")
    sys.exit(1)
free, total = torch.cuda.mem_get_info()
print(f"trl={trl_path.parent.parent.name} transformers={transformers.__version__} "
      f"torch={torch.__version__} {torch.cuda.get_device_name(0)} "
      f"显存 {free / 2**30:.1f}/{total / 2**30:.1f} GiB 空闲")
if free / 2**30 < 18:
    print(f"[warn] 当前空闲显存只有 {free / 2**30:.1f} GiB，3B bf16 + 梯度检查点一般要 11-14 GiB，"
          "如果开机后一直是这个数值，先确认没有别的进程占卡")
PY

  [ -f data/uf-subset/dataset_dict.json ] \
    || die "缺少 data/uf-subset，先执行：python scripts/prepare_data.py"

  "$PY" tools/check_configs.py || die "配置检查未通过"

  local avail
  avail=$(df -Pk . | awk 'NR==2 {print int($4/1024/1024)}' || echo 0)
  avail=${avail:-0}
  echo "磁盘可用 ${avail} GiB（建议 >20 GiB）"
  [ "$avail" -ge 20 ] || echo "[warn] 磁盘偏小，3 个 checkpoint + 模型缓存可能不够"
  if [ -n "$LR" ]; then
    echo "[info] 学习率覆盖为 $LR（配置里是 1.0e-5），产物写到 outputs/*-$TAG"
  fi
}

# -----------------------------------------------------------------------------
smoke() {
  banner "1. 冒烟：两组各 10 步（约 5-8 分钟，验证链路、目标层、权重归一）"

  # 冒烟也写 checkpoint：只有 trainer_state.json 才能精确校验 ea/weight_mean_*，
  # 控制台日志不一定打印全部指标。save_only_model 省掉优化器状态，每个存档只有 adapter。
  rm -rf outputs/smoke-ea outputs/smoke-dpo

  echo "--- EA-DPO ---"
  "$PY" train_dpo.py --config "$EA_CFG" --max_steps 10 --eval_strategy no \
    --save_strategy steps --save_steps 10 --save_total_limit 1 --save_only_model true \
    --report_to none --output_dir outputs/smoke-ea 2>&1 | tee logs/smoke-ea.log

  "$PY" tools/report.py outputs/smoke-ea --tail 3 2>&1 | tee logs/smoke-ea-report.log \
    || die "report.py 读不到 outputs/smoke-ea 的训练状态，见 logs/smoke-ea-report.log"
  if grep -q "FAIL" logs/smoke-ea-report.log; then
    die "EA-DPO 冒烟健康检查未通过，见 logs/smoke-ea-report.log"
  fi
  grep -q "weight_mean stays at 1" logs/smoke-ea-report.log \
    || die "报告里没有权重均值检查结果，说明 ea/ 指标没进 trainer_state.json"
  echo "[ok] 权重均值恒为 1（公式 (10) 的保均值约束）"

  # 目标层：Qwen2.5-3B 有 36 层，自动选择 ceil(0.8*36)-1 = 28（第 29 层）
  # 从 trainer_state.json 读（日志文本不一定打这行，指标才是权威）
  "$PY" - outputs/smoke-ea <<'PY' || die "读不到 ea/target_layer，说明 EA 没真正启用"
import glob
import json
import sys

paths = sorted(glob.glob(sys.argv[1] + "/checkpoint-*/trainer_state.json"))
if not paths:
    print("[fail] 找不到 trainer_state.json")
    sys.exit(1)
rows = json.load(open(paths[-1], encoding="utf-8"))["log_history"]
values = [row["ea/target_layer"] for row in rows if "ea/target_layer" in row]
if not values:
    print("[fail] 日志里没有 ea/target_layer，EA-DPO 可能没启用")
    sys.exit(1)
layer = int(values[-1])
print(f"[ok] EA-DPO 目标层 = {layer}（零起始；3B 期望 28 = 第 29 层，约 80% 深度）")
if layer != 28:
    print(f"[warn] 3B 预期是 28，实际是 {layer}；换了模型才正常，否则检查 model_name_or_path")
PY

  echo "--- 标准 DPO ---"
  "$PY" train_dpo.py --config "$DPO_CFG" --max_steps 10 --eval_strategy no \
    --save_strategy steps --save_steps 10 --save_total_limit 1 --save_only_model true \
    --report_to none --output_dir outputs/smoke-dpo 2>&1 | tee logs/smoke-dpo.log

  # 对照组必须一个 ea/ 指标都没有，否则说明开关没生效
  "$PY" - outputs/smoke-dpo <<'PY' || die "对照组行为不符合预期"
import json
import sys

sys.path.insert(0, "tools")
from report import find_state

state_path = find_state(sys.argv[1])
if state_path is None:
    print("[fail] 找不到 trainer_state.json")
    sys.exit(1)
rows = json.load(open(state_path, encoding="utf-8"))["log_history"]
keys = sorted({key for row in rows for key in row})
ea_keys = [key for key in keys if key.startswith("ea/")]
if ea_keys:
    print(f"[fail] 对照组出现了 ea/ 指标：{ea_keys}")
    sys.exit(1)
print(f"[ok] 对照组没有 ea/ 指标，走的是原生 DPO 路径（{len(rows)} 个日志点）")
PY

  echo "[ok] 冒烟通过。清理：rm -rf outputs/smoke-ea outputs/smoke-dpo"
}

# -----------------------------------------------------------------------------
timing() {
  banner "2. 测速：EA-DPO 跑 40 步，换算两组和全量的预计耗时"
  rm -rf outputs/timing-ea

  "$PY" train_dpo.py --config "$EA_CFG" --max_steps 40 --eval_strategy no --save_strategy no \
    --logging_steps 10 --report_to none --output_dir outputs/timing-ea 2>&1 | tee logs/timing-ea.log

  "$PY" tools/estimate_time.py logs/timing-ea.log --steps "${STEPS:-$PLAN_STEPS}" --measured-steps 40 \
    --eval-every 500 || echo "[warn] 没能从日志里解析出速度，请手工看 logs/timing-ea.log 末尾的 Train metrics"
  rm -rf outputs/timing-ea
}

# -----------------------------------------------------------------------------
# 把"这次到底用什么跑的"写进产物目录：配置副本 + 机器/版本/命令
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
    echo "steps    : ${STEPS:-(from config)}"
    echo "lr       : ${LR:-from-config}"
    echo "tag      : ${TAG:-none}"
    echo "script   : $0 $STAGE"
  } > "$1/run_env.txt"
}

train_one() {
  # $1=名字 $2=配置 $3=输出目录
  local name="$1" cfg="$2" out="$3" log="logs/$1${TAG:+-$TAG}.log"
  [ -f "$cfg" ] || die "缺少配置文件 $cfg"
  banner "训练：$name${TAG:+（$TAG）}"
  if [ -f "$out/adapter_config.json" ] || [ -f "$out/config.json" ]; then
    echo "[skip] $out 已有训练完成的产物（adapter_config.json）。"
    echo "       要重跑：rm -rf $out；要续跑中断的任务：直接重跑本脚本（会自动 --resume_from_checkpoint auto）"
    return 0
  fi
  record_run_env "$out" "$cfg"
  if [ -n "$LR" ]; then
    "$PY" train_dpo.py --config "$cfg" --max_steps "$EFF_STEPS" --learning_rate "$LR" \
      --resume_from_checkpoint auto 2>&1 | tee "$log"
  else
    "$PY" train_dpo.py --config "$cfg" --max_steps "$EFF_STEPS" \
      --resume_from_checkpoint auto 2>&1 | tee "$log"
  fi
  echo "[ok] $name 完成，产物在 $out"
}

# -----------------------------------------------------------------------------
report() {
  banner "汇总"
  local any=0
  for out in "$DPO_OUT" "$EA_OUT"; do
    if [ -f "$out/trainer_state.json" ] || ls "$out"/checkpoint-*/trainer_state.json >/dev/null 2>&1; then
      echo "--- $out ---"
      "$PY" tools/report.py "$out" --tail 5 || true
      any=1
    fi
  done
  if [ "$any" = "1" ]; then
    echo
    "$PY" tools/compare_runs.py "$DPO_OUT" "$EA_OUT" --labels "DPO" "EA-DPO" || true
  else
    echo "还没有可汇总的结果"
  fi
  # 两组都训练完成才有意义：用同一把尺子（不加权的标准 DPO 隐式奖励）重算一遍
  if [ -f "$DPO_OUT/adapter_config.json" ] && [ -f "$EA_OUT/adapter_config.json" ]; then
    banner "同口径离线评测（标准 DPO 隐式奖励，可横向比）"
    "$PY" tools/eval_implicit_reward.py \
      --adapter "$DPO_OUT:DPO" --adapter "$EA_OUT:EA-DPO" --limit 0 \
      --out "outputs/implicit-reward${TAG:+-$TAG}.json" \
      || echo "[warn] 离线评测失败，可单独重跑：python tools/eval_implicit_reward.py --adapter $DPO_OUT:DPO --adapter $EA_OUT:EA-DPO"
  else
    echo
    echo "（两组都训练完成后，report 阶段会自动跑同口径离线评测）"
  fi
  echo
  echo "曲线：tensorboard --logdir outputs --port 6006"
  echo "产物：$DPO_OUT  和  $EA_OUT（adapter 在目录根部，checkpoint-* 是中间存档）"
}

# -----------------------------------------------------------------------------
case "$STAGE" in
  all)    preflight; smoke; timing; train_one ea "$EA_CFG" "$EA_OUT"; train_one dpo "$DPO_CFG" "$DPO_OUT"; report ;;
  smoke)  preflight; smoke ;;
  timing) preflight; timing ;;
  dpo)    preflight; train_one dpo "$DPO_CFG" "$DPO_OUT" ;;
  ea)     preflight; train_one ea "$EA_CFG" "$EA_OUT" ;;
  report) report ;;
  *)      die "未知阶段 $STAGE（可用：all / smoke / timing / dpo / ea / report）" ;;
esac

banner "完成"
echo "下一步：python tools/compare_runs.py $DPO_OUT $EA_OUT --labels DPO EA-DPO"
