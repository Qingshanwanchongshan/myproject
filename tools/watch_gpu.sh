#!/usr/bin/env bash
# Sample GPU usage while a run is going, so you can see whether the card is actually busy
# and how close memory gets to the limit (OOM risk).
#
# Usage:  bash tools/watch_gpu.sh [interval_seconds] [csv_path]
# Stop:   Ctrl-C
set -u

interval="${1:-5}"
out="${2:-gpu_watch.csv}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found; run this on the GPU host" >&2
  exit 1
fi

if [ ! -f "$out" ]; then
  echo "epoch_s,gpu,util_pct,mem_used_mib,mem_total_mib,temp_c,power_w" > "$out"
fi

echo "sampling every ${interval}s into ${out} (Ctrl-C to stop)"
while true; do
  ts="$(date +%s)"
  nvidia-smi \
    --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv,noheader,nounits 2>/dev/null \
    | sed -e 's/, */,/g' -e "s/^/${ts},/" >> "$out"
  sleep "$interval"
done
