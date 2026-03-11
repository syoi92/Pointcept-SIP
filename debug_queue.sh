#!/bin/sh
set -eu

GPU=1
DATASET="sip"

CONFIGS="
2_debug_r06_recep1
2_debug_r06_recep2
2_debug_r06_recep3
2_debug_r10_recep1
2_debug_r10_recep2
"


for cfg in $CONFIGS; do
  echo "=============================="
  echo "[RUN] $cfg"
  echo "=============================="
  sh scripts/train.sh -g "$GPU" -d "$DATASET" -c "$cfg" -n "$cfg"
  status=$?

  if [ "$status" -ne 0 ]; then
    echo "[FAIL] $cfg (exit code=$status) -> continue"
  else
    echo "[DONE] $cfg"
  fi
done

echo "ALL DONE"
