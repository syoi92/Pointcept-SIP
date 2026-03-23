#!/bin/sh
set -eu

GPU=1
DATASET="sip"

CONFIGS="
semseg-ptv2-r12-manifold
semseg-ptv2-r12-manifold-s
semseg-ptv2-r03-manifold
semseg-ptv2-r03-manifold-s
semseg-ptv2-r06-manifold
semseg-ptv2-r06-manifold-s
semseg-ptv2-r12-base
"

for cfg in $CONFIGS; do
  for trial in 1 2; do
    name="${cfg}-mdl${trial}"

    echo "=============================="
    echo "[RUN] $name"
    echo "=============================="

    sh scripts/train.sh -g "$GPU" -d "$DATASET" -c "$cfg" -n "$name"
    status=$?

    if [ "$status" -ne 0 ]; then
      echo "[FAIL] $name (exit code=$status) -> continue"
    else
      echo "[DONE] $name"
    fi
  done
done

echo "ALL DONE"