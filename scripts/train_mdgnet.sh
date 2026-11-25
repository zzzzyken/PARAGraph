#!/bin/bash
# MDGNet baseline (re-implemented, see docs/baselines_pranet_mdgnet.md): repeated splits,
# same protocol as the other baselines. 1024 px input, ViG graph fixed at 14x14.
#
# Usage:  bash scripts/train_mdgnet.sh <data_root> [name]
#
#   data_root   grading dataset folder in the fixed format (README, Data)
#   name        prefix of the outputs (default: mdgnet_<folder name of data_root>)
#
# Environment: SEEDS="42 0 1 2 1024", EPOCHS=30 (the paper used 30 on DDR, 50 on Messidor-2 and APTOS)
# Writes baseline/<name>_run<N>/ and logs/<name>_run<N>.log.

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

DATA_ROOT="${1:?usage: bash scripts/train_mdgnet.sh <data_root> [name]}"
NAME="${2:-mdgnet_$(basename "$DATA_ROOT")}"
read -r -a SEEDS <<< "${SEEDS:-42 0 1 2 1024}"
EP="${EPOCHS:-30}"
N=${#SEEDS[@]}

for i in "${!SEEDS[@]}"; do
  SPLIT_SEED="${SEEDS[$i]}"; RUN=$((i + 1))
  SAVE_DIR="./baseline/${NAME}_run${RUN}"
  LOG="logs/${NAME}_run${RUN}.log"
  echo "==== ${NAME} run ${RUN}/${N} (split_seed=${SPLIT_SEED}, ${EP}ep) ===="
  python baselines/train_mdgnet.py \
    --data_root "${DATA_ROOT}" \
    --image_size 1024 \
    --graph_hw 14 \
    --knn_k 9 \
    --epochs "${EP}" \
    --batch_size 8 \
    --lr 3e-4 \
    --weight_decay 0.05 \
    --use_oversample \
    --num_workers 4 \
    --seed 1234 \
    --split_seed "${SPLIT_SEED}" \
    --save_dir "${SAVE_DIR}" \
    > "${LOG}" 2>&1
  echo "  done: $(grep 'Best val Kappa' "${LOG}" | tail -1)"
done

echo ""
echo "${NAME} summary:"
for i in $(seq 1 ${N}); do
  LOG="logs/${NAME}_run${i}.log"
  [ -f "$LOG" ] && echo "  run${i}: $(grep 'TEST  Kappa' "$LOG" | tail -1)"
done
