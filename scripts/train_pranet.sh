#!/bin/bash
# PRANet baseline (re-implemented, see docs/baselines_pranet_mdgnet.md), two stages.
#   Stage 1 (once): PRANet's own lesion detector, trained on a lesion dataset.
#   Stage 2: grading network (EffV2-M + DAM + co-occurrence GCN over K-means lesion nodes),
#            repeated splits, same protocol as the other baselines. 1024 px input.
#
# Usage:  LESION_ROOT=<lesion dataset folder> bash scripts/train_pranet.sh <data_root> [name]
#         DETECTOR=path/to/best_detector.pth  bash scripts/train_pranet.sh <data_root> [name]
#
#   data_root     grading dataset folder in the fixed format (README, Data)
#   LESION_ROOT   lesion dataset folder in the fixed format; stage 1 trains on it
#   DETECTOR      an existing stage-1 checkpoint; stage 1 is skipped
#
# Environment: SEEDS="42 0 1 2 1024", EPOCHS=30 (the paper used 30 on DDR, 50 on Messidor-2 and APTOS)
# Writes baseline/pranet/ (detector), baseline/<name>_run<N>/ and logs/<name>_run<N>.log.

set -e
cd "$(dirname "$0")/.."
mkdir -p logs baseline/pranet

DATA_ROOT="${1:?usage: LESION_ROOT=<lesion folder> bash scripts/train_pranet.sh <data_root> [name]}"
NAME="${2:-pranet_$(basename "$DATA_ROOT")}"
read -r -a SEEDS <<< "${SEEDS:-42 0 1 2 1024}"
EP="${EPOCHS:-30}"
N=${#SEEDS[@]}

# ---- Stage 1: lesion detector ----
DETECTOR="${DETECTOR:-}"
if [ -z "$DETECTOR" ]; then
  : "${LESION_ROOT:?set LESION_ROOT (lesion dataset folder) or DETECTOR (existing detector checkpoint)}"
  echo "==== PRANet stage 1: training the lesion detector on ${LESION_ROOT} ===="
  python baselines/train_pranet_detector.py \
    --data_root "${LESION_ROOT}" \
    --image_size 1024 --epochs 60 --batch_size 2 --lr 3e-4 \
    --save_dir ./baseline/pranet \
    > logs/pranet_detector.log 2>&1
  DETECTOR=$(ls -t ./baseline/pranet/pranet_detector_*/best_detector.pth | head -1)
  echo "  detector: ${DETECTOR}  ($(grep 'best mean Dice' logs/pranet_detector.log | tail -1))"
fi
echo "Using detector checkpoint: ${DETECTOR}"

# ---- Stage 2: grading network ----
for i in "${!SEEDS[@]}"; do
  SPLIT_SEED="${SEEDS[$i]}"; RUN=$((i + 1))
  SAVE_DIR="./baseline/${NAME}_run${RUN}"
  LOG="logs/${NAME}_run${RUN}.log"
  echo "==== ${NAME} run ${RUN}/${N} (split_seed=${SPLIT_SEED}, ${EP}ep) ===="
  python baselines/train_pranet.py \
    --data_root "${DATA_ROOT}" \
    --detector_ckpt "${DETECTOR}" \
    --backbone efficientnetv2_m \
    --image_size 1024 \
    --num_nodes 12 \
    --epochs "${EP}" \
    --batch_size 4 \
    --lr 1e-4 \
    --backbone_lr 1e-5 \
    --weight_decay 0.01 \
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
