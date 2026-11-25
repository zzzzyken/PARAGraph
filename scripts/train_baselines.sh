#!/bin/bash
# Image-classifier baselines with the same repeated-split protocol as PARAGraph.
#
# Usage:  bash scripts/train_baselines.sh <model> <data_root> [name]
#
#   model       swinL | vitL | green
#   data_root   grading dataset folder in the fixed format (README, Data)
#   name        prefix of the outputs (default: <model>_<folder name of data_root>)
#
#   model   script                     input    default epochs / batch
#   swinL   baselines/train_swin.py    1024 px  30 / 4
#   vitL    baselines/train_vit.py      512 px  20 / 8   (global attention does not fit 1024 px)
#   green   baselines/train_dgdr.py    1024 px  50 / 4   (needs ../DGDR, see README)
#
# Settings can be changed through the environment:
#   SEEDS="42 0 1 2 1024"   split seeds, one run each (the model seed is fixed, --seed 1234)
#   EPOCHS, BATCH_SIZE      override the defaults above
#   LOSS_FLAGS              default "--use_focal" (focal loss); "--no_strong_aug" selects
#                           plain cross-entropy without the strong augmentation
#
# The defaults are the settings of the DDR runs. The Messidor-2 Swin-L runs used
#   EPOCHS=50 BATCH_SIZE=2 LOSS_FLAGS="--no_strong_aug" SEEDS="42 123 456 789 1024"
#
# Writes <save root>/<name>_run<N>/ and logs/<name>_run<N>.log.

set -e
cd "$(dirname "$0")/.."

USAGE="usage: bash scripts/train_baselines.sh <swinL|vitL|green> <data_root> [name]"
MODEL="${1:?$USAGE}"
DATA_ROOT="${2:?$USAGE}"
NAME="${3:-${MODEL}_$(basename "$DATA_ROOT")}"
read -r -a SEEDS <<< "${SEEDS:-42 0 1 2 1024}"
SAVE_ROOT="./checkpoints"

case "${MODEL}" in
  swinL)
    SCRIPT="baselines/train_swin.py"
    EPOCHS="${EPOCHS:-30}"; BATCH_SIZE="${BATCH_SIZE:-4}"; LOSS_FLAGS="${LOSS_FLAGS:---use_focal}"
    COMMON_ARGS="--image_size 1024 \
      --swin_model swin_large_patch4_window7_224 \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr 1e-4 \
      --backbone_lr 1e-6 \
      --weight_decay 0.01 \
      --scheduler cosine \
      --accumulation_steps 1 \
      --use_oversample \
      ${LOSS_FLAGS} \
      --focal_gamma 2.0 \
      --drop_rate 0.3 \
      --use_amp \
      --num_workers 4 \
      --seed 1234" ;;
  vitL)
    SCRIPT="baselines/train_vit.py"
    EPOCHS="${EPOCHS:-20}"; BATCH_SIZE="${BATCH_SIZE:-8}"; LOSS_FLAGS="${LOSS_FLAGS:---use_focal}"
    COMMON_ARGS="--image_size 512 \
      --vit_model vit_large_patch16_224 \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr 1e-4 \
      --backbone_lr 1e-6 \
      --weight_decay 0.01 \
      --scheduler cosine \
      --accumulation_steps 1 \
      --use_oversample \
      ${LOSS_FLAGS} \
      --focal_gamma 2.0 \
      --drop_rate 0.3 \
      --use_amp \
      --num_workers 4 \
      --seed 1234" ;;
  green)
    SCRIPT="baselines/train_dgdr.py"; SAVE_ROOT="./baseline"
    EPOCHS="${EPOCHS:-50}"; BATCH_SIZE="${BATCH_SIZE:-4}"; LOSS_FLAGS="${LOSS_FLAGS:---use_focal}"
    COMMON_ARGS="--algorithm GREEN \
      --backbone resnet50 \
      --image_size 1024 \
      --epochs ${EPOCHS} \
      --batch_size ${BATCH_SIZE} \
      --lr 1e-3 \
      --backbone_lr 1e-4 \
      --weight_decay 5e-4 \
      --momentum 0.9 \
      --optim sgd \
      --scheduler cosine \
      --accumulation_steps 1 \
      ${LOSS_FLAGS} \
      --focal_gamma 2.0 \
      --use_oversample \
      --num_workers 4 \
      --seed 1234" ;;
  *)
    echo "unknown model: ${MODEL} (swinL | vitL | green)"; exit 1 ;;
esac

mkdir -p logs
N=${#SEEDS[@]}

for i in "${!SEEDS[@]}"; do
  SPLIT_SEED="${SEEDS[$i]}"
  RUN=$((i + 1))
  SAVE_DIR="${SAVE_ROOT}/${NAME}_run${RUN}"
  LOG="logs/${NAME}_run${RUN}.log"
  echo "========================================"
  echo "[${NAME}] run ${RUN}/${N}  (split_seed=${SPLIT_SEED})"
  echo "========================================"
  python ${SCRIPT} \
    --data_root "${DATA_ROOT}" \
    $COMMON_ARGS \
    --split_seed "${SPLIT_SEED}" \
    --save_dir "${SAVE_DIR}" \
    > "${LOG}" 2>&1
  echo "[${NAME}] run ${RUN} done: $(grep "Best validation Kappa" "${LOG}" | tail -1)"
done

echo ""
echo "${NAME}: ${N} runs complete. Kappa summary:"
for i in $(seq 1 ${N}); do
  echo "  run ${i}: $(grep "Best validation Kappa" "logs/${NAME}_run${i}.log" 2>/dev/null | tail -1)"
done
