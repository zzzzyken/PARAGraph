#!/bin/bash
# Train PARAGraph: several runs that differ only in the data split.
#
# Usage:  bash scripts/train_paragraph.sh <data_root> [name]
#
#   data_root   grading dataset folder in the fixed format (README, Data)
#   name        prefix of the outputs (default: the folder name of data_root)
#
# Every run passes its seed as --split_seed:
#   folder with test.csv     the test set is fixed; train.csv is re-split 80:20 into train / val
#   folder without test.csv  a stratified 20% of train.csv is held out as the test set
#
# Settings can be changed through the environment:
#   SEEDS="42 0 1 2 1024"   split seeds, one run each (the model seed is fixed, --seed 1234)
#   EPOCHS=50
#
# Needs the two segmentation checkpoints (README, Training, steps 1 to 3):
#   checkpoints/optic_unet_resnet50/best_model.pth
#   checkpoints/lesion_unet_swint/best_model.pth
# Writes checkpoints/paragraph_<name>_run<N>/ and logs/paragraph_<name>_run<N>.log.

set -e
cd "$(dirname "$0")/.."

DATA_ROOT="${1:?usage: bash scripts/train_paragraph.sh <data_root> [name]}"
NAME="${2:-$(basename "$DATA_ROOT")}"
read -r -a SEEDS <<< "${SEEDS:-42 0 1 2 1024}"
EPOCHS="${EPOCHS:-50}"

COMMON_ARGS="--image_size 1024 \
  --optic_checkpoint ./checkpoints/optic_unet_resnet50/best_model.pth \
  --lesion_checkpoint ./checkpoints/lesion_unet_swint/best_model.pth \
  --optic_encoder resnet50 \
  --lesion_encoder swin \
  --feature_dim 256 \
  --num_gnn_layers 3 \
  --gnn_dropout 0.3 \
  --num_region_nodes 4 \
  --use_prior_fusion \
  --fusion_mode full \
  --use_optic_nodes \
  --use_gating \
  --epochs ${EPOCHS} \
  --batch_size 8 \
  --lr 1e-4 \
  --backbone_lr 1e-5 \
  --weight_decay 0.05 \
  --scheduler cosine \
  --accumulation_steps 1 \
  --freeze_backbone \
  --unfreeze_epoch $((EPOCHS + 1)) \
  --use_oversample \
  --use_amp \
  --num_workers 4 \
  --seed 1234 \
  --save_every 10"

mkdir -p logs
N=${#SEEDS[@]}

for i in "${!SEEDS[@]}"; do
  SPLIT_SEED="${SEEDS[$i]}"
  RUN=$((i + 1))
  SAVE_DIR="./checkpoints/paragraph_${NAME}_run${RUN}"
  LOG="logs/paragraph_${NAME}_run${RUN}.log"
  echo "========================================"
  echo "[PARAGraph/${NAME}] run ${RUN}/${N}  (split_seed=${SPLIT_SEED})"
  echo "========================================"
  python train_paragraph.py \
    --data_root "${DATA_ROOT}" \
    $COMMON_ARGS \
    --split_seed "${SPLIT_SEED}" \
    --save_dir "${SAVE_DIR}" \
    > "${LOG}" 2>&1
  echo "run ${RUN} done: $(grep "Best validation Kappa" "${LOG}" | tail -1)"
done

echo ""
echo "All ${N} runs complete. Kappa summary:"
for i in $(seq 1 ${N}); do
  echo "  run ${i}: $(grep "Best validation Kappa" "logs/paragraph_${NAME}_run${i}.log" | tail -1)"
done
