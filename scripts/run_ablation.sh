#!/bin/bash
# PARAGraph ablations: each configuration is one flag of train_paragraph.py, three runs each.
#
# Usage:  bash scripts/run_ablation.sh <data_root> [config ...]
#
#   data_root   grading dataset folder in the fixed format (README, Data)
#   config      any of L0 L1 L2 ODoff GLoff PFoff (default: all six)
#
#   L0     --no_lesion_unet     only the OD and global nodes
#   L1     --lesion_direct      lesions connect straight to the hubs (no category / region nodes)
#   L2     --no_region          category nodes but no region nodes
#   ODoff  --no_od_posbias      no OD positional bias in the Swin encoder
#   GLoff  --no_global_logits   graph branch only
#   PFoff  --no_prior_fusion    no prior fusion
#
# Settings can be changed through the environment:
#   SEEDS="42 0 1"   split seeds, one run each
#   EPOCHS=50        (the DDR ablations of the paper used EPOCHS=30)
#   NAME=...         prefix of the outputs (default: the folder name of data_root)
#
# The full model is the reference: scripts/train_paragraph.sh with the same seeds.
# Writes checkpoints/ablation_<name>_<config>_run<N>/ and logs/ablation_<name>_<config>_run<N>.log.

set -e
cd "$(dirname "$0")/.."

DATA_ROOT="${1:?usage: bash scripts/run_ablation.sh <data_root> [config ...]}"
shift
NAME="${NAME:-$(basename "$DATA_ROOT")}"
read -r -a SEEDS <<< "${SEEDS:-42 0 1}"
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
  --seed 1234"

declare -a CONFIGS=(
  "L0:--no_lesion_unet"
  "L1:--lesion_direct"
  "L2:--no_region"
  "ODoff:--no_od_posbias"
  "GLoff:--no_global_logits"
  "PFoff:--no_prior_fusion"
)

# keep only the configurations named on the command line, if any
if [ "$#" -gt 0 ]; then
  declare -a SELECTED=()
  for want in "$@"; do
    found=0
    for cfg in "${CONFIGS[@]}"; do
      if [ "${cfg%%:*}" = "$want" ]; then SELECTED+=("$cfg"); found=1; fi
    done
    [ "$found" = 1 ] || { echo "unknown config: $want (L0 L1 L2 ODoff GLoff PFoff)"; exit 1; }
  done
  CONFIGS=("${SELECTED[@]}")
fi

mkdir -p logs
N=${#SEEDS[@]}

for cfg in "${CONFIGS[@]}"; do
  CFG="${cfg%%:*}"
  FLAG="${cfg#*:}"
  for i in "${!SEEDS[@]}"; do
    SPLIT_SEED="${SEEDS[$i]}"
    RUN=$((i + 1))
    SAVE_DIR="./checkpoints/ablation_${NAME}_${CFG}_run${RUN}"
    LOG="logs/ablation_${NAME}_${CFG}_run${RUN}.log"
    echo "========================================"
    echo "[Ablation ${CFG}/${NAME}] run ${RUN}/${N} (split_seed=${SPLIT_SEED}) flag='${FLAG}'"
    echo "========================================"
    python train_paragraph.py \
      --data_root "${DATA_ROOT}" \
      $COMMON_ARGS \
      $FLAG \
      --split_seed "${SPLIT_SEED}" \
      --save_dir "${SAVE_DIR}" \
      > "${LOG}" 2>&1
    echo "[Ablation ${CFG}/${NAME}] run ${RUN} done: $(grep "Best validation Kappa" "${LOG}" | tail -1)"
  done
done

echo ""
echo "Ablations on ${NAME} complete. Summary:"
for cfg in "${CONFIGS[@]}"; do
  CFG="${cfg%%:*}"
  for i in $(seq 1 ${N}); do
    echo "  ${CFG} run${i}: $(grep "Best validation Kappa" "logs/ablation_${NAME}_${CFG}_run${i}.log" 2>/dev/null | tail -1)"
  done
done
