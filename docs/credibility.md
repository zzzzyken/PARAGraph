# PARAGraph — Credibility Experiments (Interpretability + Robustness)

Scripts to run the DDR **faithfulness** and **segmentation-robustness** experiments
on a PARAGraph checkpoint. No training required — these are inference-only analyses.

## 1. What's needed

**Checkpoints** (not distributed; train them with the pipeline in the main README; expected under `checkpoints/`):
- `optic_unet_resnet50/best_model.pth`  — OD + fovea segmentation (frozen)
- `lesion_unet_swint/best_model.pth`    — 4-class lesion segmentation (frozen)
- `paragraph_ddr_run1/*/best_model.pth` — the trained PARAGraph model to analyze

> No vessel checkpoint is needed. The PARAGraph checkpoint contains no
> `vessel_unet.*` weights, so `load_from_checkpoint` never builds a VesselUNet.

**Dataset** (not included): a grading dataset folder in the fixed format described in the
main README. Its test split is analysed. The commands below assume the DDR folder at
`data/ddr` (the DDR test split has 3759 gradable images).

**Environment**: same as the training env (PyTorch, timm, opencv, scikit-learn,
albumentations, numpy). A GPU is recommended but CPU works (slow).

## 2. Scripts

All three files are in `evaluation/`. Run the commands from the repository root.

| File | Experiment |
|------|------------|
| `lesion_intervene.py` | shared lesion-intervention primitives (imported by the others) |
| `robustness_seg_degradation.py` | C2/C3 — segmentation-degradation robustness curves |
| `interpret_lesion_faithfulness.py` | D1 — lesion-level attribution + faithfulness |

Set the checkpoint path once:
```bash
CKPT=$(ls checkpoints/paragraph_ddr_run1/*/best_model.pth | head -1)
```

### C2 — Robustness to segmentation degradation (full DDR test set)
Injects missed detections / misclassifications / false positives into the
LesionUNet output at increasing rates and measures QWK.
```bash
python evaluation/robustness_seg_degradation.py \
    --ckpt "$CKPT" --data_root data/ddr \
    --rates 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 \
    --perturbs missed,misclassify,false_positive \
    --n_imgs -1 --seed 0 --tag paragraph_ddr_run1 \
    --out robustness_ddr
# -> robustness_ddr/degradation_paragraph_ddr_run1.json
```
Expected (reference run): clean QWK ~88.7; `missed` degrades gracefully
(88.7 -> 59.5 at 100%); `misclassify` and `false_positive` stay ~flat (~88).

**C3 (gating ablation under degradation):** train a no-gating model with
`--no_gating` (in `train_paragraph.py`), then run the same command on that
checkpoint with `--tag paragraph_nogating`, and compare the two curves.

### D1 — Lesion-level faithfulness (deletion / insertion)
Per image: leave-one-out attribution of each lesion, then deletion/insertion
curves (importance order vs random) + comprehensiveness / sufficiency.
```bash
python evaluation/interpret_lesion_faithfulness.py \
    --ckpt "$CKPT" --data_root data/ddr \
    --n_imgs 300 --max_lesions 30 --target logits \
    --out interpret_ddr
# -> interpret_ddr/faithfulness.json
```
- `--target logits`     : faithfulness of the full dual-branch prediction
- `--target anatomical` : faithfulness of the graph branch only
Expected (reference, n=20 smoke): comprehensiveness ~0.10, sufficiency ~0.07,
deletion-gap-AUC ~0.21 (positive = importance order degrades faster than random
= faithful attribution).

## 3. Notes / gotchas
- The DDR test set is **ordered by grade** (all grade-0 first); both scripts
  internally shuffle so a sub-sample (`--n_imgs N`) is representative across grades.
  Grade-0 images have few/no lesions, so faithfulness/robustness only show signal
  on a representative (shuffled) sample. Use `--n_imgs -1` for the full set.
- `load_from_checkpoint` loads with `strict=False`; older checkpoints carry a few
  now-removed module keys (`optic_feature_proj.*`, `lesion_feature_proj.*`) that
  are harmlessly ignored — this is expected, not an error.
- All analyses are inference-only; nothing is retrained.
