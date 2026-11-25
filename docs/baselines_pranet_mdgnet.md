# Re-implemented baselines: PRANet & MDGNet

Re-implemented for the comparison in the paper. **Neither paper released
official code** (Springer paywalled PRANet, Nature MDGNet codeless), so both are
reimplemented from the papers. Components the sources do not fully specify are
marked `INFERRED` in the code and summarized below — these are the places to
revisit if numbers diverge from the papers.

## PRANet
*"PRANet: Pathological relationship perception and dual attention guided network
for DR grading", Applied Intelligence 2025 (10.1007/s10489-025-07036-6).*
Reported: DDR ACC 0.8821 / QWK 0.9039 / AUC 0.9409. Datasets DDR, APTOS2019, FGADR.

- `Models/pranet.py`: `PRANetLesionDetector` (own U-Net detector) + `PRANet` grading net.
- **Two-stage.** Stage 1 trains PRANet's *own* detector on DDR lesion masks
  (not the LesionUNet used by PARAGraph), then frozen. Stage 2 trains
  EfficientNetV2-M global branch + Dual Attention (DAM, channel+spatial) + a
  co-occurrence GCN (PRFM) over K-means lesion nodes, fused and classified.
- **Resolution: 1024px** (EffV2-M is convolutional → scales to our other models'
  scale with no architectural change; verified feasible).
- `INFERRED`: detector = compact U-Net (paper only says "lesion detection network
  / rough regions"); K = 12 lesion nodes (torch K-means on global-feature-map
  descriptors ++ 4-d lesion-type composition); co-occurrence adjacency =
  symmetric cosine affinity with self-loops + D^-1/2 normalization (paper gives
  no formula); 2-layer GCN, 256-d; CBAM-style DAM. Loss = focal CE (our shared loop).

**Run:** `LESION_ROOT=data/lesions bash scripts/train_pranet.sh data/ddr`
(Stage 1 once on the lesion data folder, then Stage 2 on the grading data folder, 5 split
seeds. Reuse an existing detector with
`DETECTOR=path/to/best_detector.pth bash scripts/train_pranet.sh data/ddr`.)

## MDGNet
*"A graph convolutional network with dynamic weight fusion of multi-scale local
features for DR grading", Scientific Reports 2024 (s41598-024-56389-4).*
Reported: APTOS ACC 84.31% / F1 69.69%; DDR ACC 81.25% / F1 59.18% (no QWK reported).

- `Models/mdgnet.py`: conv stem → [MSDF + ViG] pyramid → GAP → FC head. Image-only.
  - **MSDF** = MFE (parallel depthwise convs k=1/3/5/7) + MFF (avg+max global
    context → shared MLP → softmax weights per scale → weighted fuse).
  - **ViG** = KNN graph (cdist+topk, pure torch — no torch_geometric) +
    max-relative graph conv `y_i=FC[x_i, max_j(x_j−x_i)]` + FFN, residual.
- **Resolution: 1024px input, graph fixed at 14×14** (`--graph_hw`). The paper uses
  224px (stem/16 → 14×14 = 196 nodes). A dense KNN is O(N²) in nodes, so the stem
  adaptively pools to 14×14 before the ViG body: higher-res input feeds more stem
  detail while the graph stays bounded. (Satisfies "scale up if no adverse effect";
  bump `--graph_hw` to densify the graph.)
- `INFERRED`: 3-stage pyramid (dims 96/192/384, 1 block/stage); depthwise MFE;
  4-conv /16 stem; k=9 neighbours. Loss = focal CE (our shared loop).

**Run:** `bash scripts/train_mdgnet.sh data/ddr`  (5 split seeds; one call per dataset folder)

## Protocol (identical to every other baseline)
- `grading_dataloader` splits, SEEDS=(42 0 1 2 1024) vary `--split_seed`, model
  `--seed 1234` fixed. DDR: train+valid→80:20 split, fixed test. Mirrors `baselines/train_swin.py`.
- Metrics: QWK + ACC printed; `test_results.json` saves the confusion matrix →
  F1/SPE derived post-hoc exactly as for the other baselines.
- Both plug into the shared loop (`baselines/train_efficientnet.py`) via
  `forward(images)->{'logits',...}` + `.num_classes`.

## Smoke-tested
Forward passes verified on CPU at 224/512/1024 (MDGNet 4.3M params; PRANet 60.3M,
detector 7.8M frozen; PRANet detector [B,4,H,W]).
