"""
Shared lesion-intervention primitives for the DDR interpretability / robustness
experiments.

The model consumes precomputed lesion-mask LOGITS via `lesion_masks_cached`.
A lesion is "removed" by setting its connected-component region to a large
negative logit (sigmoid -> 0, no longer detected = a missed detection).
A lesion is "misclassified" by removing it from its true class channel and
stamping it into a wrong class channel; a "false positive" stamps an extra
wrong-class detection co-located with a real lesion.

These primitives back:
  - D1 lesion-level faithfulness (deletion / insertion)
  - C2 segmentation-degradation curves (missed / misclassify / false-positive)
  - C3 gating under degradation (run the same curves on gating vs no-gating ckpts)
"""
import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import numpy as np
import torch
import torch.nn.functional as F

NEG = -20.0   # logit -> sigmoid ~ 0 (lesion removed)
POS = 10.0    # logit -> sigmoid ~ 1 (lesion stamped in)


def expected_grade(logits: torch.Tensor) -> float:
    """E[grade] = sum_k softmax(logits)_k * k (smooth, ordinal-aware)."""
    p = F.softmax(logits.view(-1), dim=0)
    k = torch.arange(p.numel(), device=p.device, dtype=p.dtype)
    return float((p * k).sum())


@torch.no_grad()
def cache_segmentation(model, x):
    """Run the segmentation front-ends once; return cached OD/fovea/lesion logits.
    Returns (od_c [1,H,W], lm_c [1,H,W], lesion_logits [1,4,H,W])."""
    optic_features, _ = model.optic_unet.encode(x)
    od_mask = model.optic_unet.decode_optic_disc(optic_features)
    landmark = model.optic_unet.decode_fovea(optic_features)
    lesion_feats, _ = model.lesion_unet.encode(x)
    lesion_masks = model.lesion_unet.decode(lesion_feats)
    return od_mask.squeeze(1), landmark.squeeze(1), lesion_masks


@torch.no_grad()
def forward_cached(model, x, od_c, lm_c, lesion_logits):
    """Forward with cached segmentation; returns the [num_classes] logits."""
    out = model(x, od_mask_cached=od_c, landmark_heatmap_cached=lm_c,
                lesion_masks_cached=lesion_logits)
    return out['logits'][0]


def enumerate_lesions(model, lesion_logits):
    """List deletable lesions via the model's own extractor (matches the graph)."""
    probs = torch.sigmoid(lesion_logits[0]).cpu().numpy()   # [4,H,W]
    lesions = []
    for cls in range(4):
        prob_np = probs[cls]
        class_mask = (prob_np > model.lesion_threshold)
        for r in model.extract_lesion_regions(class_mask, prob_np, cls):
            lesions.append({'class': cls, 'mask': r['mask'],
                            'area': r['area'], 'conf': r['confidence'],
                            'center': r['center']})
    return lesions


def remove_lesions(lesion_logits, lesions, idxs):
    """Copy of lesion_logits with the given lesions removed (set to NEG)."""
    lg = lesion_logits.clone()
    for i in idxs:
        les = lesions[i]
        m = torch.from_numpy(les['mask']).to(lg.device)
        lg[0, les['class']][m] = NEG
    return lg


def misclassify_lesions(lesion_logits, lesions, idxs, rng):
    """Move the selected lesions to a random wrong class channel."""
    lg = lesion_logits.clone()
    for i in idxs:
        les = lesions[i]; cls = les['class']
        m = torch.from_numpy(les['mask']).to(lg.device)
        lg[0, cls][m] = NEG
        wrong = int(rng.choice([c for c in range(4) if c != cls]))
        lg[0, wrong][m] = POS
    return lg


def add_false_positives(lesion_logits, lesions, idxs, rng):
    """Stamp an extra spurious wrong-class detection co-located with each
    selected lesion (a plausible false positive)."""
    lg = lesion_logits.clone()
    for i in idxs:
        les = lesions[i]; cls = les['class']
        m = torch.from_numpy(les['mask']).to(lg.device)
        wrong = int(rng.choice([c for c in range(4) if c != cls]))
        lg[0, wrong][m] = POS
    return lg


def quadratic_weighted_kappa(y_true, y_pred, num_classes):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    O = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1
    w = np.fromfunction(lambda i, j: (i - j) ** 2 / (num_classes - 1) ** 2,
                        (num_classes, num_classes))
    act = O.sum(1); pred = O.sum(0); tot = O.sum()
    E = np.outer(act, pred) / max(tot, 1)
    denom = (w * E).sum()
    return 1 - (w * O).sum() / denom if denom > 0 else 0.0
