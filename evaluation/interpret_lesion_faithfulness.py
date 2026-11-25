#!/usr/bin/env python3
"""
Lesion-level attribution + faithfulness (interpretability experiment D1).

For each image:
  1. Cache OD mask + fovea heatmap + lesion-mask logits (so the per-lesion
     intervention loop does not re-run the segmentation backbones).
  2. Enumerate detected lesions via the model's own extract_lesion_regions
     (so the lesions match exactly those used to build the graph).
  3. Per-lesion attribution = leave-one-out drop in the expected grade
     E[grade] = sum_k softmax(logits)_k * k  when that lesion is removed.
  4. Faithfulness via deletion / insertion curves (remove/add lesions in
     attribution order vs. random order) and the ERASER-style
     comprehensiveness / sufficiency metrics, aggregated over images.

Removal is done by setting the lesion's connected-component region in the
lesion-mask logits to a large negative value (so sigmoid -> 0 and the lesion
is no longer detected) — this mirrors a missed detection.

Usage:
    python evaluation/interpret_lesion_faithfulness.py --ckpt <best_model.pth> \
        --data_root data/ddr --n_imgs 100 --out interpret_out
"""
import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os, argparse, json
import numpy as np
import torch
import torch.nn.functional as F

from Models import PARAGraph
from grading_dataloader import build_dataset, get_val_transform
from lesion_intervene import (
    NEG, expected_grade, cache_segmentation, enumerate_lesions, remove_lesions,
    forward_cached as _forward_cached,
)


@torch.no_grad()
def forward_cached(model, x, od_c, lm_c, lesion_logits, target='anatomical'):
    """Forward with cached segmentation; returns (target_logits[C], expected_grade).
    target='anatomical' measures the graph branch (which uses the lesions);
    target='logits' measures the full dual-branch prediction (global-dominated)."""
    out = model(x, od_mask_cached=od_c, landmark_heatmap_cached=lm_c,
                lesion_masks_cached=lesion_logits)
    key = 'anatomical_logits' if target == 'anatomical' else 'logits'
    lg = out[key][0]
    return lg, expected_grade(lg)


def _trapz(y, x):
    """Version-independent trapezoidal integration (np.trapz removed in numpy 2)."""
    y = np.asarray(y); x = np.asarray(x)
    return float(np.sum((y[:-1] + y[1:]) / 2.0 * np.diff(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_root', required=True,
                    help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    ap.add_argument('--split_seed', type=int, default=None,
                    help='Only for folders without test.csv: seed of the held-out test part')
    ap.add_argument('--n_imgs', type=int, default=100)
    ap.add_argument('--max_lesions', type=int, default=40,
                    help='cap lesions per image for the curves (speed)')
    ap.add_argument('--out', default='interpret_faithfulness')
    ap.add_argument('--target', default='anatomical', choices=['anatomical', 'logits'],
                    help="branch to measure: 'anatomical' (graph, uses lesions) or "
                         "'logits' (full dual-branch, global-dominated)")
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    model = PARAGraph.load_from_checkpoint(args.ckpt, device=device,
                                           strict=False, verbose=True)
    model.eval()
    img_size = model.optic_unet.img_size if hasattr(model.optic_unet, 'img_size') else 1024

    test_set = build_dataset(args.data_root, 'test', get_val_transform(img_size), img_size,
                             use_3_stages=False, split_seed=args.split_seed)
    # DDR test is ordered by class; shuffle so a subsample hits images that
    # actually contain lesions (grade 0 images have few/none).
    from torch.utils.data import DataLoader
    gen = torch.Generator().manual_seed(0)
    loader = DataLoader(test_set, batch_size=1, shuffle=True,
                        num_workers=4, generator=gen)

    # aggregates: deletion/insertion expected-grade curves on a normalized x-axis
    GRID = np.linspace(0, 1, 21)   # fraction of lesions removed/inserted
    del_imp = []   # delete in importance order
    del_rnd = []   # delete in random order
    ins_imp = []   # insert in importance order
    comp_list, suff_list = [], []
    n_done = 0

    def _save():
        """Build the result dict from current accumulators and flush to disk.
        Called every 10 images AND at the end, so a crash never loses completed
        work (deletion/insertion curves are running means -> a partial file is
        already a valid, usable result)."""
        os.makedirs(args.out, exist_ok=True)
        res = {
            'n_images': n_done,
            'grid': GRID.tolist(),
            'deletion_importance': np.mean(del_imp, axis=0).tolist() if del_imp else [],
            'deletion_random': np.mean(del_rnd, axis=0).tolist() if del_rnd else [],
            'insertion_importance': np.mean(ins_imp, axis=0).tolist() if ins_imp else [],
            'comprehensiveness': float(np.mean(comp_list)) if comp_list else None,
            'sufficiency': float(np.mean(suff_list)) if suff_list else None,
            'deletion_gap_auc': (_trapz(np.mean(del_rnd, axis=0) - np.mean(del_imp, axis=0), GRID)
                                 if del_imp else None),
        }
        with open(os.path.join(args.out, 'faithfulness.json'), 'w') as f:
            json.dump(res, f, indent=2)
        return res

    for batch in loader:
        if n_done >= args.n_imgs:
            break
        x = batch['image'].to(device)
        od_c, lm_c, lesion_logits = cache_segmentation(model, x)

        lesions = enumerate_lesions(model, lesion_logits)
        if len(lesions) < 3:
            continue  # need a few lesions for a meaningful curve
        if len(lesions) > args.max_lesions:
            # keep the highest-confidence ones (cap for speed)
            lesions = sorted(lesions, key=lambda l: l['conf'], reverse=True)[:args.max_lesions]
        K = len(lesions)

        # base prediction (all lesions present)
        base_lg, base_eg = forward_cached(model, x, od_c, lm_c, lesion_logits, args.target)
        # empty prediction (all lesions removed)
        empty_lg = remove_lesions(lesion_logits, lesions, list(range(K)))
        _, empty_eg = forward_cached(model, x, od_c, lm_c, empty_lg, args.target)

        # leave-one-out attribution: |E[grade] base - E[grade] without lesion i|
        attr = np.zeros(K)
        for i in range(K):
            lg_i = remove_lesions(lesion_logits, lesions, [i])
            _, eg_i = forward_cached(model, x, od_c, lm_c, lg_i, args.target)
            attr[i] = base_eg - eg_i
        order_imp = list(np.argsort(-np.abs(attr)))        # most important first
        rng = np.random.RandomState(0)
        order_rnd = list(rng.permutation(K))

        # deletion curves: remove first m lesions (in order), record E[grade]
        def del_curve(order):
            ys = [base_eg]
            for m in range(1, K + 1):
                lg_m = remove_lesions(lesion_logits, lesions, order[:m])
                _, eg = forward_cached(model, x, od_c, lm_c, lg_m, args.target)
                ys.append(eg)
            xs = np.linspace(0, 1, K + 1)
            return np.interp(GRID, xs, ys)
        # insertion curve (importance order): start empty, add lesions
        def ins_curve(order):
            ys = [empty_eg]
            for m in range(1, K + 1):
                keep = order[:m]
                drop = [i for i in range(K) if i not in keep]
                lg_m = remove_lesions(lesion_logits, lesions, drop)
                _, eg = forward_cached(model, x, od_c, lm_c, lg_m, args.target)
                ys.append(eg)
            xs = np.linspace(0, 1, K + 1)
            return np.interp(GRID, xs, ys)

        del_imp.append(del_curve(order_imp))
        del_rnd.append(del_curve(order_rnd))
        ins_imp.append(ins_curve(order_imp))

        # comprehensiveness: drop in prob of base class when top-20% removed
        # sufficiency: prob of base class kept when only top-20% retained
        base_cls = int(base_lg.argmax())
        base_p = float(F.softmax(base_lg, dim=0)[base_cls])
        k20 = max(1, int(0.2 * K))
        lg_comp = remove_lesions(lesion_logits, lesions, order_imp[:k20])
        comp_lg, _ = forward_cached(model, x, od_c, lm_c, lg_comp, args.target)
        comp = base_p - float(F.softmax(comp_lg, dim=0)[base_cls])
        keep = order_imp[:k20]; drop = [i for i in range(K) if i not in keep]
        lg_suff = remove_lesions(lesion_logits, lesions, drop)
        suff_lg, _ = forward_cached(model, x, od_c, lm_c, lg_suff, args.target)
        suff = base_p - float(F.softmax(suff_lg, dim=0)[base_cls])
        comp_list.append(comp); suff_list.append(suff)

        n_done += 1
        if n_done % 10 == 0:
            _save()  # incremental flush: a crash never loses completed images
            print(f"  processed {n_done} images (lesions in last: {K}) [saved]")

    res = _save()
    print("\n=== Faithfulness summary ===")
    print(f"  images: {n_done}")
    print(f"  comprehensiveness (higher=better): {res['comprehensiveness']}")
    print(f"  sufficiency       (lower =better): {res['sufficiency']}")
    print(f"  deletion gap AUC  (higher=better): {res['deletion_gap_auc']}")
    print(f"  saved -> {os.path.join(args.out, 'faithfulness.json')}")


if __name__ == '__main__':
    main()
