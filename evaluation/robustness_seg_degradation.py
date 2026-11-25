#!/usr/bin/env python3
"""
Segmentation-degradation robustness curves (experiments C2 / C3) on DDR.

Injects controlled segmentation errors into the LesionUNet output at increasing
rates and measures PARAGraph's QWK as it degrades:
  - missed         : randomly drop a fraction of detected lesions
  - misclassify    : relabel a fraction of lesions to a wrong class
  - false_positive : add spurious wrong-class detections for a fraction of lesions

C2: run on PARAGraph -> a degrading curve. SwinL is a flat reference line
    (it does not use the segmentation), so its QWK is constant = its test score.
C3: run on a gating checkpoint vs a no-gating checkpoint (trained with
    --no_gating) and compare how the two curves separate as degradation grows.

Per image the segmentation is cached once; every (perturbation, rate) reuses it
and only re-runs graph building + GNN.

Usage:
    python evaluation/robustness_seg_degradation.py --ckpt <best_model.pth> \
        --data_root data/ddr --rates 0,0.1,0.2,0.3,0.5,0.7,1.0 --out robustness_out
"""
import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os, argparse, json
import numpy as np
import torch

from Models import PARAGraph
from grading_dataloader import build_dataset, get_val_transform
from lesion_intervene import (
    cache_segmentation, forward_cached, enumerate_lesions,
    remove_lesions, misclassify_lesions, add_false_positives,
    quadratic_weighted_kappa,
)

PERTURBS = {
    'missed': remove_lesions,
    'misclassify': misclassify_lesions,
    'false_positive': add_false_positives,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_root', required=True,
                    help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    ap.add_argument('--split_seed', type=int, default=None,
                    help='Only for folders without test.csv: seed of the held-out test part')
    ap.add_argument('--rates', default='0,0.1,0.2,0.3,0.5,0.7,1.0')
    ap.add_argument('--perturbs', default='missed,misclassify,false_positive')
    ap.add_argument('--n_imgs', type=int, default=-1, help='-1 = full test set')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', default='robustness_seg')
    ap.add_argument('--tag', default='paragraph', help='label for the output file')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)
    rates = [float(r) for r in args.rates.split(',')]
    perturbs = args.perturbs.split(',')
    rng = np.random.RandomState(args.seed)

    model = PARAGraph.load_from_checkpoint(args.ckpt, device=device,
                                           strict=False, verbose=True)
    model.eval()
    num_classes = model.num_classes

    test_set = build_dataset(args.data_root, 'test', get_val_transform(1024), 1024,
                             use_3_stages=False, split_seed=args.split_seed)
    # the DDR test set is ordered by class; shuffle so a subsample (n_imgs>0)
    # is representative across grades (full set n_imgs=-1 is order-independent).
    from torch.utils.data import DataLoader
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(test_set, batch_size=1, shuffle=True,
                        num_workers=4, generator=gen)

    # collect predictions per (perturb, rate); rate=0 shared as the clean baseline
    labels = []
    preds = {p: {r: [] for r in rates} for p in perturbs}
    clean_preds = []
    n_done = 0

    for batch in loader:
        if args.n_imgs > 0 and n_done >= args.n_imgs:
            break
        x = batch['image'].to(device)
        y = int(batch['label'][0])
        labels.append(y)

        od_c, lm_c, lesion_logits = cache_segmentation(model, x)
        lesions = enumerate_lesions(model, lesion_logits)
        K = len(lesions)

        clean_lg = forward_cached(model, x, od_c, lm_c, lesion_logits)
        clean_preds.append(int(clean_lg.argmax()))

        for p in perturbs:
            fn = PERTURBS[p]
            for r in rates:
                if r == 0.0:
                    preds[p][r].append(int(clean_lg.argmax()))
                    continue
                n_sel = int(round(r * K))
                if n_sel == 0 or K == 0:
                    preds[p][r].append(int(clean_lg.argmax()))
                    continue
                idxs = list(rng.choice(K, size=min(n_sel, K), replace=False))
                lg_pert = fn(lesion_logits, lesions, idxs, rng) \
                    if p != 'missed' else fn(lesion_logits, lesions, idxs)
                out_lg = forward_cached(model, x, od_c, lm_c, lg_pert)
                preds[p][r].append(int(out_lg.argmax()))

        n_done += 1
        if n_done % 25 == 0:
            print(f"  processed {n_done} images")

    # QWK per (perturb, rate)
    curves = {}
    for p in perturbs:
        curves[p] = {str(r): quadratic_weighted_kappa(labels, preds[p][r], num_classes)
                     for r in rates}
    clean_qwk = quadratic_weighted_kappa(labels, clean_preds, num_classes)

    res = {'tag': args.tag, 'n_images': n_done, 'rates': rates,
           'clean_qwk': clean_qwk, 'curves': curves}
    out_path = os.path.join(args.out, f'degradation_{args.tag}.json')
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=2)

    print(f"\n=== Degradation curves ({args.tag}, n={n_done}) ===")
    print(f"  clean QWK: {clean_qwk*100:.2f}")
    for p in perturbs:
        line = "  ".join(f"{r}:{curves[p][str(r)]*100:.1f}" for r in rates)
        print(f"  {p:14s} {line}")
    print(f"  saved -> {out_path}")


if __name__ == '__main__':
    main()
