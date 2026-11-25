#!/usr/bin/env python3
"""
GT-grounded lesion faithfulness (paper protocol).

Same as interpret_lesion_faithfulness.py but, per the paper's credibility
section, the model's predicted lesion masks are REPLACED with pixel-level
ground-truth lesion masks before building the graph:
only test images of the grading dataset that also appear in the lesion
dataset are used, GT masks (MA/HE/EX/SE) are converted to lesion-logit maps
(+8 inside / -8 outside), and every lesion enumerated for the deletion /
insertion analysis is therefore a ground-truth lesion. OD/fovea remain
predicted (the paper replaces lesion masks only).

Saves incrementally every 10 images (a crash never loses completed work).

Usage:
    python evaluation/interpret_lesion_faithfulness_gt.py --ckpt <best_model.pth> \
        --data_root data/ddr --seg_root data/lesions --out interpret_out_gt

--data_root is a grading dataset folder and --seg_root a lesion dataset folder, both in
the fixed format (see README, Data). Images are matched by file name.
"""
import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os, argparse, json
import numpy as np
import torch
from PIL import Image

from Models import PARAGraph
from torch.utils.data import DataLoader

from grading_dataloader import build_dataset, get_val_transform
from segmentation_dataloader import IMAGE_EXTENSIONS, index_masks
from lesion_intervene import (
    NEG, expected_grade, cache_segmentation, enumerate_lesions, remove_lesions,
)
from interpret_lesion_faithfulness import forward_cached, _trapz

LESION_NAMES = ['MA', 'HE', 'EX', 'SE']   # channel order of LesionUNet


def build_gt_index(seg_root):
    """{image name without extension: {lesion: mask path}} over all splits of a lesion dataset folder.

    An image without a mask file for some lesion class has no lesion of that class.
    """
    idx = {}
    image_root = os.path.join(seg_root, 'images')
    for split in sorted(os.listdir(image_root)):
        if not os.path.isdir(os.path.join(image_root, split)):
            continue
        masks = {les: index_masks(seg_root, split, les) for les in LESION_NAMES}
        for name in os.listdir(os.path.join(image_root, split)):
            if not name.lower().endswith(IMAGE_EXTENSIONS):
                continue
            stem = os.path.splitext(name)[0]
            idx[stem] = {les: masks[les].get(stem) for les in LESION_NAMES}
    return idx


def gt_lesion_logits(mask_paths, hw, device):
    """Load GT masks -> lesion-logit map [1,4,H,W] (+8 lesion / -8 background)."""
    H, W = hw
    chans = []
    for les in LESION_NAMES:
        p = mask_paths.get(les)
        if p and os.path.exists(p):
            m = Image.open(p).convert('L').resize((W, H), Image.NEAREST)
            m = (np.asarray(m) > 127).astype(np.float32)   # masks stored as JPEG
        else:
            m = np.zeros((H, W), dtype=np.float32)
        chans.append(m)
    mask = torch.from_numpy(np.stack(chans))               # [4,H,W]
    return (mask * 16.0 - 8.0).unsqueeze(0).to(device)     # logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--data_root', required=True,
                    help='Grading dataset folder in the fixed format (its test split is used)')
    ap.add_argument('--seg_root', required=True,
                    help='Lesion dataset folder in the fixed format, source of the ground-truth masks')
    ap.add_argument('--split_seed', type=int, default=None,
                    help='Only for folders without test.csv: seed of the held-out test part')
    ap.add_argument('--max_lesions', type=int, default=40)
    ap.add_argument('--out', default='interpret_out_gt')
    ap.add_argument('--target', default='anatomical', choices=['anatomical', 'logits'])
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.out, exist_ok=True)

    model = PARAGraph.load_from_checkpoint(args.ckpt, device=device,
                                           strict=False, verbose=True)
    model.eval()
    img_size = model.optic_unet.img_size if hasattr(model.optic_unet, 'img_size') else 1024

    gt_index = build_gt_index(args.seg_root)
    print(f"[GT] {len(gt_index)} segmentation-annotated images indexed")

    test_set = build_dataset(args.data_root, 'test', get_val_transform(img_size), img_size,
                             use_3_stages=False, split_seed=args.split_seed)
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)   # iterate all, filter by GT

    GRID = np.linspace(0, 1, 21)
    del_imp, del_rnd, ins_imp = [], [], []
    comp_list, suff_list = [], []
    n_done = 0

    def _save():
        os.makedirs(args.out, exist_ok=True)
        res = {
            'n_images': n_done,
            'protocol': 'gt_grounded',
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

    import torch.nn.functional as F  # noqa: F401 (parity with base script)
    with torch.no_grad():
        for batch in loader:
            image_id = os.path.splitext(batch['image_id'][0])[0]
            if image_id not in gt_index:
                continue
            x = batch['image'].to(device)
            od_c, lm_c, _pred = cache_segmentation(model, x)
            H, W = _pred.shape[-2:]
            lesion_logits = gt_lesion_logits(gt_index[image_id], (H, W), device)

            lesions = enumerate_lesions(model, lesion_logits)
            if len(lesions) < 3:
                continue
            if len(lesions) > args.max_lesions:
                lesions = sorted(lesions, key=lambda l: l['conf'], reverse=True)[:args.max_lesions]
            K = len(lesions)

            base_lg, base_eg = forward_cached(model, x, od_c, lm_c, lesion_logits, args.target)
            empty_lg_map = remove_lesions(lesion_logits, lesions, list(range(K)))
            _, empty_eg = forward_cached(model, x, od_c, lm_c, empty_lg_map, args.target)

            attr = np.zeros(K)
            for i in range(K):
                lg_i = remove_lesions(lesion_logits, lesions, [i])
                _, eg_i = forward_cached(model, x, od_c, lm_c, lg_i, args.target)
                attr[i] = base_eg - eg_i
            order_imp = list(np.argsort(-np.abs(attr)))
            rng = np.random.RandomState(0)
            order_rnd = list(rng.permutation(K))

            def del_curve(order):
                ys = [base_eg]
                for m in range(1, K + 1):
                    lg_m = remove_lesions(lesion_logits, lesions, order[:m])
                    _, eg = forward_cached(model, x, od_c, lm_c, lg_m, args.target)
                    ys.append(eg)
                return np.interp(GRID, np.linspace(0, 1, K + 1), ys)

            def ins_curve(order):
                ys = [empty_eg]
                for m in range(1, K + 1):
                    keep = order[:m]
                    drop = [i for i in range(K) if i not in keep]
                    lg_m = remove_lesions(lesion_logits, lesions, drop)
                    _, eg = forward_cached(model, x, od_c, lm_c, lg_m, args.target)
                    ys.append(eg)
                return np.interp(GRID, np.linspace(0, 1, K + 1), ys)

            del_imp.append(del_curve(order_imp))
            del_rnd.append(del_curve(order_rnd))
            ins_imp.append(ins_curve(order_imp))

            import torch.nn.functional as F
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
                _save()
                print(f"  processed {n_done} GT images (lesions in last: {K}) [saved]", flush=True)

    res = _save()
    print("\n=== GT-grounded faithfulness summary ===")
    print(f"  images: {n_done}")
    print(f"  comprehensiveness (higher=better): {res['comprehensiveness']}")
    print(f"  sufficiency       (lower =better): {res['sufficiency']}")
    print(f"  deletion gap AUC  (higher=better): {res['deletion_gap_auc']}")
    print(f"  saved -> {os.path.join(args.out, 'faithfulness.json')}")


if __name__ == '__main__':
    main()
