"""
Pixel-level evaluation of a lesion segmentation network: precision, recall, F1,
false-positive rate and IoU per lesion class.

    python evaluation/eval_lesion_segmentation.py \
        --checkpoints ./checkpoints/lesion_unet_swint/best_model.pth \
        --data_root data/lesions --split test

Pass several checkpoints to compare them with the first one.
--data_root is a lesion dataset folder in the fixed format (see README, Data).
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import argparse
import os

import torch
import numpy as np
from torch.utils.data import DataLoader
from Models import LesionUNet
from segmentation_dataloader import MaskDataset, get_lesion_val_transform

LESION_NAMES = ['MA', 'HE', 'EX', 'SE']
THRESHOLD = 0.5
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(checkpoint_path, encoder='swin', image_size=1024):
    model = LesionUNet(
        num_lesions=4,
        encoder_pretrained=False,
        encoder_freeze=False,
        encoder_name=encoder,
        img_size=image_size,
    ).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.no_grad()
def evaluate(model, dataloader, threshold=THRESHOLD):
    # per-class accumulators: TP, FP, FN, TN (pixel-level)
    TP = np.zeros(4)
    FP = np.zeros(4)
    FN = np.zeros(4)
    TN = np.zeros(4)

    for batch in dataloader:
        imgs   = batch['image'].to(DEVICE)
        masks  = batch['mask'].to(DEVICE).float()   # [B, 4, H, W]

        out = model(imgs, mode='full')
        pred_prob = torch.sigmoid(out['masks'])      # [B, 4, H, W]
        pred_bin  = (pred_prob > threshold).float()

        for c in range(4):
            p = pred_bin[:, c]
            g = masks[:, c]
            TP[c] += (p * g).sum().item()
            FP[c] += (p * (1 - g)).sum().item()
            FN[c] += ((1 - p) * g).sum().item()
            TN[c] += ((1 - p) * (1 - g)).sum().item()

    results = {}
    for c, name in enumerate(LESION_NAMES):
        precision = TP[c] / (TP[c] + FP[c] + 1e-8)
        recall    = TP[c] / (TP[c] + FN[c] + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)
        fp_rate   = FP[c] / (FP[c] + TN[c] + 1e-8)   # FPR = FP / (FP + TN)
        iou       = TP[c] / (TP[c] + FP[c] + FN[c] + 1e-8)
        results[name] = {
            'Precision': precision,
            'Recall':    recall,
            'F1':        f1,
            'FPR':       fp_rate,
            'IoU':       iou,
        }
    return results


def print_results(name, results):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"{'Class':<6} {'Precision':>10} {'Recall':>10} {'F1':>8} {'FPR':>10} {'IoU':>8}")
    print(f"{'-'*60}")
    for cls, m in results.items():
        print(f"{cls:<6} {m['Precision']:>10.4f} {m['Recall']:>10.4f} {m['F1']:>8.4f} {m['FPR']:>10.6f} {m['IoU']:>8.4f}")
    mean_p  = np.mean([m['Precision'] for m in results.values()])
    mean_r  = np.mean([m['Recall']    for m in results.values()])
    mean_f1 = np.mean([m['F1']        for m in results.values()])
    mean_fpr= np.mean([m['FPR']       for m in results.values()])
    mean_iou= np.mean([m['IoU']       for m in results.values()])
    print(f"{'-'*60}")
    print(f"{'Mean':<6} {mean_p:>10.4f} {mean_r:>10.4f} {mean_f1:>8.4f} {mean_fpr:>10.6f} {mean_iou:>8.4f}")


def main():
    parser = argparse.ArgumentParser(description='Pixel-level evaluation of lesion segmentation checkpoints')
    parser.add_argument('--checkpoints', nargs='+', required=True, help='One or more LesionUNet checkpoints')
    parser.add_argument('--names', nargs='+', default=None, help='Display names (optional)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Lesion dataset folder in the fixed format (see README, Data)')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--encoder', type=str, default='swin', choices=['resnet50', 'swin'])
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    args = parser.parse_args()

    names = args.names or [os.path.basename(os.path.dirname(c)) or c for c in args.checkpoints]
    dataset = MaskDataset(args.data_root, args.split, LESION_NAMES,
                          get_lesion_val_transform(args.image_size), args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    all_results = []
    for name, ckpt in zip(names, args.checkpoints):
        model = load_model(ckpt, args.encoder, args.image_size)
        results = evaluate(model, loader, args.threshold)
        print_results(name, results)
        all_results.append((name, results))
        del model
        torch.cuda.empty_cache()

    ref_name, ref = all_results[0]
    for name, results in all_results[1:]:
        print(f"\n{'='*60}")
        print(f"  Delta ({name} - {ref_name})")
        print(f"{'='*60}")
        print(f"{'Class':<6} {'dPrecision':>12} {'dRecall':>10} {'dF1':>8} {'dFPR':>10}")
        print(f"{'-'*60}")
        for cls in LESION_NAMES:
            dp   = results[cls]['Precision'] - ref[cls]['Precision']
            dr   = results[cls]['Recall']    - ref[cls]['Recall']
            df1  = results[cls]['F1']        - ref[cls]['F1']
            dfpr = results[cls]['FPR']       - ref[cls]['FPR']
            print(f"{cls:<6} {dp:>+12.4f} {dr:>+10.4f} {df1:>+8.4f} {dfpr:>+10.6f}")


if __name__ == '__main__':
    main()
