#!/usr/bin/env python3
"""
PRANet Stage 1: train PRANet's OWN lesion detector on DDR lesion masks.

This is PRANet's own detection network (Models/pranet.PRANetLesionDetector), a
compact U-Net segmenter over the 4 lesion types (MA/HE/EX/SE). It is trained
once on DDR's pixel-level lesion masks, then FROZEN and reused by the PRANet
grading network (train_pranet.py Stage 2) to build lesion nodes on every
grading dataset. It is intentionally separate from the project LesionUNet.

Loss: BCE-with-logits (per-lesion pos_weight) + soft Dice.
Selection: best mean per-lesion Dice on the validation set.
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import json
import argparse
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Models import PRANetLesionDetector
from segmentation_dataloader import create_lesion_dataloaders

LESION_NAMES = ['MA', 'HE', 'EX', 'SE']


def dice_loss(probs, gt, eps=1e-6):
    probs = probs.flatten(2)
    gt = gt.flatten(2).float()
    inter = (probs * gt).sum(-1)
    denom = probs.sum(-1) + gt.sum(-1)
    dice = (2 * inter + eps) / (denom + eps)
    return 1 - dice.mean()


@torch.no_grad()
def eval_dice(model, loader, device, thresh=0.5, eps=1e-6):
    model.eval()
    inter = np.zeros(4); union = np.zeros(4)
    for batch in loader:
        img = batch['image'].to(device)
        gt = batch['mask'].to(device).float()
        pred = (torch.sigmoid(model(img)['masks']) > thresh).float()
        for c in range(4):
            p = pred[:, c].flatten(1); g = gt[:, c].flatten(1)
            inter[c] += (p * g).sum().item()
            union[c] += p.sum().item() + g.sum().item()
    dice = (2 * inter + eps) / (union + eps)
    return dice, dice.mean()


def main(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'pranet_detector_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    loaders = create_lesion_dataloaders(data_root=args.data_root, image_size=args.image_size,
                                 batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"train batches: {len(loaders['train'])}, val: {len(loaders.get('val', []))}")

    model = PRANetLesionDetector(base=args.base).to(device)
    print(f"detector params: {sum(p.numel() for p in model.parameters()):,}")

    # per-lesion positive weights (lesions are sparse) -- estimate from a few batches
    pos_w = torch.tensor([10., 5., 5., 10.], device=device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_dice = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in loaders['train']:
            img = batch['image'].to(device)
            gt = batch['mask'].to(device).float()
            logits = model(img)['masks']
            bce = F.binary_cross_entropy_with_logits(logits, gt, pos_weight=pos_w)
            dsc = dice_loss(torch.sigmoid(logits), gt)
            loss = bce + dsc
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        scheduler.step()
        val_src = loaders.get('val') or loaders.get('test')
        dice, mdice = eval_dice(model, val_src, device)
        print(f"Epoch {epoch}/{args.epochs}  loss={running/len(loaders['train']):.4f}  "
              f"val Dice: " + " ".join(f"{n}={d:.3f}" for n, d in zip(LESION_NAMES, dice)) +
              f"  mean={mdice:.3f}")
        if mdice > best_dice:
            best_dice = mdice
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'mean_dice': float(mdice), 'args': vars(args)},
                       os.path.join(save_dir, 'best_detector.pth'))
            print(f"  -> saved best detector (mean Dice {best_dice:.3f})")
    print(f"\nDone. best mean Dice {best_dice:.3f}. Checkpoint: {save_dir}/best_detector.pth")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="PRANet Stage 1: train lesion detector on DDR masks")
    p.add_argument('--data_root', type=str, required=True,
                   help='Lesion dataset folder in the fixed format (see README, Data)')
    p.add_argument('--image_size', type=int, default=1024)
    p.add_argument('--base', type=int, default=32, help='detector base channel width')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='./baseline/pranet')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    print(json.dumps(vars(args), indent=2))
    main(args)
