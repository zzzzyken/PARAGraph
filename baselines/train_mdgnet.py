#!/usr/bin/env python3
"""
MDGNet baseline training (single stage, image-only).

Reimplementation baseline for the PARAGraph comparison (see Models/mdgnet.py for
the architecture and the INFERRED-detail caveats). Reuses the shared training
loop (train_epoch / evaluate / QWK / confusion matrix) from train_efficientnet
and the grading_dataloader split protocol, so MDGNet is evaluated under exactly
the same protocol as every other baseline (5 seeds, same splits, QWK/ACC + the
confusion matrix from which F1/SPE are derived).
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import json
import argparse
import datetime
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from Models import MDGNet
from grading_dataloader import (
    create_grading_dataloaders,
    print_dataset_statistics,
    STAGE_NAMES,
)
from train_efficientnet import (
    train_epoch, evaluate, quadratic_weighted_kappa,
    build_confusion_matrix, print_confusion_matrix, set_seed, compute_class_weights,
)


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'mdgnet_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    num_classes = 3 if args.use_3_stages else 5

    dataloaders = create_grading_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3_stages=args.use_3_stages,
        use_oversample=args.use_oversample,
        use_strong_aug=args.use_strong_aug,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print_dataset_statistics(dataloaders, use_3_stages=args.use_3_stages)

    model = MDGNet(
        num_classes=num_classes, graph_hw=args.graph_hw,
        knn_k=args.knn_k, drop_rate=args.drop_rate,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"MDGNet params: {total_params:,}  (graph {args.graph_hw}x{args.graph_hw}, k={args.knn_k})")

    # trained from scratch -> single LR group
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(dataloaders['train'].dataset, num_classes, device)

    best_kappa, best_epoch = -1.0, 0
    history = {'train': [], 'val': []}
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr = train_epoch(model, dataloaders['train'], optimizer, device, epoch,
                         class_weights=class_weights, use_focal=args.use_focal,
                         focal_gamma=args.focal_gamma)
        va = evaluate(model, dataloaders['test'], device, class_weights=class_weights,
                      use_focal=args.use_focal, focal_gamma=args.focal_gamma)
        scheduler.step()
        history['train'].append(tr); history['val'].append(va)
        print(f"Train K={tr['kappa']:.4f} Acc={tr['accuracy']:.4f} | "
              f"Val K={va['kappa']:.4f} Acc={va['accuracy']:.4f}")
        if va['kappa'] > best_kappa:
            best_kappa, best_epoch = va['kappa'], epoch
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'best_kappa': best_kappa, 'val_accuracy': va['accuracy'],
                        'args': vars(args)}, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> saved best (K={best_kappa:.4f})")

    # final test with best ckpt
    ckpt = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test = evaluate(model, dataloaders['test'], device, class_weights=class_weights,
                    use_focal=args.use_focal, focal_gamma=args.focal_gamma)
    cm = build_confusion_matrix(test['labels'], test['predictions'], num_classes)
    class_names = STAGE_NAMES if num_classes == 3 else [f'Grade_{i}' for i in range(num_classes)]
    print(f"\nTEST  Kappa={test['kappa']:.4f}  Acc={test['accuracy']:.4f}")
    print_confusion_matrix(cm, class_names)
    with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
        json.dump({'kappa': test['kappa'], 'accuracy': test['accuracy'],
                   'class_accuracy': test['class_accuracy'],
                   'confusion_matrix': cm.tolist(), 'best_epoch': best_epoch}, f, indent=4)
    print(f"Best val Kappa {best_kappa:.4f} @epoch {best_epoch}. Saved to {save_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Train MDGNet baseline for DR grading')
    p.add_argument('--data_root', type=str, required=True,
                   help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    p.add_argument('--split_seed', type=int, default=None,
                   help='Seed of the stratified re-split (default: use the CSV files as they are)')
    p.add_argument('--val_ratio', type=float, default=0.2,
                   help='Validation share when --split_seed re-splits train.csv')
    p.add_argument('--test_ratio', type=float, default=0.2,
                   help='Share held out as the test set when the folder has no test.csv')
    p.add_argument('--image_size', type=int, default=1024,
                   help='input resolution (scaled to our other baselines; graph stays graph_hw)')
    p.add_argument('--graph_hw', type=int, default=14, help='ViG graph grid side (paper: 14)')
    p.add_argument('--knn_k', type=int, default=9)
    p.add_argument('--use_3_stages', action='store_true', default=False)
    p.add_argument('--drop_rate', type=float, default=0.3)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--use_class_weights', action='store_true', default=False)
    p.add_argument('--use_focal', action='store_true', default=True)
    p.add_argument('--no_focal', dest='use_focal', action='store_false')
    p.add_argument('--focal_gamma', type=float, default=2.0)
    p.add_argument('--use_oversample', action='store_true', default=True)
    p.add_argument('--no_oversample', dest='use_oversample', action='store_false')
    p.add_argument('--use_strong_aug', action='store_true', default=True)
    p.add_argument('--no_strong_aug', dest='use_strong_aug', action='store_false')
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--save_dir', type=str, default='./baseline/mdgnet')
    p.add_argument('--seed', type=int, default=1234)
    args = p.parse_args()
    print(json.dumps(vars(args), indent=2))
    main(args)
