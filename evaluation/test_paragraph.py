#!/usr/bin/env python3
"""
PARAGraph 独立测试脚本

在固定格式的数据文件夹（见 README 的 Data 一节）上评估一个或多个 checkpoint，
输出 Accuracy、QWK Kappa、Per-class Accuracy、混淆矩阵。

用法:
    # 单个 checkpoint
    python evaluation/test_paragraph.py \
        --checkpoints ./checkpoints/paragraph_idrid/best_model.pth \
        --data_root data/idrid

    # 多个 checkpoint 对比
    python evaluation/test_paragraph.py \
        --checkpoints ckpt_A.pth ckpt_B.pth \
        --names "Full Model" "No Gating" \
        --data_root data/idrid

    # 指定数据集和 split
    python evaluation/test_paragraph.py \
        --checkpoints best_model.pth \
        --data_root data/idrid \
        --split test          # test / val / both
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import sys
import json
import datetime
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models import PARAGraph
from grading_dataloader import (
    build_dataset,
    get_val_transform,
    STAGE_NAMES,
)


# ─── Metrics ────────────────────────────────────────────────────────────────
def quadratic_weighted_kappa(y_true, y_pred, num_classes):
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1
    weights = np.array([[(i - j) ** 2 / (num_classes - 1) ** 2
                          for j in range(num_classes)]
                         for i in range(num_classes)], dtype=np.float64)
    hist_true = confusion.sum(axis=1)
    hist_pred = confusion.sum(axis=0)
    expected  = np.outer(hist_true, hist_pred) / confusion.sum()
    denom = (weights * expected).sum()
    return 0.0 if denom == 0 else 1.0 - (weights * confusion).sum() / denom


def build_confusion_matrix(labels, preds, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(labels, preds):
        cm[int(t), int(p)] += 1
    return cm


def print_confusion_matrix(cm, class_names):
    col_w = max(len(n) for n in class_names) + 2
    label  = 'True\\Pred'
    header = f"{label:<{col_w}}" + "".join(f"{n:>{col_w}}" for n in class_names)
    print(header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:<{col_w}}" + "".join(f"{cm[i,j]:>{col_w}}" for j in range(len(class_names)))
        print(row)


# ─── Evaluation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    num_classes = model.num_classes

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        images = batch['image'].to(device)
        labels = batch['label']
        output = model(images)
        preds  = output['logits'].argmax(dim=1).cpu()
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean()
    kappa    = quadratic_weighted_kappa(all_labels, all_preds, num_classes)
    cm       = build_confusion_matrix(all_labels, all_preds, num_classes)

    class_names = STAGE_NAMES if num_classes == 3 else [f'Grade_{i}' for i in range(num_classes)]
    class_acc = {}
    for i, name in enumerate(class_names):
        total = (all_labels == i).sum()
        class_acc[name] = float((all_preds[all_labels == i] == i).sum()) / total if total > 0 else 0.0

    return {
        'accuracy':    accuracy,
        'kappa':       kappa,
        'class_acc':   class_acc,
        'cm':          cm,
        'preds':       all_preds,
        'labels':      all_labels,
        'num_classes': num_classes,
    }


# ─── Dataloader helpers ───────────────────────────────────────────────────────
def make_loader(data_root, split, image_size, batch_size, num_workers,
                use_3_stages=False, split_seed=None, val_ratio=0.2, test_ratio=0.2):
    """Loader for one split ('val' or 'test') of a dataset folder in the fixed format."""
    try:
        ds = build_dataset(data_root, split, get_val_transform(image_size), image_size,
                           use_3_stages, split_seed, val_ratio, test_ratio)
    except ValueError as e:
        print(f"  {e}")
        return None
    counts = {}
    for lbl in ds.get_labels():
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"  [{ds.name} {split}] {len(ds)} samples | "
          f"dist: { {k: counts[k] for k in sorted(counts)} }")
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


# ─── Print results ────────────────────────────────────────────────────────────
def print_results(name, split, metrics):
    class_names = (STAGE_NAMES if metrics['num_classes'] == 3
                   else [f'Grade_{i}' for i in range(metrics['num_classes'])])
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {name}  [{split.upper()}]")
    print(bar)
    print(f"  Accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"  QWK Kappa: {metrics['kappa']:.4f}")
    print("\n  Per-class Accuracy:")
    for cls, acc in metrics['class_acc'].items():
        print(f"    {cls:<12}: {acc*100:.2f}%")
    print("\n  Confusion Matrix:")
    print_confusion_matrix(metrics['cm'], class_names)


# ─── Main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Test PARAGraph checkpoints on a dataset folder')
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='One or more checkpoint .pth paths')
    parser.add_argument('--names', nargs='+', default=None,
                        help='Display names for each checkpoint (optional)')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Use the same value as in training to rebuild the same splits')
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--split', type=str, default='test',
                        choices=['test', 'val', 'both'],
                        help='Which split to evaluate (default: test)')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--strict', action='store_true', default=False,
                        help='Strict state_dict loading')
    parser.add_argument('--save_dir', type=str, default='./test_results',
                        help='Directory to save results JSON (default: ./test_results)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Display names
    names = args.names if args.names else [os.path.basename(os.path.dirname(c)) or c
                                           for c in args.checkpoints]
    if len(names) < len(args.checkpoints):
        names += [os.path.basename(c) for c in args.checkpoints[len(names):]]

    splits = ['val', 'test'] if args.split == 'both' else [args.split]

    os.makedirs(args.save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Summary table storage
    summary = []  # list of (name, split, accuracy, kappa)
    all_results = []  # for JSON export

    for ckpt_path, model_name in zip(args.checkpoints, names):
        print(f"\n{'─'*60}")
        print(f"Loading: {model_name}")
        print(f"  from : {ckpt_path}")

        model = PARAGraph.load_from_checkpoint(
            ckpt_path,
            device=str(device),
            eval_mode=True,
            strict=args.strict,
            verbose=True,
        )
        model = model.to(device)
        num_classes = model.num_classes

        for split in splits:
            # Build dataloader
            loader = make_loader(args.data_root, split, args.image_size,
                                 args.batch_size, args.num_workers,
                                 use_3_stages=(num_classes == 3),
                                 split_seed=args.split_seed,
                                 val_ratio=args.val_ratio, test_ratio=args.test_ratio)
            if loader is None:
                print(f"  No {split} split in {args.data_root}, skipping")
                continue

            metrics = evaluate(model, loader, device)
            print_results(model_name, split, metrics)
            summary.append((model_name, split, metrics['accuracy'], metrics['kappa']))

            class_names = (STAGE_NAMES if metrics['num_classes'] == 3
                           else [f'Grade_{i}' for i in range(metrics['num_classes'])])
            all_results.append({
                'name':       model_name,
                'checkpoint': ckpt_path,
                'dataset':    args.data_root,
                'split':      split,
                'accuracy':   round(float(metrics['accuracy']), 6),
                'kappa':      round(float(metrics['kappa']), 6),
                'class_acc':  {k: round(v, 6) for k, v in metrics['class_acc'].items()},
                'confusion_matrix': metrics['cm'].tolist(),
                'class_names': class_names,
            })

        del model
        torch.cuda.empty_cache()

    # ── Save JSON ──
    json_path = os.path.join(args.save_dir, f'results_{timestamp}.json')
    with open(json_path, 'w') as f:
        json.dump({'timestamp': timestamp, 'results': all_results}, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # ── Summary table ──
    if len(summary) > 1:
        print(f"\n{'='*60}")
        print("  SUMMARY")
        print(f"{'='*60}")
        print(f"  {'Name':<30} {'Split':<6} {'Acc':>8} {'Kappa':>8}")
        print(f"  {'-'*54}")
        for name, split, acc, kappa in summary:
            print(f"  {name:<30} {split:<6} {acc*100:>7.2f}% {kappa:>8.4f}")
        print(f"{'='*60}")


if __name__ == '__main__':
    main()
