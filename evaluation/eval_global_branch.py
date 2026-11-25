#!/usr/bin/env python3
"""
单独评估 PARAGraph 的 Global node 分支（不使用 GNN/病灶分支）。

用法:
    python evaluation/eval_global_branch.py \
        --checkpoint ./checkpoints/paragraph_messidor/paragraph_xxx/best_model.pth \
        --data_root data/messidor2 --split_seed 42 \
        --image_size 1024
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models import PARAGraph, OpticUNet, LesionUNet
from grading_dataloader import build_dataset, get_val_transform, STAGE_NAMES


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


@torch.no_grad()
def evaluate_global_only(model, dataloader, device):
    model.eval()
    all_preds_global = []
    all_preds_fusion = []
    all_labels = []
    num_classes = model.num_classes

    for batch in tqdm(dataloader, desc="  Evaluating", leave=False):
        images = batch['image'].to(device)
        labels = batch['label']
        output = model(images)

        # Global branch only
        global_preds = output['global_logits'].argmax(dim=1).cpu()
        # Fusion (n:1)
        fusion_preds = output['logits'].argmax(dim=1).cpu()

        all_preds_global.extend(global_preds.numpy())
        all_preds_fusion.extend(fusion_preds.numpy())
        all_labels.extend(labels.numpy() if hasattr(labels, 'numpy') else labels)

    all_labels = np.array(all_labels)
    all_preds_global = np.array(all_preds_global)
    all_preds_fusion = np.array(all_preds_fusion)

    class_names = [f'Grade_{i}' for i in range(num_classes)]

    def report(name, preds):
        acc   = (preds == all_labels).mean()
        kappa = quadratic_weighted_kappa(all_labels, preds, num_classes)
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  Accuracy: {acc:.4f}   Kappa: {kappa:.4f}")
        print(f"  Per-class accuracy:")
        for i, cname in enumerate(class_names):
            total = (all_labels == i).sum()
            ca = float((preds[all_labels == i] == i).sum()) / total if total > 0 else 0.0
            print(f"    {cname}: {ca:.4f}  (n={total})")

    report("Global Branch Only", all_preds_global)
    report("Fusion (n:1)", all_preds_fusion)


def find_best_checkpoint(save_dir):
    """在 save_dir 下递归找 best_model.pth"""
    for root, dirs, files in os.walk(save_dir):
        if 'best_model.pth' in files:
            return os.path.join(root, 'best_model.pth')
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='PARAGraph best_model.pth 路径，或包含它的 save_dir')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Use the same value as in training to rebuild the same test split')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--optic_checkpoint', type=str,
                        default='./checkpoints/optic_unet_resnet50/best_model.pth')
    parser.add_argument('--lesion_checkpoint', type=str,
                        default='./checkpoints/lesion_unet_swint/best_model.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Find checkpoint
    ckpt_path = args.checkpoint
    if os.path.isdir(ckpt_path):
        ckpt_path = find_best_checkpoint(ckpt_path)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Cannot find checkpoint: {args.checkpoint}")
    print(f"Checkpoint: {ckpt_path}")

    # Load segmentation models
    optic_unet = OpticUNet(encoder_pretrained=True, predict_fovea=True,
                           encoder_name='resnet50', img_size=args.image_size)
    state = torch.load(args.optic_checkpoint, map_location='cpu', weights_only=False)
    optic_unet.load_state_dict(state.get('model_state_dict', state), strict=False)

    lesion_unet = LesionUNet(num_lesions=4, encoder_pretrained=False,
                             encoder_freeze=False, encoder_name='swin',
                             img_size=args.image_size)
    state = torch.load(args.lesion_checkpoint, map_location='cpu', weights_only=False)
    lesion_unet.load_state_dict(state.get('model_state_dict', state), strict=False)

    # Load PARAGraph
    model = PARAGraph(optic_unet=optic_unet, lesion_unet=lesion_unet,
                      feature_dim=256, num_classes=5).to(device)
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=True)
    print("PARAGraph loaded.")

    # Dataloader
    test_set = build_dataset(args.data_root, 'test', get_val_transform(args.image_size), args.image_size,
                             use_3_stages=False, split_seed=args.split_seed)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    print(f"Test set: {len(test_loader.dataset)} samples")

    evaluate_global_only(model, test_loader, device)


if __name__ == '__main__':
    main()
