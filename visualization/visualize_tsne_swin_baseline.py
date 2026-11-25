#!/usr/bin/env python3
"""
Swin Baseline t-SNE 可视化

提取 SwinClassifier backbone 的 pooled 特征 [B, 1536]，
用 t-SNE 降至 2D 可视化不同 DR 等级的分布。

用法:
    python visualization/visualize_tsne_swin_baseline.py \
        --checkpoint ./baseline/swin_baseline/swin_idrid/best_model.pth \
        --data_root data/idrid \
        --split test
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from Models import SwinClassifier
from grading_dataloader import create_grading_dataloaders


def load_swin_baseline(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    args = ckpt['args']
    model = SwinClassifier(
        num_classes=3 if args.get('use_3_stages', False) else 5,
        model_name=args.get('swin_model', 'swin_large_patch4_window7_224'),
        pretrained=False,
        drop_rate=args.get('drop_rate', 0.3),
        image_size=args.get('image_size', 1024),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"Loaded SwinClassifier (epoch={ckpt.get('epoch','?')}, "
          f"best_kappa={ckpt.get('best_kappa', '?'):.4f})")
    return model, args


@torch.no_grad()
def extract_features(model, dataloader, device, max_samples=None):
    """提取 backbone pooled 特征 [B, 1536] 和预测结果"""
    all_feats, all_labels, all_preds = [], [], []
    total = 0
    for batch in tqdm(dataloader, desc="Extracting features"):
        images = batch['image'].to(device)
        labels = batch['label']

        feats  = model.backbone(images)           # [B, 1536]
        logits = model.classifier(feats)          # [B, num_classes]
        preds  = logits.argmax(dim=1).cpu()

        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy() if hasattr(labels, 'numpy') else np.array(labels))
        all_preds.append(preds.numpy())

        total += images.size(0)
        if max_samples and total >= max_samples:
            break

    return (np.concatenate(all_feats,   axis=0),
            np.concatenate(all_labels,  axis=0),
            np.concatenate(all_preds,   axis=0))


def plot_tsne(feat2d, labels, predictions, num_classes, class_names,
              title, save_path):
    if num_classes == 5:
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    else:
        colors = ['#2ecc71', '#f39c12', '#e74c3c']

    correct   = labels == predictions
    incorrect = ~correct

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(title, fontsize=15, fontweight='bold')

    # 左：ground truth 着色
    ax = axes[0]
    for g in range(num_classes):
        m = labels == g
        ax.scatter(feat2d[m, 0], feat2d[m, 1],
                   c=colors[g], label=class_names[g],
                   alpha=0.65, s=35, edgecolors='k', linewidths=0.3)
    ax.set_title('Ground Truth Labels', fontsize=13)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    # 右：错误高亮
    ax = axes[1]
    ax.scatter(feat2d[correct, 0], feat2d[correct, 1],
               c='lightgray', alpha=0.3, s=20, label='Correct', edgecolors='none')
    for g in range(num_classes):
        m = incorrect & (labels == g)
        if m.sum() > 0:
            ax.scatter(feat2d[m, 0], feat2d[m, 1],
                       c=colors[g], label=f'{class_names[g]} (Error)',
                       alpha=0.9, s=60, edgecolors='red', linewidths=1.5, marker='X')
    acc = correct.mean() * 100
    ax.set_title(f'Prediction Errors  (acc={acc:.1f}%)', fontsize=13)
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)

    print(f"\nAccuracy: {acc:.2f}%")
    for g in range(num_classes):
        gm = labels == g
        print(f"  {class_names[g]}: {(correct & gm).sum()}/{gm.sum()}")


def main():
    parser = argparse.ArgumentParser(description='Swin Baseline t-SNE')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='SwinClassifier best_model.pth 路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Use the same value as in training to rebuild the same splits')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./tsne_plots')
    args = parser.parse_args()
    data_name = os.path.basename(os.path.normpath(args.data_root))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print(f"\nLoading SwinClassifier from: {args.checkpoint}")
    model, ckpt_args = load_swin_baseline(args.checkpoint, device)
    num_classes  = model.num_classes
    use_3_stages = ckpt_args.get('use_3_stages', False)
    image_size   = args.image_size or ckpt_args.get('image_size', 1024)

    # 构建 dataloader
    print(f"\nDatasets: {args.data_root}, split={args.split}")
    dataloaders = create_grading_dataloaders(
        data_root=args.data_root,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3_stages=use_3_stages,
        use_oversample=False,
        use_strong_aug=False,
        split_seed=args.split_seed,
    )

    loader = dataloaders.get(args.split)
    if loader is None:
        print(f"No '{args.split}' loader available, exiting.")
        return
    print(f"  {args.split} set: {len(loader.dataset)} samples")

    # 提取特征
    features, labels, predictions = extract_features(
        model, loader, device, max_samples=args.max_samples
    )
    print(f"\nFeatures shape: {features.shape}")

    class_names = (['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
                   if num_classes == 5 else ['Healthy', 'NPDR', 'PDR'])

    print("\nClass distribution:")
    for g in range(num_classes):
        cnt = (labels == g).sum()
        print(f"  {class_names[g]}: {cnt} ({100*cnt/len(labels):.1f}%)")

    # t-SNE
    perp = min(args.perplexity, len(features) - 1)
    print(f"\nRunning t-SNE (n={len(features)}, perplexity={perp})...")
    feat2d = TSNE(n_components=2, perplexity=perp,
                  random_state=42, max_iter=1000).fit_transform(features)

    # 保存
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_name = os.path.basename(os.path.dirname(args.checkpoint))
    save_path = os.path.join(
        args.save_dir,
        f'tsne_swin_baseline_{ckpt_name}_{data_name}_{args.split}.png'
    )
    plot_tsne(
        feat2d, labels, predictions,
        num_classes=num_classes,
        class_names=class_names,
        title=f'Swin Baseline Features — {ckpt_name} [{args.split.upper()}]',
        save_path=save_path,
    )
    print("\nDone!")


if __name__ == '__main__':
    main()
