#!/usr/bin/env python3
"""
比较多个 PARAGraph 模型的 t-SNE 可视化

用于对比不同消融实验、不同训练阶段的模型学到的特征表示
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from Models import PARAGraph
from grading_dataloader import create_grading_dataloaders


def extract_features(model, dataloader, device, max_samples=None):
    """提取特征"""
    model.eval()
    all_features = []
    all_labels = []
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features", leave=False):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            output = model(images)
            graph_features = output['graph_features']

            all_features.append(graph_features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

            total_samples += images.size(0)
            if max_samples and total_samples >= max_samples:
                break

    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    return features, labels


def plot_comparison(features_list, labels_list, model_names, save_path=None,
                   num_classes=5, perplexity=30):
    """
    绘制多个模型的 t-SNE 对比图

    Args:
        features_list: List of [N, D] feature matrices
        labels_list: List of [N] label arrays
        model_names: List of model names
        save_path: Save path
        num_classes: Number of classes
        perplexity: t-SNE perplexity
    """
    n_models = len(features_list)

    # 颜色和类别名称
    if num_classes == 5:
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
        class_names = ['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4 (PDR)']
    else:
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        class_names = ['Healthy', 'NPDR', 'PDR']

    # 创建子图
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 8))
    if n_models == 1:
        axes = [axes]

    for idx, (features, labels, model_name, ax) in enumerate(zip(features_list, labels_list, model_names, axes)):
        print(f"\nRunning t-SNE for {model_name} (perplexity={perplexity})...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                    init='pca', n_iter=1000)
        features_2d = tsne.fit_transform(features)

        # 绘制散点图
        for grade in range(num_classes):
            mask = labels == grade
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                      c=colors[grade], label=class_names[grade],
                      alpha=0.6, s=30, edgecolors='k', linewidths=0.5)

        ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
        ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
        ax.set_title(model_name, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(alpha=0.3)

    plt.suptitle('PARAGraph Feature Comparison (t-SNE)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved comparison plot to: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Compare t-SNE for Multiple PARAGraph Models')

    # Models (支持多个 checkpoint)
    parser.add_argument('--checkpoints', type=str, nargs='+', required=True,
                       help='List of checkpoint paths')
    parser.add_argument('--names', type=str, nargs='+', default=None,
                       help='Model names (default: use checkpoint basenames)')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Use the same value as in training to rebuild the same splits')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--use_3_stages', action='store_true', default=False)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])

    # t-SNE
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--max_samples', type=int, default=None)

    # Output
    parser.add_argument('--save_dir', type=str, default='./tsne_plots')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 默认模型名称
    if args.names is None:
        args.names = [os.path.basename(ckpt).replace('.pth', '') for ckpt in args.checkpoints]

    assert len(args.checkpoints) == len(args.names), "Number of checkpoints and names must match"

    # 创建 dataloader (所有模型共享)
    num_classes = 3 if args.use_3_stages else 5
    print(f"\nCreating dataloaders...")
    dataloaders = create_grading_dataloaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3_stages=args.use_3_stages,
        image_size=args.image_size,
        use_oversample=False,
        use_strong_aug=False,
        split_seed=args.split_seed,
    )

    dataloader = dataloaders[args.split]
    print(f"  {args.split.capitalize()} set: {len(dataloader.dataset)} samples")

    # 提取所有模型的特征
    features_list = []
    labels_list = []

    for checkpoint, name in zip(args.checkpoints, args.names):
        print(f"\n{'='*80}")
        print(f"Loading: {name}")
        print(f"Path: {checkpoint}")

        model = PARAGraph.from_checkpoint(checkpoint, device=device)
        model.eval()

        features, labels = extract_features(model, dataloader, device, max_samples=args.max_samples)
        features_list.append(features)
        labels_list.append(labels)

        print(f"  Features: {features.shape}")

        # 验证所有模型的标签一致
        if len(labels_list) > 1:
            assert np.array_equal(labels, labels_list[0]), "Labels must be identical across models"

    # 绘制对比图
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, f'tsne_comparison_{args.split}.png')

    plot_comparison(
        features_list, labels_list, args.names,
        save_path=save_path,
        num_classes=num_classes,
        perplexity=args.perplexity
    )

    print("\nDone!")


if __name__ == '__main__':
    main()
