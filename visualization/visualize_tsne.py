#!/usr/bin/env python3
"""
PARAGraph vs Swin Baseline t-SNE 对比可视化

左图: PARAGraph 图级别特征 (graph_features, 分类器之前的 pooled features)
右图: Swin Baseline backbone 特征 (SwinClassifier.backbone → [B, 1536])

两图均按真实 DR 分级着色。

用法:
    python visualization/visualize_tsne.py \
        --checkpoint ./checkpoints/paragraph/paragraph_idrid/best_model.pth \
        --swin_checkpoint ./baseline/swin_baseline/swin_idrid/best_model.pth \
        --data_root data/idrid
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import sys
import random
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm

from Models import PARAGraph, SwinClassifier, EfficientNetClassifier, CABNetClassifier
from grading_dataloader import create_grading_dataloaders

SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DGDR_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'DGDR'))
if DGDR_DIR not in sys.path:
    sys.path.insert(0, DGDR_DIR)


def load_cabnet(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ckpt.get('args', {})
    model = CABNetClassifier(
        num_classes=3 if a.get('use_3_stages', False) else 5,
        backbone_name=a.get('backbone', 'resnet50'),
        pretrained=False,
        drop_rate=a.get('drop_rate', 0.3),
        ca_k=a.get('ca_k', 5),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded CABNet (epoch={ckpt.get('epoch','?')}, "
          f"best_kappa={ckpt.get('best_kappa','?'):.4f})")
    return model


@torch.no_grad()
def extract_cabnet_features(model, dataloader, device, max_samples=None):
    """提取 CABNet ca_feat 特征（分类头最后一层前，[B, C//2]）"""
    model.eval()
    all_feats, all_labels = [], []
    total = 0
    for batch in tqdm(dataloader, desc="CABNet features"):
        images = batch['image'].to(device)
        labels = batch['label']
        feat_map = model.backbone(images)
        feat = model.classifier.cbam(feat_map)
        pooled = model.classifier.global_pool(feat).flatten(1)
        global_feat = model.classifier.global_fc(pooled).unsqueeze(-1).unsqueeze(-1)
        ca_feat = model.classifier.ca_block(global_feat)
        feats = model.classifier.final_pool(ca_feat).flatten(1)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy() if hasattr(labels, 'numpy') else np.array(labels))
        total += images.size(0)
        if max_samples and total >= max_samples:
            break
    return (np.concatenate(all_feats,  axis=0),
            np.concatenate(all_labels, axis=0))


def load_efficientnet(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ckpt.get('args', {})
    model = EfficientNetClassifier(
        num_classes=3 if a.get('use_3_stages', False) else 5,
        model_name=a.get('efficientnet_model', 'tf_efficientnet_b7'),
        pretrained=False,
        drop_rate=a.get('drop_rate', 0.3),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded EfficientNet (epoch={ckpt.get('epoch','?')}, "
          f"best_kappa={ckpt.get('best_kappa','?'):.4f})")
    return model


@torch.no_grad()
def extract_efficientnet_features(model, dataloader, device, max_samples=None):
    """提取 EfficientNet backbone 特征 [B, num_features]（timm 已做 global avgpool）"""
    model.eval()
    all_feats, all_labels = [], []
    total = 0
    for batch in tqdm(dataloader, desc="EfficientNet features"):
        images = batch['image'].to(device)
        labels = batch['label']
        feats = model.backbone(images)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy() if hasattr(labels, 'numpy') else np.array(labels))
        total += images.size(0)
        if max_samples and total >= max_samples:
            break
    return (np.concatenate(all_feats,  axis=0),
            np.concatenate(all_labels, axis=0))


def load_swin_baseline(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    a = ckpt['args']
    model = SwinClassifier(
        num_classes=3 if a.get('use_3_stages', False) else 5,
        model_name=a.get('swin_model', 'swin_large_patch4_window7_224'),
        pretrained=False,
        drop_rate=a.get('drop_rate', 0.3),
        image_size=a.get('image_size', 1024),
    )
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()
    print(f"  Loaded SwinClassifier (epoch={ckpt.get('epoch','?')}, "
          f"best_kappa={ckpt.get('best_kappa','?'):.4f})")
    return model


@torch.no_grad()
def extract_paragraph_features(model, dataloader, device, max_samples=None):
    """提取 PARAGraph 图级别特征 (graph_features)"""
    model.eval()
    all_feats, all_labels = [], []
    total = 0
    for batch in tqdm(dataloader, desc="PARAGraph features"):
        images = batch['image'].to(device)
        labels = batch['label']
        output = model(images)
        all_feats.append(output['graph_features'].cpu().numpy())
        all_labels.append(labels.numpy() if hasattr(labels, 'numpy') else np.array(labels))
        total += images.size(0)
        if max_samples and total >= max_samples:
            break
    return (np.concatenate(all_feats,  axis=0),
            np.concatenate(all_labels, axis=0))


@torch.no_grad()
def extract_swin_features(model, dataloader, device, max_samples=None):
    """提取 SwinClassifier backbone 特征 [B, 1536]"""
    model.eval()
    all_feats, all_labels = [], []
    total = 0
    for batch in tqdm(dataloader, desc="Swin Baseline features"):
        images = batch['image'].to(device)
        labels = batch['label']
        feats = model.backbone(images)
        feats = model.classifier[0](feats)  # LayerNorm → Linear 之前
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy() if hasattr(labels, 'numpy') else np.array(labels))
        total += images.size(0)
        if max_samples and total >= max_samples:
            break
    return (np.concatenate(all_feats,  axis=0),
            np.concatenate(all_labels, axis=0))


def run_tsne(features, perplexity, desc=''):
    perp = min(perplexity, len(features) - 1)
    print(f"Running t-SNE {desc} (n={len(features)}, perplexity={perp})...")
    return TSNE(n_components=2, perplexity=perp,
                random_state=42, init='pca', max_iter=1000).fit_transform(features)


def _draw_ax(ax, feat2d, labels, num_classes, class_names, title):
    if num_classes == 5:
        colors = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
    else:
        colors = ['#2ecc71', '#f39c12', '#e74c3c']

    for g in range(num_classes):
        m = labels == g
        ax.scatter(feat2d[m, 0], feat2d[m, 1],
                   c=colors[g], label=class_names[g],
                   alpha=0.65, s=55, edgecolors='k', linewidths=0.3)
    ax.set_title(title, fontsize=33, fontweight='bold')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(alpha=0.3)


def plot_single(feat2d, labels, num_classes, class_names, title, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_ax(ax, feat2d, labels, num_classes, class_names, title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def combine_images(img_paths, labels, title, save_path, num_classes=5, class_names=None):
    """将 N 张已保存的 PNG 图像横向拼接，并在底部放置共享 legend"""
    if class_names is None:
        class_names = [f'Grade {i}' for i in range(num_classes)]
    colors = (['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6']
              if num_classes == 5 else ['#2ecc71', '#f39c12', '#e74c3c'])

    n = len(img_paths)
    fig = plt.figure(figsize=(9 * n, 9))
    gs  = fig.add_gridspec(2, n, height_ratios=[10, 1], hspace=0.05)

    for col, (img_path, label) in enumerate(zip(img_paths, labels)):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(plt.imread(img_path))
        ax.set_title(label, fontsize=33, fontweight='bold')
        ax.axis('off')

    # 共享 legend（底部跨列）
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis('off')
    handles = [plt.scatter([], [], c=c, s=105, edgecolors='k',
                           linewidths=0.3, label=name)
               for c, name in zip(colors, class_names)]
    ax_leg.legend(handles=handles, loc='center', ncol=num_classes,
                  fontsize=29, framealpha=0.9, edgecolor='#cccccc',
                  markerscale=2.0, columnspacing=2.0)

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='PARAGraph vs Swin Baseline t-SNE')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='PARAGraph best_model.pth 路径 (不提供则跳过 PARAGraph)')
    parser.add_argument('--swin_checkpoint', type=str, default=None,
                        help='SwinClassifier best_model.pth 路径 (不提供则跳过 Swin-Large)')
    parser.add_argument('--efficientnet_checkpoint', type=str, default=None,
                        help='EfficientNetClassifier best_model.pth 路径')
    parser.add_argument('--cabnet_checkpoint', type=str, default=None,
                        help='CABNetClassifier best_model.pth 路径')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Use the same value as in training to rebuild the same splits')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--use_3_stages', action='store_true', default=False)
    parser.add_argument('--perplexity', type=int, default=30)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./tsne_plots')
    args = parser.parse_args()
    data_name = os.path.basename(os.path.normpath(args.data_root))

    if not any([args.checkpoint, args.swin_checkpoint,
                args.efficientnet_checkpoint, args.cabnet_checkpoint]):
        print("Error: 至少提供 --checkpoint / --swin_checkpoint / --efficientnet_checkpoint / --cabnet_checkpoint 之一")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    def make_loader(use_3_stages):
        """每次调用都创建一个全新独立的 DataLoader"""
        dls = create_grading_dataloaders(
            data_root=args.data_root,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            use_3_stages=use_3_stages,
            use_oversample=False,
            use_strong_aug=False,
            split_seed=args.split_seed,
        )
        return dls.get(args.split)

    num_classes         = None
    pg_save_path        = None
    swin_save_path      = None
    eff_save_path       = None
    cabnet_save_path    = None

    # ── PARAGraph（完全独立）──────────────────────────────────────────
    if args.checkpoint:
        print(f"\nLoading PARAGraph from: {args.checkpoint}")
        pg_model = PARAGraph.load_from_checkpoint(
            args.checkpoint, device=str(device), eval_mode=True
        )
        pg_model = pg_model.to(device)
        pg_model.eval()
        num_classes = pg_model.num_classes

        loader = make_loader(pg_model.num_classes == 3)
        if loader is None:
            print(f"No '{args.split}' loader for PARAGraph, skipping.")
        else:
            print(f"  {args.split} set: {len(loader.dataset)} samples")
            graph_feats, labels = extract_paragraph_features(
                pg_model, loader, device, max_samples=args.max_samples
            )
            print(f"  PARAGraph features: {graph_feats.shape}")

            class_names = (['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
                           if num_classes == 5 else ['Healthy', 'NPDR', 'PDR'])
            graph_2d = run_tsne(graph_feats, args.perplexity, desc='[PARAGraph]')

            pg_name = (os.path.basename(os.path.dirname(args.checkpoint)) or
                       os.path.basename(args.checkpoint).replace('.pth', ''))
            pg_save_path = os.path.join(
                args.save_dir, f'tsne_paragraph_{pg_name}_{data_name}.png'
            )
            plot_single(
                graph_2d, labels,
                num_classes=num_classes,
                class_names=class_names,
                title='',
                save_path=pg_save_path,
            )
        del pg_model

    # ── Swin-Large（完全独立）────────────────────────────────────────
    if args.swin_checkpoint:
        print(f"\nLoading Swin-Large from: {args.swin_checkpoint}")
        sw_model = load_swin_baseline(args.swin_checkpoint, device)
        sw_num_classes = sw_model.num_classes
        if num_classes is None:
            num_classes = sw_num_classes

        loader = make_loader(sw_num_classes == 3)
        if loader is None:
            print(f"No '{args.split}' loader for Swin-Large, skipping.")
        else:
            print(f"  {args.split} set: {len(loader.dataset)} samples")
            swin_feats, labels = extract_swin_features(
                sw_model, loader, device, max_samples=args.max_samples
            )
            print(f"  Swin-Large features: {swin_feats.shape}")

            class_names = (['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
                           if sw_num_classes == 5 else ['Healthy', 'NPDR', 'PDR'])
            swin_2d = run_tsne(swin_feats, args.perplexity, desc='[Swin-Large]')

            sw_name = (os.path.basename(os.path.dirname(args.swin_checkpoint)) or
                       os.path.basename(args.swin_checkpoint).replace('.pth', ''))
            swin_save_path = os.path.join(
                args.save_dir, f'tsne_swin_{sw_name}_{data_name}.png'
            )
            plot_single(
                swin_2d, labels,
                num_classes=sw_num_classes,
                class_names=class_names,
                title='',
                save_path=swin_save_path,
            )
        del sw_model

    # ── EfficientNet ──────────────────────────────────────────────────
    if args.efficientnet_checkpoint:
        print(f"\nLoading EfficientNet from: {args.efficientnet_checkpoint}")
        eff_model = load_efficientnet(args.efficientnet_checkpoint, device)
        eff_num_classes = eff_model.num_classes
        if num_classes is None:
            num_classes = eff_num_classes

        loader = make_loader(eff_num_classes == 3)
        if loader is None:
            print(f"No '{args.split}' loader for EfficientNet, skipping.")
        else:
            print(f"  {args.split} set: {len(loader.dataset)} samples")
            eff_feats, labels = extract_efficientnet_features(
                eff_model, loader, device, max_samples=args.max_samples
            )
            print(f"  EfficientNet features: {eff_feats.shape}")

            class_names = (['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
                           if eff_num_classes == 5 else ['Healthy', 'NPDR', 'PDR'])
            eff_2d = run_tsne(eff_feats, args.perplexity, desc='[EfficientNet]')

            eff_name = (os.path.basename(os.path.dirname(args.efficientnet_checkpoint)) or
                        os.path.basename(args.efficientnet_checkpoint).replace('.pth', ''))
            eff_save_path = os.path.join(
                args.save_dir, f'tsne_efficientnet_{eff_name}_{data_name}.png'
            )
            plot_single(
                eff_2d, labels,
                num_classes=eff_num_classes,
                class_names=class_names,
                title='',
                save_path=eff_save_path,
            )
        del eff_model

    # ── CABNet ────────────────────────────────────────────────────────
    if args.cabnet_checkpoint:
        print(f"\nLoading CABNet from: {args.cabnet_checkpoint}")
        cab_model = load_cabnet(args.cabnet_checkpoint, device)
        cab_num_classes = cab_model.num_classes
        if num_classes is None:
            num_classes = cab_num_classes

        loader = make_loader(cab_num_classes == 3)
        if loader is None:
            print(f"No '{args.split}' loader for CABNet, skipping.")
        else:
            print(f"  {args.split} set: {len(loader.dataset)} samples")
            cab_feats, labels = extract_cabnet_features(
                cab_model, loader, device, max_samples=args.max_samples
            )
            print(f"  CABNet features: {cab_feats.shape}")

            class_names = (['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
                           if cab_num_classes == 5 else ['Healthy', 'NPDR', 'PDR'])
            cab_2d = run_tsne(cab_feats, args.perplexity, desc='[CABNet]')

            cab_name = (os.path.basename(os.path.dirname(args.cabnet_checkpoint)) or
                        os.path.basename(args.cabnet_checkpoint).replace('.pth', ''))
            cabnet_save_path = os.path.join(
                args.save_dir, f'tsne_cabnet_{cab_name}_{data_name}.png'
            )
            plot_single(
                cab_2d, labels,
                num_classes=cab_num_classes,
                class_names=class_names,
                title='',
                save_path=cabnet_save_path,
            )
        del cab_model

    # ── 拼接（收集所有已生成的单图）─────────────────────────────────
    panel_paths  = [p for p in [pg_save_path, swin_save_path,
                                eff_save_path, cabnet_save_path] if p]
    panel_labels = (
        (['PARAGraph']    if pg_save_path     else []) +
        (['Swin']         if swin_save_path   else []) +
        (['EfficientNet'] if eff_save_path    else []) +
        (['CABNet']       if cabnet_save_path else [])
    )
    if len(panel_paths) >= 2:
        combined_path = os.path.join(
            args.save_dir, f'tsne_comparison_{data_name}.png'
        )
        _nc = num_classes if num_classes is not None else 5
        _cn = (['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
               if _nc == 5 else ['Healthy', 'NPDR', 'PDR'])
        combine_images(
            panel_paths, panel_labels,
            title=f'{" vs ".join(panel_labels)} — {data_name.upper()}',
            save_path=combined_path,
            num_classes=_nc,
            class_names=_cn,
        )

    print("\nDone!")


if __name__ == '__main__':
    main()
