#!/usr/bin/env python3
"""
PARAGraph 节点 IG 归因 — 批量数据集可视化

对数据集测试集中每张图依次：
  1. 完整 forward 建图
  2. Integrated Gradients (baseline=0) 计算节点归因强度
  3. 保存每张图的可视化

最后输出一张 summary 图：各节点类型的平均归因（按预测类别分组）。

用法:
    python visualization/visualize_ig_dataset.py \
        --checkpoint ./checkpoints/paragraph/paragraph_idrid/best_model.pth \
        --data_root data/idrid \
        --split test \
        [--steps 20] \
        [--save_dir ./ig_plots]
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from tqdm import tqdm, trange
import csv

from Models import PARAGraph
from grading_dataloader import create_grading_dataloaders


from PIL import Image as PILImage

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

NODE_NAMES = {
    0: 'OD', 1: 'Fovea',
    2: 'MA', 3: 'HE', 4: 'EX', 5: 'SE',
    6: 'MA_Token', 7: 'HE_Token', 8: 'EX_Token', 9: 'SE_Token',
    10: 'Region',
}
NODE_MARKERS = {
    0: 'o', 1: 'D',
    2: '^', 3: 's', 4: '*', 5: 'P',
    6: 'h', 7: 'h', 8: 'h', 9: 'h',
    10: 'p',
}
NODE_EDGE_COLORS = {
    0: 'white',   1: 'white',
    2: '#ffe082', 3: '#ff8a65', 4: '#e53935', 5: '#ab47bc',
    6: '#ffd54f', 7: '#ff7043', 8: '#c62828', 9: '#7b1fa2',
    10: 'cyan',
}

# 4 语义组：用于 IG 归一化分数
GROUP_NAMES = ['Anatomical', 'Image-Level', 'Categorical', 'Region']
GROUP_MAP = {
    'Anatomical':  [0],            # OD node
    'Image-Level': [1],            # Landmark / Fovea node
    'Categorical': [6, 7, 8, 9],   # MA/HE/EX/SE token nodes
    'Region':      [10],           # Region nodes
}
GROUP_COLORS = {
    'Anatomical':  '#3498db',
    'Image-Level': '#2ecc71',
    'Categorical': '#e67e22',
    'Region':      '#9b59b6',
}

# Category 内部细分（用于并列子图）
CAT_NAMES = ['MA', 'HE', 'EX', 'SE']
CAT_MAP = {
    'MA': [6],
    'HE': [7],
    'EX': [8],
    'SE': [9],
}
CAT_COLORS = {
    'MA': '#ffd54f',
    'HE': '#ff7043',
    'EX': '#c62828',
    'SE': '#7b1fa2',
}


# ── 工具函数 ───────────────────────────────────────────────────────────────────
def load_original_image(image_path):
    """
    加载原始图像，裁剪掉黑色边框。

    Returns:
        cropped:      [H_crop, W_crop, 3] uint8  裁剪后的图（用于显示）
        orig_hw:      (H_orig, W_orig)            裁剪前的尺寸（用于坐标 scale）
        crop_offset:  (x0, y0)                   crop 框左上角（用于坐标偏移）
    """
    img = np.array(PILImage.open(image_path).convert('RGB'))
    H_orig, W_orig = img.shape[:2]

    # 找非黑区域的 bounding box
    gray = img.mean(axis=2)
    mask = gray > 10
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img, (H_orig, W_orig), (0, 0)

    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    cropped = img[y0:y1 + 1, x0:x1 + 1]
    return cropped, (H_orig, W_orig), (x0, y0)


def get_key_indices(model, graph):
    num_lesions = graph.get('num_lesions', 0)
    num_tokens  = graph.get('num_tokens', 0)
    num_regions = graph.get('num_regions', 0)
    token_start  = 2 + num_lesions
    region_start = 2 + num_lesions + num_tokens
    if model.use_optic_nodes:
        ki = [0, 1]
        if num_tokens  > 0:
            ki += list(range(token_start,  token_start  + num_tokens))
        if num_regions > 0:
            ki += list(range(region_start, region_start + num_regions))
    else:
        ki = (list(range(region_start, region_start + num_regions))
              if num_regions > 0 else list(range(2 + num_lesions + num_tokens)))
    return ki


# ── GNN 可微前向 ───────────────────────────────────────────────────────────────
def gnn_forward(model, h, edge_index, edge_weights, edge_type_bias, key_indices):
    h = F.dropout(h, p=model.gnn_dropout, training=False)
    for gnn_layer, norm in zip(model.gnn_layers, model.gnn_norms):
        if edge_index.shape[1] > 0:
            h_new = gnn_layer(h, edge_index, edge_weights, edge_type_bias)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=model.gnn_dropout, training=False)
            h = h + h_new
    key_feat = h[key_indices].unsqueeze(0)
    queries  = model.graph_pool_queries.unsqueeze(0)
    pooled, _ = model.graph_pool_attn(queries, key_feat, key_feat)
    return model.classifier(pooled.squeeze(0).flatten())


# ── IG 计算 ────────────────────────────────────────────────────────────────────
def compute_ig(model, node_features, edge_index, edge_weights, edge_type_bias,
               key_indices, target_class, steps):
    nf = node_features.detach()
    ei = edge_index.detach()
    ew = edge_weights.detach()
    eb = edge_type_bias.detach()
    grad_sum = torch.zeros_like(nf)
    for step in range(steps):
        alpha   = (step + 1) / steps
        h_alpha = (alpha * nf).detach().clone().requires_grad_(True)
        logits  = gnn_forward(model, h_alpha, ei, ew, eb, key_indices)
        logits[target_class].backward()
        grad_sum += h_alpha.grad.detach()
    ig_attr   = (grad_sum / steps) * nf
    node_attr = ig_attr.norm(dim=-1).cpu().numpy()
    return node_attr


# ── 节点信息收集 ───────────────────────────────────────────────────────────────
def collect_nodes(graph):
    nodes = []
    for i, nd in enumerate(graph['nodes']):
        nodes.append({'idx': i, 'type': nd['type'], 'abs': nd['absolute_coords']})
    for ln in graph.get('lesion_nodes', []):
        nodes.append({'idx': ln['node_idx'], 'type': ln['type'], 'abs': ln['absolute_coords']})
    for tn in graph.get('token_nodes', []):
        nodes.append({'idx': tn['node_idx'], 'type': tn['type'], 'abs': tn['absolute_coords']})
    for rn in graph.get('region_nodes', []):
        nodes.append({'idx': rn['node_idx'], 'type': rn['type'], 'abs': rn['absolute_coords']})
    return nodes


# ── 单图可视化 ─────────────────────────────────────────────────────────────────
def plot_ig(image_np, graph, node_attr, pred_class, gt_class,
            class_names, save_path, image_size,
            orig_hw=None, crop_offset=(0, 0)):
    """
    image_np:     [H_crop, W_crop, 3] uint8  裁剪后的原图（用于显示）
    orig_hw:      (H_orig, W_orig)            裁剪前原图尺寸（用于坐标 scale）
    crop_offset:  (x0, y0)                   crop 框左上角偏移（用于坐标修正）

    节点坐标变换: image_size → 原图 → 减去 crop 偏移 → 裁剪图坐标
    """
    all_nodes = collect_nodes(graph)
    N = len(node_attr)
    valid = [n for n in all_nodes if n['idx'] < N]

    if orig_hw is not None:
        H_orig, W_orig = orig_hw
        scale_x = W_orig / image_size
        scale_y = H_orig / image_size
    else:
        scale_x = scale_y = 1.0
    off_x, off_y = crop_offset

    # 颜色: 全局归一化（保持跨节点类型的可比性）
    attr_max       = node_attr.max() + 1e-8
    attr_norm_color = node_attr / attr_max   # [0, 1], 用于颜色

    # 大小: 组内归一化（保证每类节点都可见）
    # 节点类型分组
    GROUP = {0: 'struct', 1: 'struct',
             2: 'lesion', 3: 'lesion', 4: 'lesion', 5: 'lesion',
             6: 'token',  7: 'token',  8: 'token',  9: 'token',
             10: 'region'}
    SIZE_RANGE = {'struct': (250, 600), 'token': (150, 450),
                  'region': (120, 380), 'lesion': (60, 280)}
    # 各组最大归因
    group_max = {}
    for nd in valid:
        g = GROUP.get(nd['type'], 'lesion')
        v = float(node_attr[nd['idx']])
        group_max[g] = max(group_max.get(g, 0.0), v)

    cmap     = plt.cm.plasma
    norm_map = Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(image_np)
    ax.axis('off')

    pred_name = class_names[pred_class]
    gt_name   = class_names[gt_class]
    tick = '✓' if pred_class == gt_class else '✗'
    ax.set_title(
        f'Node IG — Pred: {pred_name}  GT: {gt_name}  {tick}',
        fontsize=35, fontweight='bold', pad=6
    )

    plotted_types = set()
    for node in sorted(valid, key=lambda n: float(node_attr[n['idx']])):
        idx   = node['idx']
        ntype = node['type']
        cx, cy = node['abs']
        cx = cx * scale_x - off_x
        cy = cy * scale_y - off_y

        a_color = float(attr_norm_color[idx])              # 全局归一化 → 颜色
        g       = GROUP.get(ntype, 'lesion')
        a_size  = float(node_attr[idx]) / (group_max.get(g, 1e-8) + 1e-8)  # 组内归一化 → 大小
        s_min, s_max = SIZE_RANGE[g]
        size = s_min + a_size * (s_max - s_min)

        ax.scatter([cx], [cy],
                   s=size,
                   c=[cmap(norm_map(a_color))],
                   marker=NODE_MARKERS.get(ntype, 'o'),
                   edgecolors=NODE_EDGE_COLORS.get(ntype, 'white'),
                   linewidths=1.2, alpha=0.9, zorder=5)
        plotted_types.add(ntype)

    sm = ScalarMappable(cmap=cmap, norm=norm_map)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.027, pad=0.01)
    cbar.set_label('IG (normalized)', fontsize=35, fontweight='bold')
    cbar.ax.tick_params(labelsize=35)
    plt.draw()
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(35)
        label.set_fontweight('bold')

    handles = [
        mlines.Line2D([], [], marker=NODE_MARKERS.get(t, 'o'),
                      linestyle='None', markersize=7,
                      label=NODE_NAMES.get(t, f'T{t}'),
                      markeredgecolor=NODE_EDGE_COLORS.get(t, 'gray'),
                      color='gray')
        for t in sorted(plotted_types)
    ]
    ax.legend(handles=handles, loc='lower right', framealpha=0.85, ncol=2,
              prop={'weight': 'bold', 'size': 35})

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ── Summary 图 ─────────────────────────────────────────────────────────────────
def plot_summary(records, num_classes, class_names, save_path):
    """
    records: list of dict with keys:
        pred_class, gt_class, correct, group_scores
        (group_scores: {group_name: normalized_score})
    """
    correct_recs = [r for r in records if r['correct']]
    wrong_recs   = [r for r in records if not r['correct']]

    def mean_scores(recs):
        if not recs:
            return {g: 0.0 for g in GROUP_NAMES}
        return {g: np.mean([r['group_scores'].get(g, 0) for r in recs])
                for g in GROUP_NAMES}

    overall_mean = mean_scores(records)
    correct_mean = mean_scores(correct_recs)
    wrong_mean   = mean_scores(wrong_recs)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    x = np.arange(len(GROUP_NAMES))
    bar_colors = [GROUP_COLORS[g] for g in GROUP_NAMES]

    for ax, data, title in zip(
        axes,
        [overall_mean, correct_mean, wrong_mean],
        [f'All ({len(records)})',
         f'Correct ({len(correct_recs)})',
         f'Wrong ({len(wrong_recs)})']
    ):
        vals = [data[g] for g in GROUP_NAMES]
        bars = ax.bar(x, vals, color=bar_colors, edgecolor='k', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(GROUP_NAMES, rotation=20, ha='right', fontsize=14)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_ylabel('Mean Normalized IG Score', fontsize=14)
        ax.tick_params(axis='y', labelsize=14)
        ax.grid(axis='y', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=13)

    fig.suptitle('Node Group IG Attribution Summary',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary saved: {save_path}")


def plot_per_class_summary(records, num_classes, class_names, save_path,
                           show_title=True):
    """并列热力图: 左=4语义组, 右=Category内部细分(MA/HE/EX/SE token)"""
    counts = np.zeros(num_classes)
    for r in records:
        counts[r['pred_class']] += 1

    # 左图矩阵: [num_classes, 4 groups]
    mat_g = np.zeros((num_classes, len(GROUP_NAMES)))
    for r in records:
        c = r['pred_class']
        for j, g in enumerate(GROUP_NAMES):
            mat_g[c, j] += r['group_scores'].get(g, 0)
    for c in range(num_classes):
        if counts[c] > 0:
            mat_g[c] /= counts[c]

    # 右图矩阵: [num_classes, 4 cat tokens]
    mat_c = np.zeros((num_classes, len(CAT_NAMES)))
    for r in records:
        c = r['pred_class']
        for j, g in enumerate(CAT_NAMES):
            mat_c[c, j] += r['cat_scores'].get(g, 0)
    for c in range(num_classes):
        if counts[c] > 0:
            mat_c[c] /= counts[c]

    row_labels = [class_names[c] for c in range(num_classes)]
    h = max(3, num_classes * 0.9)

    fig, axes = plt.subplots(1, 2, figsize=(13, h),
                             gridspec_kw={'width_ratios': [4, 4]})

    # 左: 4 语义组
    ax = axes[0]
    im0 = ax.imshow(mat_g, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(GROUP_NAMES)))
    ax.set_xticklabels(GROUP_NAMES, rotation=25, ha='right', fontsize=16)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(row_labels, fontsize=16)
    if show_title:
        ax.set_title('High-Level Node Group', fontsize=18, fontweight='bold')
    cb0 = plt.colorbar(im0, ax=ax, label='Norm. IG', fraction=0.046, pad=0.04)
    cb0.set_label('Norm. IG', fontsize=16)
    cb0.ax.tick_params(labelsize=16)
    plt.draw()
    for label in cb0.ax.get_yticklabels():
        label.set_fontsize(16)
    for i in range(num_classes):
        for j in range(len(GROUP_NAMES)):
            ax.text(j, i, f'{mat_g[i,j]:.3f}', ha='center', va='center',
                    fontsize=16, color='black')

    # 右: Category 细分
    ax = axes[1]
    im1 = ax.imshow(mat_c, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xticks(range(len(CAT_NAMES)))
    ax.set_xticklabels(CAT_NAMES, rotation=25, ha='right', fontsize=16)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels(row_labels, fontsize=16)
    if show_title:
        ax.set_title('Categorical Node', fontsize=18, fontweight='bold')
    cb1 = plt.colorbar(im1, ax=ax, label='Norm. IG', fraction=0.046, pad=0.04)
    cb1.set_label('Norm. IG', fontsize=16)
    cb1.ax.tick_params(labelsize=16)
    plt.draw()
    for label in cb1.ax.get_yticklabels():
        label.set_fontsize(16)
    for i in range(num_classes):
        for j in range(len(CAT_NAMES)):
            ax.text(j, i, f'{mat_c[i,j]:.3f}', ha='center', va='center',
                    fontsize=16, color='black')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Per-class heatmap saved: {save_path}")


# ── 主函数 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='PARAGraph IG — Dataset Batch')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Use the same value as in training to rebuild the same splits')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'])
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--use_3_stages', action='store_true', default=False)
    parser.add_argument('--steps', type=int, default=20,
                        help='IG 积分步数 (批量时建议 20)')
    parser.add_argument('--save_dir', type=str, default='./ig_plots')
    parser.add_argument('--save_images', action='store_true', default=False,
                        help='保存每张图的 IG 可视化（默认关闭）')
    args = parser.parse_args()
    data_name = os.path.basename(os.path.normpath(args.data_root))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print(f"\nLoading PARAGraph: {args.checkpoint}")
    model = PARAGraph.load_from_checkpoint(
        args.checkpoint, device=str(device), eval_mode=True
    )
    model = model.to(device).eval()
    num_classes = model.num_classes
    class_names = (['Grade 0', 'Grade 1', 'Grade 2', 'Grade 3', 'Grade 4']
                   if num_classes == 5 else ['Healthy', 'NPDR', 'PDR'])

    # Dataloader
    print(f"\nDatasets: {args.data_root}, split={args.split}")
    dataloaders = create_grading_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=1,
        num_workers=2,
        use_3_stages=args.use_3_stages,
        use_oversample=False,
        use_strong_aug=False,
        split_seed=args.split_seed,
    )
    loader = dataloaders.get(args.split)
    if loader is None:
        print(f"No '{args.split}' loader, exiting.")
        return
    print(f"  {args.split} set: {len(loader.dataset)} samples")

    # 输出目录
    ckpt_name = (os.path.basename(os.path.dirname(args.checkpoint)) or
                 os.path.basename(args.checkpoint).replace('.pth', ''))
    out_dir = os.path.join(args.save_dir, f'{ckpt_name}_{data_name}')
    os.makedirs(out_dir, exist_ok=True)

    # ── 主循环 ─────────────────────────────────────────────────────────────────
    records = []
    csv_rows = []

    for batch in tqdm(loader, desc='Processing'):
        image_tensor = batch['image'].to(device)   # [1, 3, H, W]
        gt_class     = int(batch['label'][0].item())
        image_id     = batch['image_id'][0]         # str

        # Step 1: 完整 forward 建图
        with torch.no_grad():
            output = model(image_tensor)

        graph      = output['graphs'][0]
        pred_class = int(output['logits'][0].argmax().item())
        correct    = (pred_class == gt_class)

        # Step 2: 提取节点特征 & 边
        with torch.no_grad():
            node_features, edge_index = model.get_node_features_for_gnn(graph)
            edge_weights   = model.get_edge_weights_for_gnn(graph)
            edge_type_bias = model.get_edge_type_bias_for_gnn(graph)

        key_indices = get_key_indices(model, graph)

        # Step 3: IG（目标=预测类）
        node_attr = compute_ig(
            model, node_features, edge_index, edge_weights, edge_type_bias,
            key_indices, pred_class, steps=args.steps
        )

        # 按 4 语义组聚合归因，并做每张图的归一化（各组分数之和=1）
        all_nodes = collect_nodes(graph)
        N = len(node_attr)
        group_sums   = {g: 0.0 for g in GROUP_MAP}
        group_counts = {g: 0   for g in GROUP_MAP}
        for nd in all_nodes:
            if nd['idx'] >= N:
                continue
            for g, types in GROUP_MAP.items():
                if nd['type'] in types:
                    group_sums[g]   += float(node_attr[nd['idx']])
                    group_counts[g] += 1
        group_scores = {g: group_sums[g] / max(group_counts[g], 1) for g in GROUP_MAP}
        total = sum(group_scores.values()) + 1e-8
        group_scores_norm = {g: v / total for g, v in group_scores.items()}

        # Category 内部细分归一化（MA/HE/EX/SE token，各 token 之和=1）
        cat_sums   = {g: 0.0 for g in CAT_MAP}
        cat_counts = {g: 0   for g in CAT_MAP}
        for nd in all_nodes:
            if nd['idx'] >= N:
                continue
            for g, types in CAT_MAP.items():
                if nd['type'] in types:
                    cat_sums[g]   += float(node_attr[nd['idx']])
                    cat_counts[g] += 1
        cat_scores = {g: cat_sums[g] / max(cat_counts[g], 1) for g in CAT_MAP}
        cat_total  = sum(cat_scores.values()) + 1e-8
        cat_scores_norm = {g: v / cat_total for g, v in cat_scores.items()}

        records.append({
            'image_id':     image_id,
            'gt_class':     gt_class,
            'pred_class':   pred_class,
            'correct':      correct,
            'group_scores': group_scores_norm,
            'cat_scores':   cat_scores_norm,
        })

        # CSV 行
        row = {'image_id': image_id, 'gt': gt_class, 'pred': pred_class,
               'correct': int(correct)}
        for g in GROUP_NAMES:
            row[g] = f"{group_scores_norm.get(g, 0.0):.4f}"
        for g in CAT_NAMES:
            row[g] = f"{cat_scores_norm.get(g, 0.0):.4f}"
        csv_rows.append(row)

        # Step 4: 单图可视化（裁剪黑边原图，坐标缩放 + 偏移修正）
        if args.save_images:
            orig_path = batch['image_path'][0]
            image_np, orig_hw, crop_offset = load_original_image(orig_path)
            stem      = os.path.splitext(image_id)[0]
            mark      = 'C' if correct else f'E{gt_class}p{pred_class}'
            save_path = os.path.join(out_dir, f'{stem}_{mark}.png')
            plot_ig(image_np, graph, node_attr, pred_class, gt_class,
                    class_names, save_path,
                    image_size=args.image_size, orig_hw=orig_hw,
                    crop_offset=crop_offset)

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = os.path.join(args.save_dir, f'ig_summary_{ckpt_name}_{data_name}.csv')
    if csv_rows:
        fieldnames = list(csv_rows[0].keys())
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\nCSV saved: {csv_path}")

    # ── Summary 图 ─────────────────────────────────────────────────────────────
    summary_path  = os.path.join(
        args.save_dir, f'ig_bar_{ckpt_name}_{data_name}.png')
    heatmap_path  = os.path.join(
        args.save_dir, f'ig_heatmap_{ckpt_name}_{data_name}.png')

    plot_summary(records, num_classes, class_names, summary_path)
    plot_per_class_summary(records, num_classes, class_names, heatmap_path)

    # 仅正确预测样本的 heatmap（不加标题）
    correct_records = [r for r in records if r['correct']]
    if correct_records:
        heatmap_correct_path = os.path.join(
            args.save_dir, f'ig_heatmap_correct_{ckpt_name}_{data_name}.png')
        plot_per_class_summary(correct_records, num_classes, class_names,
                               heatmap_correct_path, show_title=True)

    # 终端打印整体 summary
    n_correct = sum(1 for r in records if r['correct'])
    print(f"\n{'─'*50}")
    print(f"Total: {len(records)}  Correct: {n_correct}  "
          f"Acc: {100*n_correct/len(records):.1f}%")
    print(f"\nMean normalized IG score by node group (overall):")
    all_means = {g: np.mean([r['group_scores'].get(g, 0) for r in records])
                 for g in GROUP_NAMES}
    for g in sorted(GROUP_NAMES, key=lambda x: -all_means[x]):
        print(f"  {g:<18s}: {all_means[g]:.4f}")

    print("\nDone!")


if __name__ == '__main__':
    main()
