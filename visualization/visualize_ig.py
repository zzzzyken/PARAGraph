#!/usr/bin/env python3
"""
PARAGraph 节点 Integrated Gradients (IG) 贡献可视化

对每个图节点的初始特征向量计算 IG 归因（baseline = 零向量），
将每个节点的归因强度 (||IG_i||_2) 叠加到原始眼底图上：
  - 点大小 / 颜色 (plasma) 反映归因强度
  - 点形状区分节点类型 (OD / Fovea / MA / HE / EX / SE / Token / Region)

用法:
    python visualization/visualize_ig.py \
        --checkpoint ./checkpoints/paragraph/paragraph_messidor/best_model.pth \
        --image /path/to/fundus.jpg \
        [--gt_class 2] \
        [--target_class -1]    # -1 = 使用预测类别
        [--steps 50]
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
import matplotlib.patheffects as pe
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from PIL import Image
from tqdm import trange
import torchvision.transforms as T

from Models import PARAGraph

# ── 图像预处理 ─────────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def preprocess(image_path, image_size):
    img = Image.open(image_path).convert('RGB')
    tfm = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tfm(img).unsqueeze(0)  # [1, 3, H, W]


def load_original_image(image_path):
    """
    加载原始图像，裁剪黑色边框。

    Returns:
        cropped:      [H_crop, W_crop, 3] uint8
        orig_hw:      (H_orig, W_orig)   裁剪前尺寸（坐标 scale 用）
        crop_offset:  (x0, y0)           crop 框左上角（坐标偏移用）
    """
    img = np.array(Image.open(image_path).convert('RGB'))
    H_orig, W_orig = img.shape[:2]

    gray = img.mean(axis=2)
    mask = gray > 10
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img, (H_orig, W_orig), (0, 0)

    y0, y1 = int(rows[0]), int(rows[-1])
    x0, x1 = int(cols[0]), int(cols[-1])
    return img[y0:y1 + 1, x0:x1 + 1], (H_orig, W_orig), (x0, y0)


# ── 可微 GNN 前向（固定图结构） ────────────────────────────────────────────────
def gnn_forward(model, h, edge_index, edge_weights, edge_type_bias, key_indices):
    """从初始节点特征出发，运行 GNN → 注意力池化 → 分类头，返回 logits [C]。"""
    # eval 模式下 F.dropout 是 no-op
    h = F.dropout(h, p=model.gnn_dropout, training=False)
    for gnn_layer, norm in zip(model.gnn_layers, model.gnn_norms):
        if edge_index.shape[1] > 0:
            h_new = gnn_layer(h, edge_index, edge_weights, edge_type_bias)
            h_new = norm(h_new)
            h_new = F.gelu(h_new)
            h_new = F.dropout(h_new, p=model.gnn_dropout, training=False)
            h = h + h_new

    key_feat = h[key_indices].unsqueeze(0)           # [1, K, D]
    queries  = model.graph_pool_queries.unsqueeze(0) # [1, C, D]
    pooled, _ = model.graph_pool_attn(queries, key_feat, key_feat)
    return model.classifier(pooled.squeeze(0).flatten())  # [num_classes]


# ── Integrated Gradients ───────────────────────────────────────────────────────
def compute_ig(model, node_features, edge_index, edge_weights, edge_type_bias,
               key_indices, target_class, steps=50):
    """
    Integrated Gradients 归因（baseline = 零向量）。

    IG_i = (f_i - 0) * E_{alpha in [0,1]}[d output_c / d h_i |_{h=alpha*f}]

    Returns:
        node_attr: np.ndarray [N]  每个节点的归因强度 (L2 norm over feature dim)
    """
    nf = node_features.detach()          # 脱离计算图
    ei = edge_index.detach()
    ew = edge_weights.detach()
    eb = edge_type_bias.detach()

    grad_sum = torch.zeros_like(nf)

    for step in trange(steps, desc='IG steps', leave=False):
        alpha = (step + 1) / steps
        # 构造叶子 tensor（alpha * f），启用梯度
        h_alpha = (alpha * nf).detach().clone().requires_grad_(True)

        logits = gnn_forward(model, h_alpha, ei, ew, eb, key_indices)
        logits[target_class].backward()

        grad_sum += h_alpha.grad.detach()

    avg_grad = grad_sum / steps
    ig_attr  = avg_grad * nf                        # [N, D]
    node_attr = ig_attr.norm(dim=-1).cpu().numpy()  # [N]
    return node_attr


# ── 节点类型元信息 ─────────────────────────────────────────────────────────────
NODE_NAMES = {
    0: 'OD', 1: 'Fovea',
    2: 'MA', 3: 'HE', 4: 'EX', 5: 'SE',
    6: 'MA Token', 7: 'HE Token', 8: 'EX Token', 9: 'SE Token',
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


# ── 可视化 ─────────────────────────────────────────────────────────────────────
def collect_nodes(graph):
    """收集图中所有节点的 (node_idx, type, absolute_coords)。"""
    nodes = []
    for i, nd in enumerate(graph['nodes']):
        nodes.append({'idx': i,
                      'type': nd['type'],
                      'abs': nd['absolute_coords']})
    for ln in graph.get('lesion_nodes', []):
        nodes.append({'idx': ln['node_idx'],
                      'type': ln['type'],
                      'abs': ln['absolute_coords']})
    for tn in graph.get('token_nodes', []):
        nodes.append({'idx': tn['node_idx'],
                      'type': tn['type'],
                      'abs': tn['absolute_coords']})
    for rn in graph.get('region_nodes', []):
        nodes.append({'idx': rn['node_idx'],
                      'type': rn['type'],
                      'abs': rn['absolute_coords']})
    return nodes


def plot_ig(image_np, graph, node_attr, pred_class, gt_class,
            class_names, save_path, image_size,
            orig_hw=None, crop_offset=(0, 0)):
    """
    image_np:     [H_crop, W_crop, 3] uint8  裁剪后原图
    orig_hw:      (H_orig, W_orig)            裁剪前尺寸（坐标 scale）
    crop_offset:  (x0, y0)                   crop 左上角偏移
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

    # 颜色: 全局归一化
    attr_max        = node_attr.max() + 1e-8
    attr_norm_color = node_attr / attr_max

    # 大小: 组内归一化
    GROUP = {0: 'struct', 1: 'struct',
             2: 'lesion', 3: 'lesion', 4: 'lesion', 5: 'lesion',
             6: 'token',  7: 'token',  8: 'token',  9: 'token',
             10: 'region'}
    SIZE_RANGE = {'struct': (250, 600), 'token': (150, 450),
                  'region': (120, 380), 'lesion': (60, 280)}
    group_max = {}
    for nd in valid:
        g = GROUP.get(nd['type'], 'lesion')
        v = float(node_attr[nd['idx']])
        group_max[g] = max(group_max.get(g, 0.0), v)

    cmap     = plt.cm.plasma
    norm_map = Normalize(vmin=0, vmax=1)

    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(image_np)
    ax.axis('off')

    pred_name = class_names[pred_class]
    gt_name   = class_names[gt_class]
    tick = '✓' if pred_class == gt_class else '✗'
    ax.set_title(
        f'Node IG Attribution  —  Pred: {pred_name}  GT: {gt_name}  {tick}',
        fontsize=35, fontweight='bold', pad=8
    )

    # 按归因从小到大画（高归因最后画，在最上层）
    plotted_types = set()
    for node in sorted(valid, key=lambda n: float(node_attr[n['idx']])):
        idx   = node['idx']
        ntype = node['type']
        cx, cy = node['abs']
        cx = cx * scale_x - off_x
        cy = cy * scale_y - off_y

        a_color = float(attr_norm_color[idx])
        g       = GROUP.get(ntype, 'lesion')
        a_size  = float(node_attr[idx]) / (group_max.get(g, 1e-8) + 1e-8)
        s_min, s_max = SIZE_RANGE[g]
        size = s_min + a_size * (s_max - s_min)

        ax.scatter([cx], [cy], s=size, c=[cmap(norm_map(a_color))],
                   marker=NODE_MARKERS.get(ntype, 'o'),
                   edgecolors=NODE_EDGE_COLORS.get(ntype, 'white'),
                   linewidths=1.2, alpha=0.9, zorder=5)
        plotted_types.add(ntype)

    # Colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm_map)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.028, pad=0.01)
    cbar.set_label('IG Attribution (normalized)', fontsize=35, fontweight='bold')
    cbar.ax.tick_params(labelsize=35)
    plt.draw()
    for label in cbar.ax.get_yticklabels():
        label.set_fontsize(35)
        label.set_fontweight('bold')

    # 图例：形状 → 节点类型
    handles = []
    for ntype in sorted(plotted_types):
        h = mlines.Line2D([], [], marker=NODE_MARKERS.get(ntype, 'o'),
                          linestyle='None', markersize=8,
                          label=NODE_NAMES.get(ntype, f'Type {ntype}'),
                          markeredgecolor=NODE_EDGE_COLORS.get(ntype, 'gray'),
                          color='gray')
        handles.append(h)
    ax.legend(handles=handles, loc='lower right', framealpha=0.85, ncol=2,
              prop={'weight': 'bold', 'size': 35})

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close(fig)


# ── 主函数 ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='PARAGraph Node IG Visualization')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='PARAGraph best_model.pth 路径')
    parser.add_argument('--image', type=str, required=True,
                        help='输入眼底图路径 (jpg/png)')
    parser.add_argument('--gt_class', type=int, default=-1,
                        help='真实 DR 等级 (-1=未知)')
    parser.add_argument('--target_class', type=int, default=-1,
                        help='IG 目标类 (-1=使用预测类)')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=50,
                        help='IG 积分步数')
    parser.add_argument('--save_dir', type=str, default='./ig_plots')
    args = parser.parse_args()

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

    # 加载图像：原图用于显示，预处理版本用于模型推理
    print(f"Loading image: {args.image}")
    image_tensor                    = preprocess(args.image, args.image_size).to(device)
    image_np, orig_hw, crop_offset  = load_original_image(args.image)

    # ── Step 1: 完整 forward 建图 ──────────────────────────────────────────────
    print("\nBuilding graph (full forward)...")
    with torch.no_grad():
        output = model(image_tensor)

    graph      = output['graphs'][0]
    pred_class = int(output['logits'][0].argmax().item())
    gt_class   = args.gt_class if args.gt_class >= 0 else pred_class
    target_cls = args.target_class if args.target_class >= 0 else pred_class

    probs = output['probs'][0].cpu().numpy()
    print(f"Prediction: {class_names[pred_class]}  (conf={probs[pred_class]:.3f})")
    if args.gt_class >= 0:
        print(f"Ground truth: {class_names[gt_class]}")
    print(f"IG target: {class_names[target_cls]}")

    num_lesions = graph.get('num_lesions', 0)
    num_tokens  = graph.get('num_tokens', 0)
    num_regions = graph.get('num_regions', 0)
    print(f"Graph nodes: OD+Fovea=2, lesions={num_lesions}, "
          f"tokens={num_tokens}, regions={num_regions}")

    # ── Step 2: 提取节点特征 & 边信息 ─────────────────────────────────────────
    with torch.no_grad():
        node_features, edge_index = model.get_node_features_for_gnn(graph)
        edge_weights   = model.get_edge_weights_for_gnn(graph)
        edge_type_bias = model.get_edge_type_bias_for_gnn(graph)

    # key_indices（与 forward 完全一致）
    token_start_idx  = 2 + num_lesions
    region_start_idx = 2 + num_lesions + num_tokens
    if model.use_optic_nodes:
        key_indices = [0, 1]
        if num_tokens  > 0:
            key_indices += list(range(token_start_idx,  token_start_idx  + num_tokens))
        if num_regions > 0:
            key_indices += list(range(region_start_idx, region_start_idx + num_regions))
    else:
        key_indices = (list(range(region_start_idx, region_start_idx + num_regions))
                       if num_regions > 0 else list(range(len(node_features))))

    print(f"Key nodes for pooling: {len(key_indices)}")

    # ── Step 3: IG ────────────────────────────────────────────────────────────
    print(f"\nComputing IG (steps={args.steps})...")
    node_attr = compute_ig(
        model, node_features, edge_index, edge_weights, edge_type_bias,
        key_indices, target_cls, steps=args.steps
    )

    # top-5 归因节点
    all_nodes_info = {}
    for i, nd in enumerate(graph['nodes']):
        all_nodes_info[i] = NODE_NAMES.get(nd['type'], '?')
    for ln in graph.get('lesion_nodes', []):
        all_nodes_info[ln['node_idx']] = (
            f"{ln['lesion_name']}(conf={ln['confidence']:.2f})")
    for tn in graph.get('token_nodes', []):
        all_nodes_info[tn['node_idx']] = tn['token_name']
    for rn in graph.get('region_nodes', []):
        all_nodes_info[rn['node_idx']] = f"Region(n={rn['num_lesions']})"

    top5 = np.argsort(node_attr)[::-1][:5]
    print("Top-5 nodes by IG attribution:")
    for rank, idx in enumerate(top5):
        info = all_nodes_info.get(int(idx), f"idx={idx}")
        print(f"  #{rank+1}: {info:<30s}  {node_attr[idx]:.4f}")

    # ── Step 4: 可视化 ─────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    img_stem  = os.path.splitext(os.path.basename(args.image))[0]
    ckpt_name = (os.path.basename(os.path.dirname(args.checkpoint)) or
                 os.path.basename(args.checkpoint).replace('.pth', ''))
    save_path = os.path.join(
        args.save_dir, f'ig_{ckpt_name}_{img_stem}_pred{pred_class}.png'
    )

    plot_ig(image_np, graph, node_attr,
            pred_class=pred_class, gt_class=gt_class,
            class_names=class_names, save_path=save_path,
            image_size=args.image_size, orig_hw=orig_hw,
            crop_offset=crop_offset)
    print("\nDone!")


if __name__ == '__main__':
    main()
