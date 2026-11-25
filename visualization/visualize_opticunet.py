"""
OpticUNet 完整可视化

在图像上同时展示:
- Optic Disc 分割结果
- Fovea Center 定位结果
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image

from Models import OpticUNet
from segmentation_dataloader import create_fovea_dataloaders


def visualize_model(
    model,
    dataloader,
    device,
    save_dir,
    num_samples=8,
    od_sigma=100.0
):
    """
    可视化 OpticUNet 的完整输出

    Args:
        model: OpticUNet 模型
        dataloader: 数据加载器
        device: 设备
        save_dir: 保存目录
        num_samples: 可视化样本数量
        od_sigma: OD 距离约束的高斯标准差
    """
    model.eval()

    # 收集足够的样本
    all_images = []
    all_gt_fovea_x = []
    all_gt_fovea_y = []
    all_image_ids = []

    for batch in dataloader:
        all_images.append(batch['image'])
        all_gt_fovea_x.append(batch['fovea_x'])
        all_gt_fovea_y.append(batch['fovea_y'])
        all_image_ids.extend(batch['image_id'])

        if sum(img.shape[0] for img in all_images) >= num_samples:
            break

    images = torch.cat(all_images, dim=0)[:num_samples]
    gt_fovea_x = torch.cat(all_gt_fovea_x, dim=0)[:num_samples]
    gt_fovea_y = torch.cat(all_gt_fovea_y, dim=0)[:num_samples]
    image_ids = all_image_ids[:num_samples]

    images = images.to(device)

    with torch.no_grad():
        output = model(images)

        # OD 分割
        od_mask_logits = output['od_mask']
        od_mask_prob = torch.sigmoid(od_mask_logits)

        # Fovea 定位
        fovea_heatmap_logits = output['fovea_heatmap']
        fovea_heatmap_prob = torch.sigmoid(fovea_heatmap_logits)

        # 获取 Fovea 坐标
        pred_coords = model.get_fovea_coordinates(fovea_heatmap_logits, method='argmax')
        pred_coords = pred_coords.float()

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    images_vis = images * std + mean
    images_vis = images_vis.cpu().numpy().transpose(0, 2, 3, 1)
    images_vis = np.clip(images_vis, 0, 1)

    # 确保尺寸匹配
    H, W = images_vis.shape[1:3]
    if od_mask_prob.shape[-2:] != (H, W):
        od_mask_prob = F.interpolate(od_mask_prob, size=(H, W), mode='bilinear', align_corners=False)
    if fovea_heatmap_prob.shape[-2:] != (H, W):
        fovea_heatmap_prob = F.interpolate(fovea_heatmap_prob, size=(H, W), mode='bilinear', align_corners=False)

    od_mask_np = od_mask_prob.cpu().numpy()[:, 0]
    fovea_heatmap_np = fovea_heatmap_prob.cpu().numpy()[:, 0]

    # ============ 可视化 1: 综合展示 ============
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(num_samples):
        pred_x = pred_coords[i, 0].cpu().item()
        pred_y = pred_coords[i, 1].cpu().item()
        gt_x = gt_fovea_x[i].item()
        gt_y = gt_fovea_y[i].item()
        dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        # 1. 原图 + OD 分割 + Fovea 定位
        axes[i, 0].imshow(images_vis[i])

        # OD 分割轮廓 (绿色)
        od_binary = (od_mask_np[i] > 0.5).astype(np.float32)
        axes[i, 0].contour(od_binary, levels=[0.5], colors='lime', linewidths=2)

        # Fovea: GT (绿色 x) 和 Pred (红色 +)
        axes[i, 0].scatter(gt_x, gt_y, c='cyan', s=200, marker='x', linewidths=3, label='GT Fovea', zorder=10)
        axes[i, 0].scatter(pred_x, pred_y, c='red', s=200, marker='+', linewidths=3, label='Pred Fovea', zorder=10)

        # 画 GT 和 Pred 之间的连线
        axes[i, 0].plot([gt_x, pred_x], [gt_y, pred_y], 'y--', linewidth=2, alpha=0.7)

        axes[i, 0].legend(loc='upper right', fontsize=10)
        axes[i, 0].set_title(f'{image_ids[i]} | Dist: {dist:.1f}px', fontsize=12)
        axes[i, 0].axis('off')

        # 2. OD 分割 mask
        axes[i, 1].imshow(images_vis[i])
        od_overlay = np.zeros((*od_mask_np[i].shape, 4))
        od_overlay[..., 1] = 1  # Green
        od_overlay[..., 3] = od_mask_np[i] * 0.5  # Alpha
        axes[i, 1].imshow(od_overlay)
        axes[i, 1].set_title('OD Segmentation', fontsize=12)
        axes[i, 1].axis('off')

        # 3. Fovea heatmap
        axes[i, 2].imshow(images_vis[i], alpha=0.5)
        axes[i, 2].imshow(fovea_heatmap_np[i], cmap='hot', alpha=0.6)
        axes[i, 2].scatter(pred_x, pred_y, c='cyan', s=150, marker='+', linewidths=2, zorder=10)
        axes[i, 2].set_title('Fovea Heatmap', fontsize=12)
        axes[i, 2].axis('off')

        # 4. OD 距离权重图
        od_weight = model.compute_od_distance_weight(
            od_mask_logits[i:i+1], sigma=od_sigma
        ).cpu().numpy()[0, 0]
        axes[i, 3].imshow(od_weight, cmap='viridis')
        axes[i, 3].set_title(f'OD Distance Weight (σ={od_sigma})', fontsize=12)
        axes[i, 3].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'opticunet_full_visualization.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ============ 可视化 2: 单独的大图展示 ============
    for i in range(min(4, num_samples)):
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))

        pred_x = pred_coords[i, 0].cpu().item()
        pred_y = pred_coords[i, 1].cpu().item()
        gt_x = gt_fovea_x[i].item()
        gt_y = gt_fovea_y[i].item()
        dist = np.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)

        # 显示原图
        ax.imshow(images_vis[i])

        # OD 分割 (绿色填充 + 轮廓)
        od_binary = (od_mask_np[i] > 0.5).astype(np.float32)
        od_overlay = np.zeros((*od_mask_np[i].shape, 4))
        od_overlay[..., 1] = 1  # Green
        od_overlay[..., 3] = od_binary * 0.3  # Alpha
        ax.imshow(od_overlay)
        ax.contour(od_binary, levels=[0.5], colors='lime', linewidths=3)

        # Fovea 标注
        ax.scatter(gt_x, gt_y, c='cyan', s=400, marker='x', linewidths=4, label=f'GT Fovea ({gt_x:.0f}, {gt_y:.0f})', zorder=10)
        ax.scatter(pred_x, pred_y, c='red', s=400, marker='+', linewidths=4, label=f'Pred Fovea ({pred_x:.0f}, {pred_y:.0f})', zorder=10)

        # 距离连线
        ax.plot([gt_x, pred_x], [gt_y, pred_y], 'yellow', linewidth=2, linestyle='--', alpha=0.8)

        # 在图像上显示距离
        mid_x = (gt_x + pred_x) / 2
        mid_y = (gt_y + pred_y) / 2
        ax.annotate(f'{dist:.1f}px', xy=(mid_x, mid_y), fontsize=14,
                    color='yellow', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

        ax.legend(loc='upper left', fontsize=12, facecolor='white', framealpha=0.9)
        ax.set_title(f'{image_ids[i]} - OD Segmentation + Fovea Localization', fontsize=14, fontweight='bold')
        ax.axis('off')

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'opticunet_detail_{image_ids[i]}.png')
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")

    # ============ 可视化 3: 统计信息 ============
    # 计算所有样本的距离
    all_distances = []
    for batch in dataloader:
        imgs = batch['image'].to(device)
        gx = batch['fovea_x']
        gy = batch['fovea_y']

        with torch.no_grad():
            out = model(imgs)
            coords = model.get_fovea_coordinates(
                out['fovea_heatmap'],
                method='argmax'
            ).cpu()

        for j in range(imgs.shape[0]):
            d = np.sqrt((coords[j, 0].item() - gx[j].item()) ** 2 +
                       (coords[j, 1].item() - gy[j].item()) ** 2)
            all_distances.append(d)

    all_distances = np.array(all_distances)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 距离直方图
    axes[0].hist(all_distances, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(np.mean(all_distances), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(all_distances):.1f}px')
    axes[0].axvline(np.median(all_distances), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(all_distances):.1f}px')
    axes[0].set_xlabel('Euclidean Distance (pixels)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Fovea Localization Error Distribution', fontsize=14)
    axes[0].legend(fontsize=11)

    # 统计信息文本
    stats_text = f"""
    Total Samples: {len(all_distances)}
    Mean Distance: {np.mean(all_distances):.2f} px
    Median Distance: {np.median(all_distances):.2f} px
    Std Distance: {np.std(all_distances):.2f} px
    Min Distance: {np.min(all_distances):.2f} px
    Max Distance: {np.max(all_distances):.2f} px

    < 10px: {(all_distances < 10).sum()} ({100*(all_distances < 10).mean():.1f}%)
    < 20px: {(all_distances < 20).sum()} ({100*(all_distances < 20).mean():.1f}%)
    < 50px: {(all_distances < 50).sum()} ({100*(all_distances < 50).mean():.1f}%)
    """
    axes[1].text(0.1, 0.5, stats_text, fontsize=14, family='monospace',
                 verticalalignment='center', transform=axes[1].transAxes)
    axes[1].set_title('Statistics', fontsize=14)
    axes[1].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'opticunet_statistics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    print(f"\n=== Fovea Localization Statistics ===")
    print(f"Mean Distance: {np.mean(all_distances):.2f} px")
    print(f"Median Distance: {np.median(all_distances):.2f} px")
    print(f"Std Distance: {np.std(all_distances):.2f} px")


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 创建数据加载器
    print("\nLoading data...")
    dataloaders = create_fovea_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        heatmap_sigma=15.0
    )

    # 创建模型
    print("\nCreating model...")
    model = OpticUNet(encoder_pretrained=False, predict_fovea=True)

    # 加载 checkpoint
    print(f"\nLoading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_distance' in checkpoint:
        print(f"  Best val distance: {checkpoint['val_distance']:.1f}px")

    model = model.to(device)
    model.eval()

    # 可视化
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    visualize_model(
        model=model,
        dataloader=dataloaders['test'],
        device=device,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        od_sigma=args.od_sigma
    )

    print(f"\nVisualization completed!")
    print(f"Results saved to: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize OpticUNet Model')

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='Fovea dataset folder in the fixed format (see README, Data)')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Input image size')

    # 模型参数
    parser.add_argument('--checkpoint', type=str,
                       default='./checkpoints/fovea/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--od_sigma', type=float, default=100.0,
                       help='OD distance constraint sigma')

    # 可视化参数
    parser.add_argument('--num_samples', type=int, default=8,
                       help='Number of samples to visualize')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints/fovea/visualizations',
                       help='Directory to save visualizations')

    args = parser.parse_args()
    main(args)
