#!/usr/bin/env python3
"""
可视化 LesionUNet 在病灶分割数据集上的预测结果 (--data_root: lesion dataset folder in the fixed format, see README)
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.lesion_unet import LesionUNet
from segmentation_dataloader import MaskDataset, LESION_NAMES, get_lesion_val_transform

# 路径配置
MODEL_PATH = "./checkpoints_unet/best_model.pth"

# 病变信息
LESION_INFO = {
    0: {'name': 'MA', 'full_name': 'Microaneurysms', 'color': (255, 0, 0)},
    1: {'name': 'HE', 'full_name': 'Haemorrhages', 'color': (0, 255, 0)},
    2: {'name': 'EX', 'full_name': 'Hard Exudates', 'color': (255, 255, 0)},
    3: {'name': 'SE', 'full_name': 'Soft Exudates', 'color': (0, 255, 255)},
}


def denormalize_image(image_tensor):
    """反归一化图像 (ImageNet mean/std)"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image_tensor.cpu() * std + mean
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image, 0, 1)


def overlay_mask(image, mask, color, alpha=0.5):
    """将 mask 叠加到图像上"""
    overlay = image.copy()
    color_normalized = np.array(color) / 255.0

    mask_binary = mask > 0.5
    if mask_binary.sum() > 0:
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_binary,
                overlay[:, :, c] * (1 - alpha) + color_normalized[c] * alpha,
                overlay[:, :, c]
            )
    return overlay


def visualize_prediction(model, sample, device, save_path=None):
    """可视化单个样本的预测结果"""
    model.eval()

    image = sample['image'].unsqueeze(0).to(device)
    gt_mask = sample['mask'].cpu().numpy()  # [4, H, W]
    image_id = sample['image_id']

    # 推理
    with torch.no_grad():
        output = model(image, mode='full')
        pred_logits = output['masks']  # [1, 4, H, W]
        pred_mask = torch.sigmoid(pred_logits).squeeze(0).cpu().numpy()  # [4, H, W]

    # 反归一化图像
    img_np = denormalize_image(sample['image'])

    # 创建可视化
    fig, axes = plt.subplots(3, 5, figsize=(25, 15))

    # 第一行: 原图 + 4个GT mask
    axes[0, 0].imshow(img_np)
    axes[0, 0].set_title(f'Original: {image_id}', fontsize=12)
    axes[0, 0].axis('off')

    for i in range(4):
        info = LESION_INFO[i]
        gt = gt_mask[i]
        overlay = overlay_mask(img_np, gt, info['color'], alpha=0.6)
        axes[0, i+1].imshow(overlay)
        pixel_count = int((gt > 0.5).sum())
        axes[0, i+1].set_title(f'GT {info["name"]}: {pixel_count} px', fontsize=11,
                               color='green' if pixel_count > 0 else 'gray')
        axes[0, i+1].axis('off')

    # 第二行: 原图 + 4个预测 mask
    axes[1, 0].imshow(img_np)
    axes[1, 0].set_title('Original', fontsize=12)
    axes[1, 0].axis('off')

    for i in range(4):
        info = LESION_INFO[i]
        pred = pred_mask[i]
        overlay = overlay_mask(img_np, pred, info['color'], alpha=0.6)
        axes[1, i+1].imshow(overlay)
        pixel_count = int((pred > 0.5).sum())
        axes[1, i+1].set_title(f'Pred {info["name"]}: {pixel_count} px', fontsize=11,
                               color='blue' if pixel_count > 0 else 'gray')
        axes[1, i+1].axis('off')

    # 第三行: GT所有病变叠加 + Pred所有病变叠加 + 各病变对比
    # GT 所有
    gt_all = img_np.copy()
    for i in range(4):
        gt_all = overlay_mask(gt_all, gt_mask[i], LESION_INFO[i]['color'], alpha=0.5)
    axes[2, 0].imshow(gt_all)
    axes[2, 0].set_title('GT All Lesions', fontsize=12)
    axes[2, 0].axis('off')

    # Pred 所有
    pred_all = img_np.copy()
    for i in range(4):
        pred_all = overlay_mask(pred_all, pred_mask[i], LESION_INFO[i]['color'], alpha=0.5)
    axes[2, 1].imshow(pred_all)
    axes[2, 1].set_title('Pred All Lesions', fontsize=12)
    axes[2, 1].axis('off')

    # 各病变的预测概率热力图
    for i in range(3):
        if i < 4:
            info = LESION_INFO[i]
            im = axes[2, i+2].imshow(pred_mask[i], cmap='hot', vmin=0, vmax=1)
            axes[2, i+2].set_title(f'{info["name"]} Probability', fontsize=11)
            axes[2, i+2].axis('off')
            plt.colorbar(im, ax=axes[2, i+2], fraction=0.046, pad=0.04)

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=tuple(c/255 for c in info['color']),
                            label=f"{info['name']}: {info['full_name']}")
                      for info in LESION_INFO.values()]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=11)

    plt.suptitle(f'LesionUNet Prediction - {image_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.06, top=0.94)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    else:
        plt.show()

    plt.close()

    # 返回 IoU 统计
    ious = {}
    for i in range(4):
        gt_binary = gt_mask[i] > 0.5
        pred_binary = pred_mask[i] > 0.5
        intersection = (gt_binary & pred_binary).sum()
        union = (gt_binary | pred_binary).sum()
        iou = intersection / (union + 1e-6)
        ious[LESION_INFO[i]['name']] = iou

    return ious


def main(args):
    print("=" * 60)
    print("LesionUNet Prediction Visualization")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 加载模型
    print(f"\nLoading model from {args.model_path}...")
    model = LesionUNet(num_lesions=4, encoder_pretrained=False)

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best IoU: {checkpoint.get('best_iou', 'unknown')}")
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # 加载数据集
    print(f"\nLoading the {args.split} split of {args.data_root}...")
    transform = get_lesion_val_transform(args.image_size)
    dataset = MaskDataset(args.data_root, args.split, LESION_NAMES, transform, args.image_size)
    print(f"  Loaded {len(dataset)} images")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 可视化
    print(f"\nGenerating visualizations...")
    all_ious = {name: [] for name in ['MA', 'HE', 'EX', 'SE']}

    for i in range(min(args.n_images, len(dataset))):
        sample = dataset[i]
        image_id = sample['image_id']
        print(f"  Processing {i+1}/{args.n_images}: {image_id}")

        save_path = os.path.join(args.save_dir, f"{image_id}_prediction.png")
        ious = visualize_prediction(model, sample, device, save_path)

        for name, iou in ious.items():
            all_ious[name].append(iou)

        print(f"    IoU - MA: {ious['MA']:.4f}, HE: {ious['HE']:.4f}, "
              f"EX: {ious['EX']:.4f}, SE: {ious['SE']:.4f}")

    # 打印平均 IoU
    print(f"\n" + "=" * 60)
    print("Average IoU:")
    for name, iou_list in all_ious.items():
        avg_iou = np.mean(iou_list)
        print(f"  {name}: {avg_iou:.4f}")
    print(f"  Mean: {np.mean([np.mean(v) for v in all_ious.values()]):.4f}")

    print(f"\nDone! Visualizations saved to {args.save_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Visualize LesionUNet predictions')
    parser.add_argument('--model_path', type=str, default='./checkpoints_unet/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Lesion dataset folder in the fixed format (see README, Data)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Dataset split')
    parser.add_argument('--image_size', type=int, default=512,
                       help='Input image size')
    parser.add_argument('--n_images', type=int, default=5,
                       help='Number of images to visualize')
    parser.add_argument('--save_dir', type=str, default='./unet_predictions',
                       help='Directory to save visualizations')

    args = parser.parse_args()
    main(args)
