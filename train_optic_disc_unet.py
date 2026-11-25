"""
训练 UNet 用于 Optic Disc 分割

使用 IDRiD 数据集中的 Optic Disc 分割标注
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

from Models import OpticUNet
from segmentation_dataloader import create_optic_disc_dataloaders

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed, logging disabled")


def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    """
    计算 Dice 系数

    Args:
        pred: [B, C, H, W] logits
        target: [B, C, H, W] binary mask
        threshold: 二值化阈值
    """
    pred = (torch.sigmoid(pred) > threshold).float()

    # 按 batch 计算
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    pred_sum = pred_flat.sum(dim=1)
    target_sum = target_flat.sum(dim=1)

    dice = (2. * intersection + smooth) / (pred_sum + target_sum + smooth)
    return dice.mean()


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    计算 IoU (Intersection over Union)

    Args:
        pred: [B, C, H, W] logits
        target: [B, C, H, W] binary mask
        threshold: 二值化阈值
    """
    pred = (torch.sigmoid(pred) > threshold).float()

    # 按 batch 计算
    B = pred.shape[0]
    pred_flat = pred.view(B, -1)
    target_flat = target.view(B, -1)

    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.mean()


class DiceBCELoss(nn.Module):
    """Dice + BCE 组合损失"""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred, target)

        # Dice Loss
        pred_prob = torch.sigmoid(pred)
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + 1e-6) / (pred_flat.sum() + target_flat.sum() + 1e-6)
        dice_loss = 1 - dice

        return self.bce_weight * bce + self.dice_weight * dice_loss


def train_epoch(model, dataloader, optimizer, criterion, device, epoch, writer=None):
    """训练一个 epoch"""
    model.train()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        optimizer.zero_grad()

        output = model(images)
        pred = output['od_mask']

        # 确保尺寸匹配
        if pred.shape[-2:] != masks.shape[-2:]:
            masks = F.interpolate(masks, size=pred.shape[-2:], mode='nearest')

        loss = criterion(pred, masks)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 计算指标
        with torch.no_grad():
            dice = dice_coefficient(pred, masks)
            iou = iou_score(pred, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice.item():.4f}',
            'iou': f'{iou.item():.4f}'
        })

    n_batches = len(dataloader)
    metrics = {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'iou': total_iou / n_batches
    }

    return metrics


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)

        output = model(images)
        pred = output['od_mask']

        if pred.shape[-2:] != masks.shape[-2:]:
            masks = F.interpolate(masks, size=pred.shape[-2:], mode='nearest')

        loss = criterion(pred, masks)
        dice = dice_coefficient(pred, masks)
        iou = iou_score(pred, masks)

        total_loss += loss.item()
        total_dice += dice.item()
        total_iou += iou.item()

    n_batches = len(dataloader)
    metrics = {
        'loss': total_loss / n_batches,
        'dice': total_dice / n_batches,
        'iou': total_iou / n_batches
    }

    return metrics


def visualize_predictions(model, dataloader, device, save_path, num_samples=4):
    """可视化预测结果"""
    import matplotlib.pyplot as plt

    model.eval()

    batch = next(iter(dataloader))
    images = batch['image'][:num_samples].to(device)
    masks = batch['mask'][:num_samples]

    with torch.no_grad():
        output = model(images)
        preds = torch.sigmoid(output['od_mask'])

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_vis = images.cpu() * std + mean
    images_vis = images_vis.numpy().transpose(0, 2, 3, 1)
    images_vis = np.clip(images_vis, 0, 1)

    # Resize masks/preds to match if needed
    if preds.shape[-2:] != masks.shape[-2:]:
        preds = F.interpolate(preds, size=masks.shape[-2:], mode='bilinear')

    masks_vis = masks.cpu().numpy()[:, 0]
    preds_vis = preds.cpu().numpy()[:, 0]

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        # Original image
        axes[i, 0].imshow(images_vis[i])
        axes[i, 0].set_title('Original')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(masks_vis[i], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')

        # Predicted mask
        axes[i, 2].imshow(preds_vis[i], cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')

        # Overlay
        overlay = images_vis[i].copy()
        pred_binary = (preds_vis[i] > 0.5).astype(np.float32)
        overlay[..., 1] = np.clip(overlay[..., 1] + pred_binary * 0.3, 0, 1)  # Green overlay
        axes[i, 3].imshow(overlay)
        axes[i, 3].set_title('Overlay')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def main(args):
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 创建数据加载器
    print("\nLoading data...")
    dataloaders = create_optic_disc_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 创建模型
    print("\nCreating model...")
    model = OpticUNet(encoder_pretrained=True, predict_fovea=False,
                      encoder_name=args.encoder)
    model = model.to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器
    criterion = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # TensorBoard
    writer = None
    if HAS_TENSORBOARD and args.log_dir:
        log_dir = os.path.join(args.log_dir, 'optic_disc_unet')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    # 训练循环
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    best_dice = 0

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer, criterion, device, epoch, writer
        )

        # 评估
        val_metrics = evaluate(model, dataloaders['val'], criterion, device)

        scheduler.step()

        # 打印结果
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}, IoU: {train_metrics['iou']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Dice: {val_metrics['dice']:.4f}, IoU: {val_metrics['iou']:.4f}")

        # TensorBoard 日志
        if writer:
            writer.add_scalars('Loss', {'train': train_metrics['loss'], 'val': val_metrics['loss']}, epoch)
            writer.add_scalars('Dice', {'train': train_metrics['dice'], 'val': val_metrics['dice']}, epoch)
            writer.add_scalars('IoU', {'train': train_metrics['iou'], 'val': val_metrics['iou']}, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # 保存最佳模型
        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': best_dice,
                'val_iou': val_metrics['iou'],
                'val_loss': val_metrics['loss']
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (Dice: {best_dice:.4f})")

    # 关闭 TensorBoard
    if writer:
        writer.close()

    # 可视化
    print("\n" + "=" * 60)
    print("Generating visualizations...")
    print("=" * 60)

    # 加载最佳模型
    checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    visualize_predictions(
        model, dataloaders['test'], device,
        save_path=os.path.join(args.save_dir, 'predictions.png'),
        num_samples=4
    )

    print(f"\nTraining completed!")
    print(f"Best validation Dice: {best_dice:.4f}")
    print(f"Model saved to: {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train UNet for Optic Disc Segmentation')

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='Dataset folder in the fixed format (see README, Data): images/<split>/ and masks/<split>/OD/')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Input image size')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # 模型参数
    parser.add_argument('--encoder', type=str, default='resnet50',
                       choices=['resnet50'],
                       help='Encoder backbone')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints/optic_unet',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs',
                       help='TensorBoard log directory')

    args = parser.parse_args()
    main(args)
