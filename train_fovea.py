"""
训练 Fovea 定位任务

基于已训练的 Optic Disc 模型，冻结 Encoder 和 OD Decoder，只训练 Fovea Decoder
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from Models import OpticUNet
from segmentation_dataloader import create_fovea_dataloaders

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed, logging disabled")


class FoveaHeatmapLoss(nn.Module):
    """
    Fovea Heatmap 损失: BCE + MSE 组合

    BCE: 把 heatmap 当作每个像素的概率
    MSE: 回归 heatmap 值
    """

    def __init__(self, bce_weight=1.0, mse_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.mse_weight = mse_weight

    def forward(self, pred, target):
        """
        Args:
            pred: [B, 1, H, W] 预测的 heatmap (logits)
            target: [B, 1, H, W] GT heatmap (0-1)
        """
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(pred, target)

        # MSE Loss
        pred_prob = torch.sigmoid(pred)
        mse_loss = F.mse_loss(pred_prob, target)

        total_loss = self.bce_weight * bce_loss + self.mse_weight * mse_loss

        return total_loss


def euclidean_distance(pred_coords, gt_coords):
    """
    计算预测坐标和真实坐标之间的欧氏距离

    Args:
        pred_coords: [B, 2] (x, y)
        gt_coords: [B, 2] (x, y)

    Returns:
        mean distance in pixels
    """
    dist = torch.sqrt(((pred_coords - gt_coords) ** 2).sum(dim=1))
    return dist.mean()


def encoder_modules(model):
    """Encoder sub-modules of OpticUNet (ResNet-50)."""
    return [model.encoder_stem, model.encoder_stage1, model.encoder_stage2,
            model.encoder_stage3, model.encoder_stage4]


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """训练一个 epoch"""
    model.train()

    # 确保只有 fovea decoder 在训练
    for module in encoder_modules(model):
        module.eval()
    model.od_decoder_stage4.eval()
    model.od_decoder_stage3.eval()
    model.od_decoder_stage2.eval()
    model.od_decoder_stage1.eval()
    model.od_final_upsample.eval()
    model.od_head.eval()

    total_loss = 0
    total_dist = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch in pbar:
        images = batch['image'].to(device)
        heatmaps = batch['heatmap'].to(device)
        gt_x = batch['fovea_x'].to(device)
        gt_y = batch['fovea_y'].to(device)
        gt_coords = torch.stack([gt_x, gt_y], dim=1)

        optimizer.zero_grad()

        output = model(images)
        pred_heatmap = output['fovea_heatmap']

        # 确保尺寸匹配
        if pred_heatmap.shape[-2:] != heatmaps.shape[-2:]:
            heatmaps = F.interpolate(heatmaps, size=pred_heatmap.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(pred_heatmap, heatmaps)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # 计算坐标距离
        with torch.no_grad():
            pred_coords = model.get_fovea_coordinates(pred_heatmap, method='argmax')
            pred_coords = pred_coords.float()
            dist = euclidean_distance(pred_coords, gt_coords)

        total_loss += loss.item()
        total_dist += dist.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dist': f'{dist.item():.1f}px'
        })

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'distance': total_dist / n_batches
    }


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """评估模型"""
    model.eval()

    total_loss = 0
    total_dist = 0

    for batch in dataloader:
        images = batch['image'].to(device)
        heatmaps = batch['heatmap'].to(device)
        gt_x = batch['fovea_x'].to(device)
        gt_y = batch['fovea_y'].to(device)
        gt_coords = torch.stack([gt_x, gt_y], dim=1)

        output = model(images)
        pred_heatmap = output['fovea_heatmap']

        if pred_heatmap.shape[-2:] != heatmaps.shape[-2:]:
            heatmaps = F.interpolate(heatmaps, size=pred_heatmap.shape[-2:], mode='bilinear', align_corners=False)

        loss = criterion(pred_heatmap, heatmaps)

        # 计算坐标距离
        pred_coords = model.get_fovea_coordinates(pred_heatmap, method='argmax')
        pred_coords = pred_coords.float()
        dist = euclidean_distance(pred_coords, gt_coords)

        total_loss += loss.item()
        total_dist += dist.item()

    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'distance': total_dist / n_batches
    }


def visualize_predictions(model, dataloader, device, save_path, num_samples=4):
    """可视化预测结果"""
    import matplotlib.pyplot as plt

    model.eval()

    batch = next(iter(dataloader))
    images = batch['image'][:num_samples].to(device)
    heatmaps_gt = batch['heatmap'][:num_samples]
    gt_x = batch['fovea_x'][:num_samples]
    gt_y = batch['fovea_y'][:num_samples]

    with torch.no_grad():
        output = model(images)
        pred_heatmap = output['fovea_heatmap']
        pred_heatmap_prob = torch.sigmoid(pred_heatmap)

        # 获取预测坐标
        pred_coords = model.get_fovea_coordinates(pred_heatmap, method='argmax')
        pred_coords = pred_coords.float()

    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_vis = images.cpu() * std + mean
    images_vis = images_vis.numpy().transpose(0, 2, 3, 1)
    images_vis = np.clip(images_vis, 0, 1)

    # Resize heatmaps if needed
    if pred_heatmap_prob.shape[-2:] != heatmaps_gt.shape[-2:]:
        pred_heatmap_prob = F.interpolate(pred_heatmap_prob, size=heatmaps_gt.shape[-2:], mode='bilinear', align_corners=False)

    fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))

    for i in range(num_samples):
        # Original image with GT point
        axes[i, 0].imshow(images_vis[i])
        axes[i, 0].scatter(gt_x[i].item(), gt_y[i].item(), c='green', s=100, marker='x', linewidths=3, label='GT')
        axes[i, 0].scatter(pred_coords[i, 0].cpu().item(), pred_coords[i, 1].cpu().item(),
                          c='red', s=100, marker='+', linewidths=3, label='Pred')
        axes[i, 0].legend()
        axes[i, 0].set_title('Image + Points')
        axes[i, 0].axis('off')

        # GT heatmap
        axes[i, 1].imshow(heatmaps_gt[i, 0].numpy(), cmap='hot')
        axes[i, 1].set_title('GT Heatmap')
        axes[i, 1].axis('off')

        # Predicted heatmap
        axes[i, 2].imshow(pred_heatmap_prob[i, 0].cpu().numpy(), cmap='hot')
        axes[i, 2].set_title('Pred Heatmap')
        axes[i, 2].axis('off')

        # Distance
        dist = np.sqrt((gt_x[i].item() - pred_coords[i, 0].cpu().item())**2 +
                       (gt_y[i].item() - pred_coords[i, 1].cpu().item())**2)
        axes[i, 3].imshow(images_vis[i])
        axes[i, 3].scatter(gt_x[i].item(), gt_y[i].item(), c='green', s=150, marker='x', linewidths=3)
        axes[i, 3].scatter(pred_coords[i, 0].cpu().item(), pred_coords[i, 1].cpu().item(),
                          c='red', s=150, marker='+', linewidths=3)
        axes[i, 3].set_title(f'Distance: {dist:.1f}px')
        axes[i, 3].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def freeze_encoder_and_od_decoder(model):
    """冻结 Encoder 和 OD Decoder"""
    # 冻结 Encoder
    for module in encoder_modules(model):
        for param in module.parameters():
            param.requires_grad = False

    # 冻结 OD Decoder
    for module in [model.od_decoder_stage4, model.od_decoder_stage3,
                   model.od_decoder_stage2, model.od_decoder_stage1,
                   model.od_final_upsample, model.od_head]:
        for param in module.parameters():
            param.requires_grad = False

    print("Frozen: Encoder + OD Decoder")
    print("Trainable: Fovea Decoder only")


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
        heatmap_sigma=args.heatmap_sigma
    )

    # 创建模型
    print("\nCreating model...")
    model = OpticUNet(encoder_pretrained=True, predict_fovea=True,
                      encoder_name=args.encoder, img_size=args.image_size)

    # 加载预训练的 OD 模型
    if args.od_checkpoint:
        print(f"\nLoading OD checkpoint from: {args.od_checkpoint}")
        checkpoint = torch.load(args.od_checkpoint, map_location='cpu', weights_only=False)

        # 加载 state_dict (只加载存在的 keys)
        model_dict = model.state_dict()
        pretrained_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))

        # 过滤掉不存在或形状不一致的 keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and v.shape == model_dict[k].shape}
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} parameters")
        n_od = sum(k.startswith('od_') for k in pretrained_dict)
        n_enc = sum(k.startswith('encoder_') for k in pretrained_dict)
        if n_od == 0 or n_enc == 0:
            raise RuntimeError(
                f"--od_checkpoint does not match the ResNet-50 OpticUNet "
                f"(encoder tensors loaded: {n_enc}, OD decoder tensors loaded: {n_od}). "
                "Train it with train_optic_disc_unet.py first.")

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model = model.to(device)

    # 冻结 Encoder 和 OD Decoder
    if args.freeze_encoder_od:
        freeze_encoder_and_od_decoder(model)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # 损失函数和优化器
    criterion = FoveaHeatmapLoss(bce_weight=1.0, mse_weight=1.0)

    # 只优化需要训练的参数
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params_list,
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # TensorBoard
    writer = None
    if HAS_TENSORBOARD and args.log_dir:
        log_dir = os.path.join(args.log_dir, 'fovea')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    # 训练循环
    print("\n" + "=" * 60)
    print("Starting Fovea training...")
    print("=" * 60)

    best_dist = float('inf')

    for epoch in range(1, args.epochs + 1):
        # 训练
        train_metrics = train_epoch(model, dataloaders['train'], optimizer, criterion, device, epoch)

        # 评估
        val_metrics = evaluate(model, dataloaders['val'], criterion, device)

        scheduler.step()

        # 打印结果
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Distance: {train_metrics['distance']:.1f}px")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Distance: {val_metrics['distance']:.1f}px")

        # TensorBoard 日志
        if writer:
            writer.add_scalars('Loss', {'train': train_metrics['loss'], 'val': val_metrics['loss']}, epoch)
            writer.add_scalars('Distance', {'train': train_metrics['distance'], 'val': val_metrics['distance']}, epoch)
            writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)

        # 保存最佳模型
        if val_metrics['distance'] < best_dist:
            best_dist = val_metrics['distance']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_distance': best_dist,
                'val_loss': val_metrics['loss']
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  ✓ Saved best model (Distance: {best_dist:.1f}px)")

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
    print(f"Best validation distance: {best_dist:.1f}px")
    print(f"Model saved to: {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Fovea Localization')

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='Dataset folder in the fixed format (see README, Data): images/<split>/ and fovea_<split>.csv')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Input image size')
    parser.add_argument('--heatmap_sigma', type=float, default=15.0,
                       help='Gaussian sigma for heatmap')

    # 模型参数
    parser.add_argument('--od_checkpoint', type=str, default=None,
                       help='Path to pretrained OD model checkpoint')
    parser.add_argument('--encoder', type=str, default='resnet50',
                       choices=['resnet50'],
                       help='Encoder backbone (ResNet-50)')
    parser.add_argument('--freeze_encoder_od', action='store_true', default=True,
                       help='Freeze encoder and OD decoder')
    parser.add_argument('--no_freeze', dest='freeze_encoder_od', action='store_false',
                       help='Do not freeze encoder and OD decoder')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')

    # 保存参数
    parser.add_argument('--save_dir', type=str, default='./checkpoints/fovea',
                       help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./runs',
                       help='TensorBoard log directory')

    args = parser.parse_args()
    main(args)
