#!/usr/bin/env python3
"""
改进的UNet训练脚本 - 解决类别不平衡和训练困难问题

主要改进:
1. 使用focal loss或weighted BCE处理极度不平衡
2. 更合理的学习率schedule
3. 更好的初始化策略
4. 在线难例挖掘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter
import os
import argparse
import json
import datetime
import math
from tqdm import tqdm
import numpy as np
from pathlib import Path

from torch.cuda.amp import GradScaler, autocast

from Models import LesionUNet
from Models.hacdr_lesion_unet import HACDRLesionUNet
from segmentation_dataloader import create_lesion_dataloaders, visualize_sample


def build_param_groups(model, base_lr, weight_decay, head_lr_mult=15.0):
    """Split params into backbone / head, with norm layers getting weight_decay=0."""
    HEAD_MODULES = ('decoder_stage', 'segmentation_head', 'final_upsample',
                    'stem_proj', 'stage1_proj', 'stage2_proj',
                    'stage3_proj', 'stage4_proj', 'bottleneck_proj')
    NORM_KEYWORDS = ('norm', 'bn', 'layer_norm')

    backbone, backbone_nd, head, head_nd = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head = any(name.startswith(h) or f'.{h}' in name for h in HEAD_MODULES)
        is_norm = any(k in name.lower() for k in NORM_KEYWORDS)
        if is_head and is_norm:
            head_nd.append(param)
        elif is_head:
            head.append(param)
        elif is_norm:
            backbone_nd.append(param)
        else:
            backbone.append(param)

    return [
        {'params': backbone,    'lr': base_lr,                'weight_decay': weight_decay},
        {'params': backbone_nd, 'lr': base_lr,                'weight_decay': 0.0},
        {'params': head,        'lr': base_lr * head_lr_mult, 'weight_decay': weight_decay},
        {'params': head_nd,     'lr': base_lr * head_lr_mult, 'weight_decay': 0.0},
    ]


def build_poly_scheduler(optimizer, max_iters, warmup_iters=1500, power=1.0):
    """Poly LR with linear warmup, stepped every iteration."""
    def lr_lambda(it):
        if it < warmup_iters:
            return max(1e-6, it / warmup_iters)
        progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
        return max(0.0, (1.0 - progress) ** power)
    return LambdaLR(optimizer, lr_lambda)


def compute_class_weights(dataloader, device, num_samples=50):
    """计算类别权重以处理不平衡"""
    print("\n计算类别权重...")

    total_pixels = 0
    lesion_pixels = torch.zeros(4)

    for batch in dataloader:
        
        masks = batch['mask']  # [B, 4, H, W]
        B, C, H, W = masks.shape

        total_pixels += B * H * W
        lesion_pixels += masks.sum(dim=(0, 2, 3))

    # 计算pos_weight (负样本数 / 正样本数)
    pos_weights = torch.zeros(4)
    for i in range(4):
        if lesion_pixels[i] > 0:
            neg_pixels = total_pixels - lesion_pixels[i]
            pos_weights[i] = neg_pixels / lesion_pixels[i]
            # 限制最大权重避免过大
            pos_weights[i] = min(pos_weights[i], 50.0)
        else:
            pos_weights[i] = 1.0

    print(f"类别权重: {pos_weights.tolist()}")
    return pos_weights.to(device)


def focal_loss(
    pred_logits,
    target,
    alpha=0.25,          # float or tensor: [C] / [1,C,1,1]
    gamma=2.0,
    pos_weight=None,     # None or tensor: [C] / [1,C,1,1]
    reduction='mean',
    eps=1e-8
):
    """
    Binary Focal Loss (with logits) for multi-label segmentation.

    pred_logits: [B, C, H, W]
    target:      [B, C, H, W] in {0,1} (float recommended)

    alpha can be:
      - float (scalar)
      - tensor broadcastable to target, e.g. [C] or [1,C,1,1]

    pos_weight (optional) uses PyTorch BCEWithLogits pos_weight:
      - tensor broadcastable to [B,C,H,W], commonly [C] or [1,C,1,1]
    """
    target = target.float()

    # ensure alpha broadcastable
    if not torch.is_tensor(alpha):
        alpha_t = alpha
    else:
        alpha_t = alpha.to(pred_logits.device)

    # ensure pos_weight broadcastable
    if pos_weight is not None:
        if not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, device=pred_logits.device, dtype=pred_logits.dtype)
        else:
            pos_weight = pos_weight.to(pred_logits.device)
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target, reduction='none', pos_weight=pos_weight
        )
    else:
        bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='none')

    prob = torch.sigmoid(pred_logits)
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    focal_weight = (1.0 - p_t).clamp(min=0.0, max=1.0) ** gamma

    # alpha balancing: alpha for positives, (1-alpha) for negatives
    if torch.is_tensor(alpha_t):
        alpha_factor = alpha_t * target + (1.0 - alpha_t) * (1.0 - target)
    else:
        alpha_factor = alpha * target + (1.0 - alpha) * (1.0 - target)

    loss = alpha_factor * focal_weight * bce_loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

def tversky_loss(pred_probs, target, alpha=0.7, beta=0.3, smooth=1.0):
    """
    Tversky Loss for binary segmentation maps [B, C, H, W].
    alpha: FP penalty weight  (higher → conservative / high precision)
    beta:  FN penalty weight  (lower  → tolerate missed detections)
    Loss = 1 - TP / (TP + alpha*FP + beta*FN)
    """
    TP = (pred_probs * target).sum(dim=(2, 3))
    FP = (pred_probs * (1 - target)).sum(dim=(2, 3))
    FN = ((1 - pred_probs) * target).sum(dim=(2, 3))
    tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
    return (1 - tversky).mean()


def compute_loss(
    model,
    pred_masks,   # [B,4,H,W] logits
    gt_masks,     # [B,4,H,W] {0,1}
    pos_weights=None,   # [4]
    use_focal=True,
    focal_gamma=2.0,
    bce_weight=1.0,
    dice_weight=1.0,
    use_tversky=False,
    tversky_alpha=0.7,
    tversky_beta=0.3,
):
    gt_masks = gt_masks.float()

    if use_focal:
        # per-class alpha：用一个“温和版”权重，避免你那种全 50 的爆炸
        # 经验：alpha 越大越偏向正类；这里用 0.25 为基准，按类稍微拉开
        if pos_weights is not None:
            # 把 pos_weight 映射到 [0.25, 0.75] 之类的范围（可调）
            w = pos_weights.float()
            w = w / (w.max() + 1e-12)
            alpha = 0.25 + 0.5 * w              # [4] in [0.25,0.75]
            alpha = alpha.view(1, -1, 1, 1)      # broadcast
        else:
            alpha = 0.25

        bce_loss = focal_loss(pred_masks, gt_masks, alpha=alpha, gamma=focal_gamma)
    else:
        if pos_weights is not None:
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_masks, gt_masks,
                pos_weight=pos_weights.view(1, -1, 1, 1),
                reduction='mean'
            )
        else:
            bce_loss = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction='mean')

    pred_probs = torch.sigmoid(pred_masks)
    if use_tversky:
        dice_loss = tversky_loss(pred_probs, gt_masks.float(),
                                 alpha=tversky_alpha, beta=tversky_beta)
    else:
        dice_loss = model._dice_loss(pred_probs, gt_masks)

    total_loss = bce_weight * bce_loss + dice_weight * dice_loss

    return {'total': total_loss, 'bce': bce_loss, 'dice': dice_loss}



def compute_metrics(pred_masks, gt_masks, threshold=0.5, eps=1e-6):
    """计算IoU和Dice指标"""
    pred = (torch.sigmoid(pred_masks) > threshold).float()
    gt = gt_masks.float()

    B, C, H, W = pred.shape
    lesion_names = ['MA', 'HE', 'EX', 'SE']

    metrics = {}
    per_class_iou = []
    per_class_dice = []

    for c, name in enumerate(lesion_names):
        p = pred[:, c]
        g = gt[:, c]

        inter = (p * g).sum(dim=(1, 2))
        p_sum = p.sum(dim=(1, 2))
        g_sum = g.sum(dim=(1, 2))
        union = p_sum + g_sum - inter

        valid = (g_sum > 0)

        if valid.any():
            iou = (inter[valid] + eps) / (union[valid] + eps)
            dice = (2 * inter[valid] + eps) / (p_sum[valid] + g_sum[valid] + eps)
            metrics[f'{name}_IoU'] = iou.mean().item()
            metrics[f'{name}_Dice'] = dice.mean().item()
            per_class_iou.append(metrics[f'{name}_IoU'])
            per_class_dice.append(metrics[f'{name}_Dice'])
        else:
            metrics[f'{name}_IoU'] = float('nan')
            metrics[f'{name}_Dice'] = float('nan')

    iou_vals = [v for v in per_class_iou if v == v]
    dice_vals = [v for v in per_class_dice if v == v]

    metrics['mean_IoU'] = float(sum(iou_vals) / max(len(iou_vals), 1))
    metrics['mean_Dice'] = float(sum(dice_vals) / max(len(dice_vals), 1))
    return metrics


def train_epoch(model, dataloader, optimizer, device, epoch, writer, pos_weights, use_focal, save_dir=None, scheduler=None, scaler=None, accum_steps=1, use_tversky=False, tversky_alpha=0.7, tversky_beta=0.3):
    """训练一个epoch"""
    model.train()

    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_metrics = {'mean_IoU': 0, 'mean_Dice': 0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        gt_masks = batch['mask'].to(device)

        with autocast(enabled=scaler is not None):
            output = model(images, mode='full')
            pred_masks = output['masks']
            loss_dict = compute_loss(model, pred_masks, gt_masks, pos_weights, use_focal,
                                     use_tversky=use_tversky, tversky_alpha=tversky_alpha,
                                     tversky_beta=tversky_beta)
            loss = loss_dict['total'] / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(dataloader):
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

            # OneCycleLR在每次参数更新后step
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        total_bce += loss_dict['bce'].item()
        total_dice += loss_dict['dice'].item()

        metrics = compute_metrics(pred_masks, gt_masks)
        total_metrics['mean_IoU'] += metrics['mean_IoU']
        total_metrics['mean_Dice'] += metrics['mean_Dice']

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'IoU': f"{metrics['mean_IoU']:.4f}"
        })

        # 可视化
        if save_dir and batch_idx < 3:
            for i in range(min(2, images.size(0))):
                vis_path = os.path.join(save_dir, f'epoch{epoch}_batch{batch_idx}_sample{i}.png')
                pred_binary = (torch.sigmoid(pred_masks[i]) > 0.5).float()
                visualize_sample(
                    image=images[i],
                    mask=gt_masks[i],
                    pred_mask=pred_binary,
                    image_id=batch['image_id'][i],
                    save_path=vis_path
                )

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'bce': total_bce / n,
        'dice_loss': total_dice / n,
        'IoU': total_metrics['mean_IoU'] / n,
        'Dice': total_metrics['mean_Dice'] / n
    }


@torch.no_grad()
def validate(model, dataloader, device, epoch, writer, pos_weights, use_focal, save_dir=None, use_tversky=False, tversky_alpha=0.7, tversky_beta=0.3):
    """验证"""
    model.eval()

    total_loss = 0
    total_bce = 0
    total_dice = 0
    total_metrics = {'mean_IoU': 0, 'mean_Dice': 0}

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')

    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        gt_masks = batch['mask'].to(device)

        output = model(images, mode='full')
        pred_masks = output['masks']

        loss_dict = compute_loss(model, pred_masks, gt_masks, pos_weights, use_focal,
                                 use_tversky=use_tversky, tversky_alpha=tversky_alpha,
                                 tversky_beta=tversky_beta)
        loss = loss_dict['total']

        total_loss += loss.item()
        total_bce += loss_dict['bce'].item()
        total_dice += loss_dict['dice'].item()

        metrics = compute_metrics(pred_masks, gt_masks)
        total_metrics['mean_IoU'] += metrics['mean_IoU']
        total_metrics['mean_Dice'] += metrics['mean_Dice']

        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'IoU': f"{metrics['mean_IoU']:.4f}"
        })

        # 可视化前几个batch
        if save_dir and batch_idx < 3:
            for i in range(min(2, images.size(0))):
                vis_path = os.path.join(save_dir, f'epoch{epoch}_batch{batch_idx}_sample{i}.png')
                pred_binary = (torch.sigmoid(pred_masks[i]) > 0.5).float()
                visualize_sample(
                    image=images[i],
                    mask=gt_masks[i],
                    pred_mask=pred_binary,
                    image_id=batch['image_id'][i],
                    save_path=vis_path
                )

    n = len(dataloader)
    return {
        'loss': total_loss / n,
        'bce': total_bce / n,
        'dice_loss': total_dice / n,
        'IoU': total_metrics['mean_IoU'] / n,
        'Dice': total_metrics['mean_Dice'] / n
    }


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 80)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'visualizations'), exist_ok=True)

    # 保存参数
    with open(os.path.join(args.save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # 加载数据
    print("\nLoading datasets...")

    dataloaders = create_lesion_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    print(f"Train samples: {len(dataloaders['train'].dataset)}")
    if 'val' in dataloaders:
        print(f"Val samples: {len(dataloaders['val'].dataset)}")

    # 计算类别权重 / 使用固定 pos_weights
    pos_weights = None
    if args.use_class_weights:
        pos_weights = compute_class_weights(dataloaders['train'], device)
    elif args.model == 'hacdr':
        # HACDR-Net paper recommended class weights [MA, HE, EX, SE]
        pos_weights = torch.tensor([1.4, 3.0, 1.4, 2.2], device=device)
        print(f"Using HACDR paper pos_weights: {pos_weights.tolist()}")

    # 创建模型
    print(f"\nCreating model ({args.model})...")
    if args.model == 'hacdr':
        model = HACDRLesionUNet(num_lesions=4).to(device)
    else:
        model = LesionUNet(
            num_lesions=4,
            encoder_pretrained=args.pretrained,
            encoder_freeze=False,
            encoder_name=args.encoder,
            img_size=args.image_size
        ).to(device)

    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Resumed weights from: {args.resume}")

    print(f"Total parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # 优化器
    steps_per_epoch = len(dataloaders['train'])
    if args.model == 'hacdr' or args.scheduler == 'poly':
        param_groups = build_param_groups(model, args.lr, args.weight_decay, head_lr_mult=15.0)
        optimizer = AdamW(param_groups, betas=(0.9, 0.999))
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 学习率调度器
    if args.scheduler == 'poly':
        epochs = math.ceil(args.max_iters / steps_per_epoch)
        scheduler = build_poly_scheduler(
            optimizer, max_iters=args.max_iters, warmup_iters=args.warmup_iters
        )
        scheduler_step = 'batch'
        print(f"Poly LR schedule: {args.max_iters} iters ({epochs} epochs), warmup={args.warmup_iters}")
    elif args.scheduler == 'onecycle':
        epochs = args.epochs
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000
        )
        scheduler_step = 'batch'
    else:
        epochs = args.epochs
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        scheduler_step = 'epoch'

    # AMP scaler
    scaler = GradScaler() if args.use_amp else None
    if args.use_amp:
        print("AMP enabled (mixed precision)")

    # TensorBoard
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.save_dir, 'tensorboard', f'unet_{timestamp}')
    writer = SummaryWriter(log_dir)

    # 训练循环
    best_iou = 0
    history = {'train': [], 'val': []}

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 80)

        # 训练
        train_scheduler = scheduler if scheduler_step == 'batch' else None
        train_metrics = train_epoch(
            model, dataloaders['train'], optimizer,
            device, epoch, writer, pos_weights, args.use_focal,
            os.path.join(args.save_dir, 'visualizations'),
            scheduler=train_scheduler,
            scaler=scaler,
            accum_steps=args.accum_steps,
            use_tversky=args.use_tversky,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
        )

        # 验证
        if 'val' in dataloaders:
            val_metrics = validate(
                model, dataloaders['val'], device, epoch, writer,
                pos_weights, args.use_focal,
                os.path.join(args.save_dir, 'visualizations'),
                use_tversky=args.use_tversky,
                tversky_alpha=args.tversky_alpha,
                tversky_beta=args.tversky_beta,
            )
        else:
            val_metrics = train_metrics

        # 学习率调度
        if scheduler_step == 'epoch':
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # 记录
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['IoU']:.4f}, Dice: {train_metrics['Dice']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['IoU']:.4f}, Dice: {val_metrics['Dice']:.4f}")
        print(f"LR: {current_lr:.6f}")

        # TensorBoard
        writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        writer.add_scalars('IoU', {
            'train': train_metrics['IoU'],
            'val': val_metrics['IoU']
        }, epoch)
        writer.add_scalar('LR', current_lr, epoch)

        # 保存最佳模型
        if val_metrics['IoU'] > best_iou:
            best_iou = val_metrics['IoU']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'history': history,
                'log_dir': log_dir,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"✓ 保存最佳模型 (IoU: {best_iou:.4f})")

        # 定期保存checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_iou': best_iou,
                'history': history,
                'log_dir': log_dir,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pth'))

        # OneCycleLR在每个batch后step
        if scheduler_step == 'batch':
            pass  # 已在train_epoch中处理

    # 保存训练历史
    with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)

    writer.close()
    print(f"\nTraining completed! Best IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {args.save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # 数据参数
    parser.add_argument('--data_root', type=str, required=True,
                       help='Dataset folder in the fixed format (see README, Data): images/<split>/ and masks/<split>/{MA,HE,EX,SE}/')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Input image size')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=100,
                       help='Max epochs (overridden by --max_iters when using poly scheduler)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (1024分辨率建议4, 512建议8)')
    parser.add_argument('--lr', type=float, default=6e-5,
                       help='Base LR (HACDR paper: 6e-5)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay (HACDR paper: 0.01)')
    parser.add_argument('--scheduler', type=str, default='poly',
                       choices=['onecycle', 'cosine', 'poly'],
                       help='LR scheduler (poly: HACDR paper setting)')
    parser.add_argument('--max_iters', type=int, default=40000,
                       help='Total training iterations for poly scheduler (HACDR paper: 40000)')
    parser.add_argument('--warmup_iters', type=int, default=1500,
                       help='Warmup iterations for poly scheduler (HACDR paper: 1500)')

    # 模型参数
    parser.add_argument('--model', type=str, default='lesion_unet',
                       choices=['lesion_unet', 'hacdr'],
                       help='Model architecture: lesion_unet or hacdr')
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--encoder', type=str, default='swin',
                       choices=['resnet50', 'swin'],
                       help='Encoder backbone (lesion_unet only): resnet50 or swin')

    # 损失函数参数
    parser.add_argument('--use_focal', action='store_true', default=True,
                       help='Use focal loss instead of BCE')
    parser.add_argument('--use_class_weights', action='store_true', default=False,
                       help='Use class weights for imbalanced data')
    parser.add_argument('--use_tversky', action='store_true', default=False,
                       help='Replace Dice with Tversky loss (conservative: penalize FP more than FN)')
    parser.add_argument('--tversky_alpha', type=float, default=0.7,
                       help='Tversky FP penalty weight (default 0.7, higher → more conservative)')
    parser.add_argument('--tversky_beta', type=float, default=0.3,
                       help='Tversky FN penalty weight (default 0.3, lower → tolerate missed detections)')

    # 其他
    parser.add_argument('--use_amp', action='store_true', default=False,
                       help='Enable automatic mixed precision (FP16) for faster training')
    parser.add_argument('--accum_steps', type=int, default=1,
                       help='Gradient accumulation steps (effective_batch = batch_size * accum_steps)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--save_dir', type=str, default='./checkpoints/lesion_unet')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume/finetune from')

    args = parser.parse_args()

    print("\nTraining Configuration:")
    print(json.dumps(vars(args), indent=2))
    print()

    main(args)
