#!/usr/bin/env python3
"""
CABNet Baseline 训练脚本

使用 CABNet (ResNet-50 + CBAM + Category Attention Block) 做 DR 分级 baseline
- 支持 3 类 (Healthy/NPDR/PDR) 或 5 类 (0-4 级) 分类
- 使用 Quadratic Weighted Kappa 作为主要评估指标
- 复用 grading_dataloader 和 train_paragraph 的工具函数

Reference:
    He et al., "CABNet: Category Attention Block for Imbalanced Diabetic Retinopathy Grading"
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import random
import argparse
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from Models import CABNetClassifier
from grading_dataloader import (
    create_grading_dataloaders,
    get_train_transform,
    get_val_transform,
    print_dataset_statistics,
    STAGE_NAMES,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed, logging disabled")


# ============ Focal Loss ============
def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: torch.Tensor = None,
    gamma: float = 2.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    """
    Multi-class Focal Loss

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    num_classes = logits.size(1)
    probs = F.softmax(logits, dim=1)
    targets_one_hot = F.one_hot(targets, num_classes).float()
    p_t = (probs * targets_one_hot).sum(dim=1)
    focal_weight = (1 - p_t) ** gamma
    ce_loss = F.cross_entropy(logits, targets, reduction='none')
    loss = focal_weight * ce_loss

    if alpha is not None:
        alpha_t = alpha[targets]
        loss = alpha_t * loss

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    return loss


# ============ Metrics ============
def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Quadratic Weighted Kappa (QWK) — DR 分级标准指标"""
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1

    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    hist_true = np.sum(confusion, axis=1)
    hist_pred = np.sum(confusion, axis=0)
    n = np.sum(confusion)
    expected = np.outer(hist_true, hist_pred) / n

    numerator = np.sum(weights * confusion)
    denominator = np.sum(weights * expected)
    if denominator == 0:
        return 0.0
    return 1.0 - numerator / denominator


def compute_class_weights(dataset, num_classes: int, device: torch.device) -> torch.Tensor:
    """计算类别权重 (逆频率) — 直接从 dataset 统计，避免 oversample 后分布失真"""
    print("\nComputing class weights...")
    class_counts = torch.zeros(num_classes)
    for i in tqdm(range(len(dataset)), desc="Counting classes"):
        sample = dataset[i]
        label = sample['label'].item() if isinstance(sample['label'], torch.Tensor) else sample['label']
        class_counts[label] += 1

    total = class_counts.sum()
    weights = total / (num_classes * class_counts.clamp(min=1))
    weights = weights / weights.mean()

    print(f"Class counts: {class_counts.tolist()}")
    print(f"Class weights: {weights.tolist()}")
    return weights.to(device)


# ============ Training ============
def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    class_weights: torch.Tensor = None,
    scheduler=None,
    scheduler_step: str = 'epoch',
    accumulation_steps: int = 1,
    use_focal: bool = False,
    focal_gamma: float = 2.0
) -> dict:
    model.train()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    num_batches = len(dataloader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        output = model(images)
        logits = output['logits']

        if use_focal:
            loss_val = focal_loss(logits, labels, alpha=class_weights, gamma=focal_gamma)
        else:
            if class_weights is not None:
                loss_val = F.cross_entropy(logits, labels, weight=class_weights)
            else:
                loss_val = F.cross_entropy(logits, labels)

        loss = loss_val / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and scheduler_step == 'batch':
                scheduler.step()

        predictions = logits.argmax(dim=-1)
        correct = (predictions == labels).sum().item()
        total_loss += loss_val.item() * images.size(0)
        total_correct += correct
        total_samples += images.size(0)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # 简化输出：每 25% 进度打印一次
        progress_pct = (batch_idx + 1) / num_batches
        if (batch_idx + 1) % max(1, num_batches // 4) == 0 or batch_idx == num_batches - 1:
            avg_loss_so_far = total_loss / total_samples
            acc_so_far = total_correct / total_samples
            print(f"  [{int(progress_pct*100):3d}%] loss={avg_loss_so_far:.4f}, acc={acc_so_far:.4f}")

    if total_samples % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa = quadratic_weighted_kappa(np.array(all_labels), np.array(all_preds), model.num_classes)

    return {'loss': avg_loss, 'accuracy': accuracy, 'kappa': kappa}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0
) -> dict:
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    num_classes = model.num_classes
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        output = model(images)
        logits = output['logits']

        if use_focal:
            loss_val = focal_loss(logits, labels, alpha=class_weights, gamma=focal_gamma)
        else:
            if class_weights is not None:
                loss_val = F.cross_entropy(logits, labels, weight=class_weights)
            else:
                loss_val = F.cross_entropy(logits, labels)

        predictions = logits.argmax(dim=-1)
        total_loss += loss_val.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for pred, label in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa = quadratic_weighted_kappa(np.array(all_labels), np.array(all_preds), num_classes)

    class_accuracy = {}
    class_names = STAGE_NAMES if num_classes == 3 else [f'Grade_{i}' for i in range(num_classes)]
    for i, name in enumerate(class_names):
        class_accuracy[name] = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'kappa': kappa,
        'class_accuracy': class_accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels)
    }


def build_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(labels, predictions):
        cm[int(t), int(p)] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, class_names: list):
    print("\nConfusion Matrix:")
    print("-" * 60)
    header = "True\\Pred"
    for name in class_names:
        header += f"\t{name[:8]}"
    print(header)
    for i, name in enumerate(class_names):
        row = f"{name[:8]}"
        for j in range(len(class_names)):
            row += f"\t{cm[i, j]}"
        print(row)


# ============ Seed ============
def set_seed(seed: int):
    """固定所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============ Main ============
def main(args):
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 80)

    # Save directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'cabnet_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    num_classes = 3 if args.use_3_stages else 5
    print(f"\nTraining CABNet ({args.backbone_name}) with {num_classes} classes")

    # ============ Data ============
    print("\nLoading datasets...")

    dataloaders = create_grading_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3_stages=args.use_3_stages,
        use_oversample=args.use_oversample,
        use_strong_aug=args.use_strong_aug,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print_dataset_statistics(dataloaders, use_3_stages=args.use_3_stages)

    # ============ Model ============
    print(f"\nCreating CABNet model (backbone={args.backbone_name})...")
    model = CABNetClassifier(
        num_classes=num_classes,
        backbone_name=args.backbone_name,
        pretrained=True,
        drop_rate=args.drop_rate,
        ca_k=args.ca_k,
    ).to(device)

    # Freeze backbone if requested
    if args.freeze_backbone:
        print("Freezing backbone weights (only training CABNet head)")
        for param in model.backbone.parameters():
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============ Optimizer ============
    head_params = list(model.classifier.parameters())

    if args.freeze_backbone:
        param_groups = [{'params': head_params, 'lr': args.lr}]
    else:
        backbone_params = list(model.backbone.parameters())
        param_groups = [
            {'params': backbone_params, 'lr': args.backbone_lr},
            {'params': head_params, 'lr': args.lr},
        ]
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    # Scheduler
    if args.scheduler == 'onecycle':
        max_lr = [args.lr] if args.freeze_backbone else [args.backbone_lr, args.lr]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=args.epochs,
            steps_per_epoch=len(dataloaders['train']),
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000
        )
        scheduler_step = 'batch'
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scheduler_step = 'epoch'

    # Class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(dataloaders['train'].dataset, num_classes, device)

    # TensorBoard
    writer = None
    if HAS_TENSORBOARD:
        log_dir = os.path.join(save_dir, 'tensorboard')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)

    # ============ Training Loop ============
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    best_kappa = -1.0
    best_epoch = 0
    history = {'train': [], 'val': []}

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=dataloaders['train'],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            class_weights=class_weights,
            scheduler=scheduler if scheduler_step == 'batch' else None,
            scheduler_step=scheduler_step,
            accumulation_steps=args.accumulation_steps,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma
        )

        # Validate
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders['test'],
            device=device,
            class_weights=class_weights,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma
        )

        if scheduler_step == 'epoch':
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Print
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Kappa: {train_metrics['kappa']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Kappa: {val_metrics['kappa']:.4f}")
        if args.freeze_backbone:
            print(f"LR (head): {current_lr:.6f}")
        else:
            print(f"LR (backbone): {current_lr:.6f}, LR (head): {optimizer.param_groups[1]['lr']:.6f}")

        print("Val per-class accuracy:")
        for name, acc in val_metrics['class_accuracy'].items():
            print(f"  {name}: {acc:.4f}")

        # TensorBoard
        if writer:
            writer.add_scalars('Loss', {'train': train_metrics['loss'], 'val': val_metrics['loss']}, epoch)
            writer.add_scalars('Accuracy', {'train': train_metrics['accuracy'], 'val': val_metrics['accuracy']}, epoch)
            writer.add_scalars('Kappa', {'train': train_metrics['kappa'], 'val': val_metrics['kappa']}, epoch)
            if args.freeze_backbone:
                writer.add_scalar('LR/head', optimizer.param_groups[0]['lr'], epoch)
            else:
                writer.add_scalar('LR/backbone', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('LR/head', optimizer.param_groups[1]['lr'], epoch)

        # Save best
        if val_metrics['kappa'] > best_kappa:
            best_kappa = val_metrics['kappa']
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_kappa': best_kappa,
                'val_accuracy': val_metrics['accuracy'],
                'history': history,
                'args': vars(args)
            }, os.path.join(save_dir, 'best_model.pth'))
            print(f"  -> Saved best model (Kappa: {best_kappa:.4f})")

        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_kappa': best_kappa,
                'history': history,
                'args': vars(args)
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

    # ============ Final Evaluation ============
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'test' in dataloaders:
        test_metrics = evaluate(
            model=model,
            dataloader=dataloaders['test'],
            device=device,
            class_weights=class_weights,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma
        )

        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Kappa: {test_metrics['kappa']:.4f}")

        print("\nTest per-class accuracy:")
        for name, acc in test_metrics['class_accuracy'].items():
            print(f"  {name}: {acc:.4f}")

        cm = build_confusion_matrix(test_metrics['labels'], test_metrics['predictions'], num_classes)
        class_names = STAGE_NAMES if num_classes == 3 else [f'Grade_{i}' for i in range(num_classes)]
        print_confusion_matrix(cm, class_names)

        with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
            json.dump({
                'loss': test_metrics['loss'],
                'accuracy': test_metrics['accuracy'],
                'kappa': test_metrics['kappa'],
                'class_accuracy': test_metrics['class_accuracy'],
                'confusion_matrix': cm.tolist()
            }, f, indent=4)

    # Save history
    with open(os.path.join(save_dir, 'history.json'), 'w') as f:
        history_clean = {
            'train': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v
                 for k, v in m.items() if k != 'class_accuracy'}
                for m in history['train']
            ],
            'val': [
                {k: float(v) if isinstance(v, (np.floating, float)) else v
                 for k, v in m.items() if k not in ['class_accuracy', 'predictions', 'labels']}
                for m in history['val']
            ]
        }
        json.dump(history_clean, f, indent=4)

    if writer:
        writer.close()

    print(f"\n" + "=" * 80)
    print(f"Training completed!")
    print(f"Best validation Kappa: {best_kappa:.4f} (epoch {best_epoch})")
    print(f"Model saved to: {save_dir}")
    print("=" * 80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CABNet Baseline for DR Grading')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Seed of the stratified re-split (default: use the CSV files as they are)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation share when --split_seed re-splits train.csv')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Share held out as the test set when the folder has no test.csv')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Input image size')
    parser.add_argument('--use_3_stages', action='store_true', default=False,
                       help='Use 3-class staging (Healthy/NPDR/PDR)')
    parser.add_argument('--use_5_stages', dest='use_3_stages', action='store_false',
                       help='Use 5-class grading (0-4), this is the default')

    # Model
    parser.add_argument('--backbone_name', type=str, default='resnet50',
                       choices=['resnet18', 'resnet50', 'resnet101'],
                       help='ResNet backbone for CABNet')
    parser.add_argument('--drop_rate', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--ca_k', type=int, default=5,
                       help='Number of sub-channels per class in CABlock')
    parser.add_argument('--freeze_backbone', action='store_true', default=False,
                       help='Freeze ResNet backbone weights (only train CABNet head)')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for CABNet head')
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                       help='Learning rate for ResNet backbone')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['onecycle', 'cosine'])
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # Loss
    parser.add_argument('--use_class_weights', action='store_true', default=False)
    parser.add_argument('--use_focal', action='store_true', default=True)
    parser.add_argument('--no_focal', dest='use_focal', action='store_false')
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # Oversampling
    parser.add_argument('--use_oversample', action='store_true', default=True)
    parser.add_argument('--no_oversample', dest='use_oversample', action='store_false')
    parser.add_argument('--use_strong_aug', action='store_true', default=False)
    parser.add_argument('--no_strong_aug', dest='use_strong_aug', action='store_false')

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./baseline/cabnet_baseline')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("CABNet Baseline Training Configuration")
    print("=" * 80)
    print(json.dumps(vars(args), indent=2))
    print()

    main(args)
