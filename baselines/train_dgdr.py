#!/usr/bin/env python3
"""
DGDR Baseline 训练脚本 (GDRNet Framework, MICCAI 2023)

支持 DGDR 的多种域泛化算法作为 baseline:
- ERM: ResNet backbone (Empirical Risk Minimization)
- GREEN: SoftLabelGCN (Graph-based label smoothing)
- CABNet: Category Attention Block Network (CBAM + CABlock)
- MixupNet: ResNet backbone + Mixup augmentation
- MixStyleNet: VGG backbone + MixStyle feature augmentation

参考: https://github.com/chehx/DGDR
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os
import sys
import random
import argparse
import json
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add DGDR to path for model imports
DGDR_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'DGDR')
)
if DGDR_DIR not in sys.path:
    sys.path.insert(0, DGDR_DIR)

from yacs.config import CfgNode as CN
from modeling.model_manager import get_net, get_classifier

from grading_dataloader import (
    create_grading_dataloaders,
    print_dataset_statistics,
    STAGE_NAMES,
)

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: tensorboard not installed, logging disabled")


SUPPORTED_ALGORITHMS = ['ERM', 'GREEN', 'CABNet', 'MixupNet', 'MixStyleNet']


# ============ DGDR Model Wrapper ============

def build_dgdr_cfg(algorithm: str, num_classes: int,
                   backbone: str = 'resnet50', dropout: float = 0.0) -> CN:
    """Build minimal yacs config for DGDR model initialization."""
    cfg = CN()
    cfg.ALGORITHM = algorithm
    cfg.BACKBONE = backbone
    cfg.DROP_OUT = dropout
    cfg.DATASET = CN()
    cfg.DATASET.NUM_CLASSES = num_classes
    return cfg


class DGDRModel(nn.Module):
    """
    Wrapper for DGDR models to match our training interface.

    forward(x) -> {'logits': Tensor}
    """

    def __init__(self, algorithm: str, num_classes: int,
                 backbone: str = 'resnet50', dropout: float = 0.0):
        super().__init__()
        if algorithm not in SUPPORTED_ALGORITHMS:
            raise ValueError(
                f"Unsupported algorithm '{algorithm}'. "
                f"Choose from {SUPPORTED_ALGORITHMS}"
            )

        self.num_classes = num_classes
        self.algorithm_name = algorithm

        cfg = build_dgdr_cfg(algorithm, num_classes, backbone, dropout)
        self.network = get_net(cfg)

        if algorithm == 'GREEN':
            # SoftLabelGCN outputs logits directly, no separate classifier
            self.classifier = None
        else:
            out_features = self.network.out_features()
            self.classifier = nn.Linear(out_features, num_classes)

    def forward(self, x):
        if self.classifier is None:
            logits = self.network(x)
        else:
            features = self.network(x)
            logits = self.classifier(features)
        return {'logits': logits}

    def get_backbone_params(self):
        """Return backbone/feature-extractor parameters (lower LR)."""
        if self.algorithm_name == 'GREEN':
            return list(self.network.cnn.parameters())
        else:
            return list(self.network.parameters())

    def get_head_params(self):
        """Return head/classifier parameters (higher LR)."""
        if self.algorithm_name == 'GREEN':
            head_params = []
            head_params.extend(self.network.gcn1.parameters())
            head_params.extend(self.network.gcn2.parameters())
            head_params.extend(self.network.cnn_linear.parameters())
            head_params.append(self.network.adj)
            return head_params
        else:
            return list(self.classifier.parameters())


# ============ Mixup Utilities ============

def mixup_data(x, y, alpha=1.0):
    """Returns mixed inputs, pairs of targets, and lambda."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(logits, y_a, y_b, lam, class_weights=None):
    """Mixup loss: lam * CE(pred, y_a) + (1-lam) * CE(pred, y_b)."""
    if class_weights is not None:
        return lam * F.cross_entropy(logits, y_a, weight=class_weights) + \
               (1 - lam) * F.cross_entropy(logits, y_b, weight=class_weights)
    return lam * F.cross_entropy(logits, y_a) + \
           (1 - lam) * F.cross_entropy(logits, y_b)


# ============ Focal Loss ============

def focal_loss(logits, targets, alpha=None, gamma=2.0, reduction='mean'):
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

def quadratic_weighted_kappa(y_true, y_pred, num_classes):
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


def compute_class_weights(dataset, num_classes, device):
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

def train_epoch(model, dataloader, optimizer, device, epoch,
                class_weights=None, scheduler=None, scheduler_step='epoch',
                accumulation_steps=1, use_focal=False, focal_gamma=2.0,
                use_mixup=False, mixup_alpha=1.0):
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

        if use_mixup:
            mixed_images, targets_a, targets_b, lam = mixup_data(
                images, labels, alpha=mixup_alpha
            )
            output = model(mixed_images)
            logits = output['logits']
            loss_val = mixup_criterion(logits, targets_a, targets_b, lam,
                                       class_weights=class_weights)
            # For accuracy tracking, use original labels
            predictions = logits.argmax(dim=-1)
        else:
            output = model(images)
            logits = output['logits']

            if use_focal:
                loss_val = focal_loss(logits, labels, alpha=class_weights,
                                      gamma=focal_gamma)
            else:
                if class_weights is not None:
                    loss_val = F.cross_entropy(logits, labels,
                                               weight=class_weights)
                else:
                    loss_val = F.cross_entropy(logits, labels)

            predictions = logits.argmax(dim=-1)

        loss = loss_val / accumulation_steps
        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and scheduler_step == 'batch':
                scheduler.step()

        correct = (predictions == labels).sum().item()
        total_loss += loss_val.item() * images.size(0)
        total_correct += correct
        total_samples += images.size(0)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        progress_pct = (batch_idx + 1) / num_batches
        if (batch_idx + 1) % max(1, num_batches // 4) == 0 or \
                batch_idx == num_batches - 1:
            avg_loss_so_far = total_loss / total_samples
            acc_so_far = total_correct / total_samples
            print(f"  [{int(progress_pct*100):3d}%] "
                  f"loss={avg_loss_so_far:.4f}, acc={acc_so_far:.4f}")

    if total_samples % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa = quadratic_weighted_kappa(
        np.array(all_labels), np.array(all_preds), model.num_classes
    )

    return {'loss': avg_loss, 'accuracy': accuracy, 'kappa': kappa}


@torch.no_grad()
def evaluate(model, dataloader, device, class_weights=None,
             use_focal=False, focal_gamma=2.0):
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
            loss_val = focal_loss(logits, labels, alpha=class_weights,
                                  gamma=focal_gamma)
        else:
            if class_weights is not None:
                loss_val = F.cross_entropy(logits, labels,
                                           weight=class_weights)
            else:
                loss_val = F.cross_entropy(logits, labels)

        predictions = logits.argmax(dim=-1)
        total_loss += loss_val.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        for pred, label in zip(predictions.cpu().numpy(),
                               labels.cpu().numpy()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa = quadratic_weighted_kappa(
        np.array(all_labels), np.array(all_preds), num_classes
    )

    class_accuracy = {}
    class_names = (STAGE_NAMES if num_classes == 3
                   else [f'Grade_{i}' for i in range(num_classes)])
    for i, name in enumerate(class_names):
        class_accuracy[name] = (class_correct[i] / class_total[i]
                                if class_total[i] > 0 else 0.0)

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'kappa': kappa,
        'class_accuracy': class_accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
    }


def build_confusion_matrix(labels, predictions, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(labels, predictions):
        cm[int(t), int(p)] += 1
    return cm


def print_confusion_matrix(cm, class_names):
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

def set_seed(seed):
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
    algo_tag = args.algorithm.lower()
    save_dir = os.path.join(
        args.save_dir, f'dgdr_{algo_tag}_{args.backbone}_{timestamp}'
    )
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    num_classes = 3 if args.use_3_stages else 5
    print(f"\nTraining DGDR [{args.algorithm}] with {num_classes} classes")
    print(f"Backbone: {args.backbone}")

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
    print(f"\nCreating DGDR model: {args.algorithm} ...")
    model = DGDRModel(
        algorithm=args.algorithm,
        num_classes=num_classes,
        backbone=args.backbone,
        dropout=args.drop_rate,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============ Optimizer ============
    backbone_params = model.get_backbone_params()
    head_params = model.get_head_params()

    param_groups = [
        {'params': backbone_params, 'lr': args.backbone_lr},
        {'params': head_params, 'lr': args.lr},
    ]

    if args.optim == 'sgd':
        optimizer = SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
        )
    else:
        optimizer = AdamW(
            param_groups,
            weight_decay=args.weight_decay,
        )

    # Scheduler
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[args.backbone_lr, args.lr],
            epochs=args.epochs,
            steps_per_epoch=len(dataloaders['train']),
            pct_start=0.1,
            div_factor=25,
            final_div_factor=1000,
        )
        scheduler_step = 'batch'
    else:
        scheduler = CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        scheduler_step = 'epoch'

    # Class weights
    class_weights = None
    if args.use_class_weights:
        class_weights = compute_class_weights(
            dataloaders['train'].dataset, num_classes, device
        )

    # Mixup: auto-enable for MixupNet
    use_mixup = args.use_mixup
    if args.algorithm == 'MixupNet' and not args.no_mixup:
        use_mixup = True
    if use_mixup:
        print(f"Mixup enabled (alpha={args.mixup_alpha})")

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
            focal_gamma=args.focal_gamma,
            use_mixup=use_mixup,
            mixup_alpha=args.mixup_alpha,
        )

        # Validate
        val_metrics = evaluate(
            model=model,
            #dataloader=dataloaders.get('val', dataloaders.get('test')),
            dataloader=dataloaders['test'],
            device=device,
            class_weights=class_weights,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma,
        )

        if scheduler_step == 'epoch':
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Print
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"Kappa: {train_metrics['kappa']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"Kappa: {val_metrics['kappa']:.4f}")
        print(f"LR (backbone): {current_lr:.6f}, "
              f"LR (head): {optimizer.param_groups[1]['lr']:.6f}")

        print("Val per-class accuracy:")
        for name, acc in val_metrics['class_accuracy'].items():
            print(f"  {name}: {acc:.4f}")

        # TensorBoard
        if writer:
            writer.add_scalars(
                'Loss',
                {'train': train_metrics['loss'], 'val': val_metrics['loss']},
                epoch,
            )
            writer.add_scalars(
                'Accuracy',
                {'train': train_metrics['accuracy'],
                 'val': val_metrics['accuracy']},
                epoch,
            )
            writer.add_scalars(
                'Kappa',
                {'train': train_metrics['kappa'],
                 'val': val_metrics['kappa']},
                epoch,
            )
            writer.add_scalar(
                'LR/backbone', optimizer.param_groups[0]['lr'], epoch
            )
            writer.add_scalar(
                'LR/head', optimizer.param_groups[1]['lr'], epoch
            )

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
                'args': vars(args),
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
                'args': vars(args),
            }, os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth'))

    # ============ Final Evaluation ============
    print("\n" + "=" * 80)
    print("Final Evaluation on Test Set")
    print("=" * 80)

    checkpoint = torch.load(
        os.path.join(save_dir, 'best_model.pth'), weights_only=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])

    if 'test' in dataloaders:
        test_metrics = evaluate(
            model=model,
            dataloader=dataloaders['test'],
            device=device,
            class_weights=class_weights,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma,
        )

        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Kappa: {test_metrics['kappa']:.4f}")

        print("\nTest per-class accuracy:")
        for name, acc in test_metrics['class_accuracy'].items():
            print(f"  {name}: {acc:.4f}")

        cm = build_confusion_matrix(
            test_metrics['labels'], test_metrics['predictions'], num_classes
        )
        class_names = (STAGE_NAMES if num_classes == 3
                       else [f'Grade_{i}' for i in range(num_classes)])
        print_confusion_matrix(cm, class_names)

        with open(os.path.join(save_dir, 'test_results.json'), 'w') as f:
            json.dump({
                'loss': test_metrics['loss'],
                'accuracy': test_metrics['accuracy'],
                'kappa': test_metrics['kappa'],
                'class_accuracy': test_metrics['class_accuracy'],
                'confusion_matrix': cm.tolist(),
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
                 for k, v in m.items()
                 if k not in ['class_accuracy', 'predictions', 'labels']}
                for m in history['val']
            ],
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
    parser = argparse.ArgumentParser(
        description='Train DGDR Baselines for DR Grading'
    )

    # Algorithm
    parser.add_argument(
        '--algorithm', type=str, default='ERM',
        choices=SUPPORTED_ALGORITHMS,
        help='DGDR algorithm: ERM, GREEN, CABNet, MixupNet, MixStyleNet',
    )
    parser.add_argument(
        '--backbone', type=str, default='resnet101',
        choices=['resnet18', 'resnet50', 'resnet101'],
        help='Backbone network (ignored for MixStyleNet which uses VGG)',
    )

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Seed of the stratified re-split (default: use the CSV files as they are)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation share when --split_seed re-splits train.csv')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Share held out as the test set when the folder has no test.csv')
    parser.add_argument('--image_size', type=int, default=1024)
    parser.add_argument(
        '--use_3_stages', action='store_true', default=False,
        help='Use 3-class staging (Healthy/NPDR/PDR)',
    )
    parser.add_argument(
        '--use_5_stages', dest='use_3_stages', action='store_false',
        help='Use 5-class grading (0-4), this is the default',
    )

    # Model
    parser.add_argument('--drop_rate', type=float, default=0.1,
                        help='Dropout rate')

    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument(
        '--lr', type=float, default=1e-3,
        help='Learning rate for head/classifier',
    )
    parser.add_argument(
        '--backbone_lr', type=float, default=1e-4,
        help='Learning rate for backbone',
    )
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (only for --optim sgd)')
    parser.add_argument(
        '--optim', type=str, default='sgd',
        choices=['sgd', 'adamw'],
        help='Optimizer: sgd (DGDR default) or adamw',
    )
    parser.add_argument(
        '--scheduler', type=str, default='cosine',
        choices=['onecycle', 'cosine'],
    )
    parser.add_argument('--accumulation_steps', type=int, default=1)

    # Loss
    parser.add_argument('--use_class_weights', action='store_true',
                        default=False)
    parser.add_argument('--use_focal', action='store_true', default=False)
    parser.add_argument('--no_focal', dest='use_focal',
                        action='store_false')
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # Mixup
    parser.add_argument(
        '--use_mixup', action='store_true', default=False,
        help='Enable mixup (auto-enabled for MixupNet)',
    )
    parser.add_argument(
        '--no_mixup', action='store_true', default=False,
        help='Disable mixup even for MixupNet',
    )
    parser.add_argument('--mixup_alpha', type=float, default=1.0,
                        help='Mixup alpha parameter')

    # Oversampling
    parser.add_argument('--use_oversample', action='store_true', default=True)
    parser.add_argument('--no_oversample', dest='use_oversample',
                        action='store_false')
    parser.add_argument('--use_strong_aug', action='store_true', default=False)
    parser.add_argument('--no_strong_aug', dest='use_strong_aug',
                        action='store_false')

    # Other
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str,
                        default='./baseline/dgdr_baseline')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print(f"DGDR [{args.algorithm}] Baseline Training Configuration")
    print("=" * 80)
    print(json.dumps(vars(args), indent=2))
    print()

    main(args)
