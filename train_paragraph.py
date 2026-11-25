#!/usr/bin/env python3
"""
PARAGraph 训练脚本

训练 PARAGraph 模型进行 DR 分级
- 使用预训练的 OpticUNet 和 LesionUNet
- 支持 3 类 (Healthy/NPDR/PDR) 或 5 类 (0-4 级) 分类
- 使用 Quadratic Weighted Kappa 作为主要评估指标
"""

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
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from Models import PARAGraph, OpticUNet, LesionUNet
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
    """
    Calculate Quadratic Weighted Kappa (QWK)

    This is the standard metric for DR grading competitions (Kaggle, etc.)

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        num_classes: Number of classes

    Returns:
        QWK score in range [-1, 1], higher is better
    """
    # Build confusion matrix
    confusion = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        confusion[int(t), int(p)] += 1

    # Build weight matrix (quadratic weights)
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)

    # Build expected confusion matrix
    hist_true = np.sum(confusion, axis=1)
    hist_pred = np.sum(confusion, axis=0)
    n = np.sum(confusion)

    expected = np.outer(hist_true, hist_pred) / n

    # Calculate kappa
    numerator = np.sum(weights * confusion)
    denominator = np.sum(weights * expected)

    if denominator == 0:
        return 0.0

    kappa = 1.0 - numerator / denominator
    return kappa


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


# ============ Training Functions ============
def train_epoch(
    model: PARAGraph,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    class_weights: torch.Tensor = None,
    scheduler=None,
    scheduler_step: str = 'epoch',
    accumulation_steps: int = 1,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    scaler: GradScaler = None,
    use_amp: bool = False,
    finetune_grade234: bool = False,
) -> dict:
    """Train one epoch"""
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
        # Forward pass (AMP autocast)
        with autocast('cuda', enabled=use_amp):
            output = model.forward(images)

            if finetune_grade234:
                # Grade 2/3/4 微调: 只对 grade>=2 的样本计算 focal loss
                # grade 0/1 用低权重保持不退化
                logits = output['logits']
                mask_234 = (labels >= 2)
                if mask_234.any():
                    loss_main = focal_loss(logits[mask_234], labels[mask_234],
                                           alpha=class_weights, gamma=focal_gamma)
                else:
                    loss_main = torch.tensor(0.0, device=logits.device)
                # grade 0/1 保持 loss (低权重防遗忘)
                mask_01 = ~mask_234
                if mask_01.any():
                    loss_preserve = focal_loss(logits[mask_01], labels[mask_01],
                                               alpha=class_weights, gamma=focal_gamma)
                    loss_val = loss_main + 0.1 * loss_preserve
                else:
                    loss_val = loss_main
            elif use_focal:
                logits = output['logits']
                loss_val = focal_loss(logits, labels, alpha=class_weights, gamma=focal_gamma)
            else:
                logits = output['logits']
                loss_val = F.cross_entropy(logits, labels, weight=class_weights)

            loss = loss_val / accumulation_steps

        if finetune_grade234:
            predictions = output['logits'].argmax(dim=1)
        else:
            predictions = output['logits'].argmax(dim=1)

        # Backward pass (AMP scaled)
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

            # OneCycleLR steps per batch
            if scheduler is not None and scheduler_step == 'batch':
                scheduler.step()

        # Metrics
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

    # Final gradient step if needed
    if total_samples % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    # Calculate epoch metrics
    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa = quadratic_weighted_kappa(
        np.array(all_labels),
        np.array(all_preds),
        model.num_classes
    )

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'kappa': kappa,
    }


@torch.no_grad()
def evaluate(
    model: PARAGraph,
    dataloader: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor = None,
    use_focal: bool = False,
    focal_gamma: float = 2.0,
    finetune_grade234: bool = False,
) -> dict:
    """Evaluate model"""
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    # Per-class metrics
    num_classes = model.num_classes
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        output = model.forward(images)

        if finetune_grade234:
            logits = output['logits']
            loss_val = focal_loss(logits, labels, alpha=class_weights, gamma=focal_gamma)
            predictions = logits.argmax(dim=1)
        elif use_focal:
            logits = output['logits']
            loss_val = focal_loss(logits, labels, alpha=class_weights, gamma=focal_gamma)
            predictions = logits.argmax(dim=1)
        else:
            logits = output['logits']
            loss_val = F.cross_entropy(logits, labels, weight=class_weights)
            predictions = logits.argmax(dim=1)

        total_loss += loss_val.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += images.size(0)

        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Per-class accuracy
        for pred, label in zip(predictions.cpu().numpy(), labels.cpu().numpy()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples
    kappa = quadratic_weighted_kappa(
        np.array(all_labels),
        np.array(all_preds),
        num_classes
    )

    # Per-class accuracy
    class_accuracy = {}
    class_names = STAGE_NAMES if num_classes == 3 else [f'Grade_{i}' for i in range(num_classes)]
    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            class_accuracy[name] = class_correct[i] / class_total[i]
        else:
            class_accuracy[name] = 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'kappa': kappa,
        'class_accuracy': class_accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
    }


def build_confusion_matrix(labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
    """Build confusion matrix"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(labels, predictions):
        cm[int(t), int(p)] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, class_names: list):
    """Pretty print confusion matrix"""
    print("\nConfusion Matrix:")
    print("-" * 60)

    # Header
    header = "True\\Pred"
    for name in class_names:
        header += f"\t{name[:8]}"
    print(header)

    # Rows
    for i, name in enumerate(class_names):
        row = f"{name[:8]}"
        for j in range(len(class_names)):
            row += f"\t{cm[i, j]}"
        print(row)


# ============ Main Training Loop ============
def set_seed(seed: int, deterministic: bool = False):
    """固定所有随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def main(args):
    # 不固定随机种子，启用 cudnn benchmark 加速
    torch.backends.cudnn.benchmark = True

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 80)

    # Create save directory
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = os.path.join(args.save_dir, f'paragraph_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)

    # Save args
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    # Determine number of classes
    num_classes = 3 if args.use_3_stages else 5
    print(f"\nTraining with {num_classes} classes")

    # ============ Load Data ============
    print("\nLoading datasets...")

    dataloaders = create_grading_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_3_stages=args.use_3_stages,
        use_oversample=args.use_oversample,
        use_strong_aug=args.use_strong_aug,
        use_balanced_sampler=args.use_balanced_sampler,
        split_seed=args.split_seed,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    print_dataset_statistics(dataloaders, use_3_stages=args.use_3_stages)

    # ============ Create Model ============
    print("\nCreating model...")

    # Load OpticUNet
    print(f"Loading OpticUNet from: {args.optic_checkpoint}")
    print(f"OpticUNet encoder: {args.optic_encoder}")
    optic_unet = OpticUNet(
        encoder_pretrained=True,
        predict_fovea=True,
        encoder_name=args.optic_encoder,
        img_size=args.image_size
    )
    if args.optic_checkpoint and os.path.exists(args.optic_checkpoint):
        checkpoint = torch.load(args.optic_checkpoint, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        optic_unet.load_state_dict(state_dict, strict=False)
        print(f"  Loaded OpticUNet checkpoint")
    else:
        print(f"  Warning: OpticUNet checkpoint not found, using random initialization")

    # Load LesionUNet
    lesion_unet = None
    if args.lesion_checkpoint:
        print(f"Loading LesionUNet from: {args.lesion_checkpoint}")
        print(f"LesionUNet encoder: {args.lesion_encoder}")
        # 自动检测 large_decoder (1月版 checkpoint: decoder_stage4 输出 1024ch 而非 256ch)
        _large_decoder = False
        if os.path.exists(args.lesion_checkpoint):
            _probe = torch.load(args.lesion_checkpoint, map_location='cpu', weights_only=False)
            _sd = _probe.get('model_state_dict', _probe)
            _bias = _sd.get('decoder_stage4.1.bias', None)
            if _bias is not None and _bias.shape[0] > 256:
                _large_decoder = True
                print(f"  Detected large_decoder checkpoint (decoder_stage4 out={_bias.shape[0]})")
            del _probe, _sd
        lesion_unet = LesionUNet(
            num_lesions=4,
            encoder_pretrained=True,
            encoder_name=args.lesion_encoder,
            img_size=args.image_size,
            large_decoder=_large_decoder,
        )
        if os.path.exists(args.lesion_checkpoint):
            checkpoint = torch.load(args.lesion_checkpoint, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            lesion_unet.load_state_dict(state_dict, strict=False)
            print(f"  Loaded LesionUNet checkpoint")
        else:
            print(f"  Warning: LesionUNet checkpoint not found, using random initialization")

    # Create PARAGraph (dual-branch: anatomical + global)
    if args.no_lesion_unet:
        lesion_unet = None
        print("[Ablation] --no_lesion_unet: LesionUNet disabled (graph = OD + Global only)")

    model = PARAGraph(
        optic_unet=optic_unet,
        lesion_unet=lesion_unet,
        feature_dim=args.feature_dim,
        num_classes=num_classes,
        num_gnn_layers=args.num_gnn_layers,
        gnn_dropout=args.gnn_dropout,
        num_region_nodes=args.num_region_nodes,
        use_prior_fusion=args.use_prior_fusion,
        fusion_mode=args.fusion_mode,
        use_optic_nodes=args.use_optic_nodes,
        use_gating=args.use_gating,
        use_global_logits=args.use_global_logits,
        use_global_node=args.use_global_node,
        use_region_nodes=args.use_region_nodes,
        use_od_posbias=args.use_od_posbias,
        lesion_direct=args.lesion_direct,
    ).to(device)

    # ============ Resume: 加载已有 PARAGraph 权重 ============
    start_epoch = 1
    resume_best_kappa = -1.0
    resume_history = None
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        print(f"Loaded PARAGraph weights from: {args.resume}")
        # 如果 checkpoint 含完整训练状态则恢复
        if 'epoch' in ckpt and 'optimizer_state_dict' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            resume_best_kappa = ckpt.get('best_kappa', -1.0)
            resume_history = ckpt.get('history', None)
            print(f"  Resuming from epoch {start_epoch}, best_kappa={resume_best_kappa:.4f}")

    # Fovea decoder always frozen (localization pretrained)
    if model.optic_unet.predict_fovea:
        for module in [model.optic_unet.fovea_decoder_stage4,
                       model.optic_unet.fovea_decoder_stage3,
                       model.optic_unet.fovea_decoder_stage2,
                       model.optic_unet.fovea_decoder_stage1,
                       model.optic_unet.fovea_final_upsample,
                       model.optic_unet.fovea_head]:
            for param in module.parameters():
                param.requires_grad = False
        print("Landmark coordinate decoder frozen (fovea_decoder + fovea_head)")

    # Freeze backbone if requested
    if args.freeze_backbone:
        print("Freezing OpticUNet and LesionUNet backbones...")
        for param in model.optic_unet.parameters():
            param.requires_grad = False
        if model.lesion_unet is not None:
            for param in model.lesion_unet.parameters():
                param.requires_grad = False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ============ Optimizer & Scheduler ============
    if args.finetune_grade234:
        # Grade234 微调: 训练 GNN + 分类器 + graph pool
        params = [{'params': [p for p in model.parameters() if p.requires_grad], 'lr': args.lr}]
    elif args.freeze_backbone:
        # backbone 冻结时，global_swin 用 backbone_lr，其余用 lr
        global_swin_param_ids = {id(p) for n, p in model.named_parameters()
                                 if p.requires_grad and n.startswith('global_swin')}
        global_swin_params = [p for p in model.parameters()
                              if p.requires_grad and id(p) in global_swin_param_ids]
        other_params = [p for p in model.parameters()
                        if p.requires_grad and id(p) not in global_swin_param_ids]
        params = [
            {'params': global_swin_params, 'lr': args.backbone_lr},
            {'params': other_params,       'lr': args.lr},
        ]
    else:
        # 可训练的 optic_unet 参数 (坐标 decoder 已冻结)
        optic_params = [p for p in model.optic_unet.parameters() if p.requires_grad]
        lesion_params = [p for p in model.lesion_unet.parameters() if p.requires_grad] if model.lesion_unet else []
        other_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and not n.startswith('optic_unet') and not n.startswith('lesion_unet')]

        params = [
            {'params': optic_params, 'lr': args.backbone_lr},
            {'params': lesion_params, 'lr': args.backbone_lr},
            {'params': other_params, 'lr': args.lr}
        ]

    optimizer = AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    if args.scheduler == 'onecycle':
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr,
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

    # 恢复 optimizer/scheduler 状态（resume 时）
    if args.resume and os.path.exists(args.resume) and start_epoch > 1:
        _ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if 'optimizer_state_dict' in _ckpt:
            optimizer.load_state_dict(_ckpt['optimizer_state_dict'])
        if 'scheduler_state_dict' in _ckpt:
            scheduler.load_state_dict(_ckpt['scheduler_state_dict'])
        del _ckpt
        print(f"  Restored optimizer and scheduler state")

    # ============ AMP (Mixed Precision) ============
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = GradScaler('cuda', enabled=use_amp) if use_amp else None
    if use_amp:
        print("Mixed precision training (AMP) enabled")

    # ============ TensorBoard ============
    writer = None
    if HAS_TENSORBOARD:
        log_dir = os.path.join(save_dir, 'tensorboard')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        print(f"TensorBoard logging to: {log_dir}")

    # ============ Training Loop ============
    print("\n" + "=" * 80)
    print("Starting training...")
    print("=" * 80)

    best_kappa = resume_best_kappa
    best_epoch = 0
    history = resume_history if resume_history else {'train': [], 'val': []}

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

            # ============ Unfreeze decoder at specified epoch ============
        if (
            args.unfreeze_decoder
            and epoch == args.unfreeze_epoch
        ):
            print("\n" + "=" * 80)
            print(f"Unfreezing decoder at epoch {epoch}")
            print("=" * 80)

            model.unfreeze_backbone_decoder(
                optic_od=True,
                optic_landmark=False,  # 坐标 decoder 始终冻结
                lesion=True,
                stages=args.unfreeze_stages
            )

            # 关键：重新构建 optimizer（decoder 用更小 lr）
            param_groups = [
                {
                    'params': [p for p in model.parameters() if p.requires_grad],
                    'lr': args.lr
                }
            ]

            # 单独给 decoder 更小 lr
            if hasattr(model, 'get_decoder_params'):
                param_groups.append({
                    'params': model.get_decoder_params(),
                    'lr': args.decoder_lr
                })

            optimizer = AdamW(
                param_groups,
                weight_decay=args.weight_decay
            )

            # 重建 scheduler
            if args.scheduler == 'cosine':
                scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=args.epochs - epoch + 1,
                    eta_min=1e-6
                )


        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=dataloaders['train'],
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=scheduler if scheduler_step == 'batch' else None,
            scheduler_step=scheduler_step,
            accumulation_steps=args.accumulation_steps,
            class_weights=class_weights,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma,
            scaler=scaler,
            use_amp=use_amp,
            finetune_grade234=args.finetune_grade234,
        )

        # Validate
        val_metrics = evaluate(
            model=model,
            dataloader=dataloaders['test'],
            device=device,
            class_weights=class_weights,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma,
            finetune_grade234=args.finetune_grade234,
        )

        # Epoch scheduler step
        if scheduler_step == 'epoch':
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        # Record history
        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # Print metrics
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, Kappa: {train_metrics['kappa']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, Kappa: {val_metrics['kappa']:.4f}")
        print(f"LR: {current_lr:.6f}")

        # Per-class accuracy
        print("Val per-class accuracy:")
        for name, acc in val_metrics['class_accuracy'].items():
            print(f"  {name}: {acc:.4f}")

        # TensorBoard logging
        if writer:
            writer.add_scalars('Loss', {
                'train': train_metrics['loss'],
                'val': val_metrics['loss']
            }, epoch)
            writer.add_scalars('Accuracy', {
                'train': train_metrics['accuracy'],
                'val': val_metrics['accuracy']
            }, epoch)
            writer.add_scalars('Kappa', {
                'train': train_metrics['kappa'],
                'val': val_metrics['kappa']
            }, epoch)
            writer.add_scalar('LR', current_lr, epoch)

        # Save best model (基于 QWK)
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

        # Save periodic checkpoint
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

    # Load best model
    checkpoint = torch.load(os.path.join(save_dir, 'best_model.pth'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Evaluate on test set
    if 'test' in dataloaders:
        test_metrics = evaluate(
            model=model,
            dataloader=dataloaders['test'],
            device=device,
            class_weights=class_weights,
            use_focal=args.use_focal,
            focal_gamma=args.focal_gamma,
            finetune_grade234=args.finetune_grade234,
        )

        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"  Kappa: {test_metrics['kappa']:.4f}")

        # Per-class accuracy
        print("\nTest per-class accuracy:")
        for name, acc in test_metrics['class_accuracy'].items():
            print(f"  {name}: {acc:.4f}")

        # Confusion matrix
        cm = build_confusion_matrix(
            test_metrics['labels'],
            test_metrics['predictions'],
            num_classes
        )
        class_names = STAGE_NAMES if num_classes == 3 else [f'Grade_{i}' for i in range(num_classes)]
        print_confusion_matrix(cm, class_names)

        # Save test results
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
        # Convert numpy values to Python floats
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
    parser = argparse.ArgumentParser(description='Train PARAGraph for DR Grading')

    # Data arguments
    parser.add_argument('--data_root', type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed', type=int, default=None,
                        help='Seed of the stratified re-split (default: use the CSV files as they are)')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation share when --split_seed re-splits train.csv')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Share held out as the test set when the folder has no test.csv')
    parser.add_argument('--image_size', type=int, default=1024,
                       help='Input image size (default: 1024)')
    parser.add_argument('--use_3_stages', action='store_true', default=False,
                       help='Use 3-class staging (Healthy/NPDR/PDR)')
    parser.add_argument('--use_5_stages', dest='use_3_stages', action='store_false',
                       help='Use 5-class grading (0-4)')

    # Model checkpoints
    parser.add_argument('--optic_checkpoint', type=str,
                       default='./checkpoints/optic_unet/best_model.pth',
                       help='Path to pretrained OpticUNet checkpoint')
    parser.add_argument('--lesion_checkpoint', type=str,
                       default='./checkpoints/lesion_unet/best_model.pth',
                       help='Path to pretrained LesionUNet checkpoint')

    # Encoder (separate for OpticUNet and LesionUNet)
    parser.add_argument('--optic_encoder', type=str, default='resnet50',
                       choices=['resnet50'],
                       help='OpticUNet encoder backbone (ResNet-50)')
    parser.add_argument('--lesion_encoder', type=str, default='swin',
                       choices=['resnet50', 'swin'],
                       help='LesionUNet encoder backbone (default: swin = Swin-T)')

    # Model architecture
    parser.add_argument('--feature_dim', type=int, default=256,
                       help='Feature dimension')
    parser.add_argument('--num_gnn_layers', type=int, default=3,
                       help='Number of GNN layers')
    parser.add_argument('--gnn_dropout', type=float, default=0.1,
                       help='GNN dropout rate')
    parser.add_argument('--num_region_nodes', type=int, default=4,
                       help='Number of K-means cluster nodes for region aggregation')
    parser.add_argument('--topk_lesions_per_cluster', type=int, default=None,
                       help='Top-k lesions per cluster for region/class nodes (None=use all)')
    # Ablation experiments
    parser.add_argument('--use_prior_fusion', action='store_true', default=True,
                       help='Use medical prior fusion (default: True)')
    parser.add_argument('--no_prior_fusion', dest='use_prior_fusion', action='store_false',
                       help='Disable medical prior fusion (ablation experiment)')
    parser.add_argument('--fusion_mode', type=str, default='full', choices=['pos_only', 'full'],
                       help='Fusion mode: pos_only=coords(2), full=coords+onehot+area+dist(8) (default: full)')
    parser.add_argument('--use_optic_nodes', action='store_true', default=True,
                       help='Use OD/Landmark nodes in GNN (default: True)')
    parser.add_argument('--no_optic_nodes', dest='use_optic_nodes', action='store_false',
                       help='Disable OD/Landmark nodes (ablation experiment)')
    parser.add_argument('--use_gating', action='store_true', default=True,
                       help='Use self-learning dual-confidence gating (default: True)')
    parser.add_argument('--no_gating', dest='use_gating', action='store_false',
                       help='Disable gating mechanism (ablation experiment)')

    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate for PARAGraph components')
    parser.add_argument('--backbone_lr', type=float, default=1e-5,
                       help='Learning rate for backbone (OpticUNet/LesionUNet)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['onecycle', 'cosine'],
                       help='Learning rate scheduler')
    parser.add_argument('--accumulation_steps', type=int, default=2,
                       help='Gradient accumulation steps')

    # Regularization
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                       help='Freeze OpticUNet and LesionUNet')
    parser.add_argument('--no_freeze_backbone', dest='freeze_backbone', action='store_false',
                       help='Unfreeze backbone for end-to-end training')
    # Decoder unfreeze (finetuning)
    parser.add_argument('--unfreeze_decoder', action='store_true', default=False,
                        help='Unfreeze decoder of OpticUNet / LesionUNet')
    parser.add_argument('--unfreeze_epoch', type=int, default=51,
                        help='Epoch to start unfreezing decoder')

    parser.add_argument('--unfreeze_stages', type=int, nargs='+', default=[3, 4],
                        help='Decoder stages to unfreeze (e.g. 3 4)')

    parser.add_argument('--decoder_lr', type=float, default=1e-5,
                        help='Learning rate for unfrozen decoder')

    # Loss
    parser.add_argument('--finetune_grade234', action='store_true', default=False,
                        help='Grade 2/3/4 fine-tune mode: freeze encoder, train GNN+classifier (requires --resume)')
    parser.add_argument('--use_class_weights', action='store_true', default=False)
    parser.add_argument('--use_focal', action='store_true', default=False,
                        help='Use focal loss')
    parser.add_argument('--focal_gamma', type=float, default=2.0)

    # Oversampling (for minority class like PDR)
    parser.add_argument('--use_oversample', action='store_true', default=True,
                       help='Use WeightedRandomSampler to oversample minority classes')
    parser.add_argument('--no_oversample', dest='use_oversample', action='store_false',
                       help='Disable oversampling')
    parser.add_argument('--use_balanced_sampler', action='store_true', default=False,
                       help='Use BalancedBatchSampler (guarantees all classes in every batch)')
    parser.add_argument('--use_strong_aug', action='store_true', default=False,
                       help='Use stronger data augmentation')
    parser.add_argument('--no_strong_aug', dest='use_strong_aug', action='store_false',
                       help='Use standard data augmentation')

    # Performance
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='Use automatic mixed precision (AMP)')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                       help='Disable AMP')

    # Other
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/paragraph',
                       help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--resume', type=str, default='',
                       help='Path to PARAGraph checkpoint (.pth) to load model weights from')
    parser.add_argument('--seed', type=int, default=1234,
                       help='Random seed for reproducibility')
    # Dual-branch (anatomical + global) is the default model. Use --no_global_logits
    # for the single-graph-branch ablation.
    parser.add_argument('--no_global_logits', dest='use_global_logits',
                       action='store_false', default=True,
                       help='Ablation: drop the global_swin head branch '
                            '(logits = anatomical_logits only).')
    # ---- Ablation switches (all default to the full model) ----
    parser.add_argument('--no_global_node', dest='use_global_node',
                       action='store_false', default=True,
                       help='Ablation: remove the Global(Fovea) node from the graph.')
    parser.add_argument('--no_region', dest='use_region_nodes',
                       action='store_false', default=True,
                       help='Ablation: do not build Region (lesion-cluster) nodes.')
    parser.add_argument('--no_od_posbias', dest='use_od_posbias',
                       action='store_false', default=True,
                       help='Ablation: disable the OD positional bias in global_swin.')
    parser.add_argument('--lesion_direct', dest='lesion_direct',
                       action='store_true', default=False,
                       help='Ablation (hierarchy L1): lesions connect directly to '
                            'OD/Global hubs, skipping Category and Region nodes.')
    parser.add_argument('--no_lesion_unet', dest='no_lesion_unet',
                       action='store_true', default=False,
                       help='Ablation (hierarchy L0 / Seg=0): do not load LesionUNet; '
                            'graph has only OD + Global nodes.')
    parser.add_argument('--deterministic', action='store_true', default=False,
                       help='Enable deterministic mode (slower but reproducible)')

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PARAGraph Training Configuration")
    print("=" * 80)
    print(json.dumps(vars(args), indent=2))
    print()

    main(args)
