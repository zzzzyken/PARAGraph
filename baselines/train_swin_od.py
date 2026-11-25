"""
OD-aware Swin 训练脚本 (Approach A: ODPosbiasClassifier)

在每个 batch 里用冻结的 OpticUNet 检测 OD 位置，
再把 (od_center, od_diameter, fovea_center) 传给 ODPosbiasClassifier。

主要改动 vs train_swin.py:
  - 模型换成 ODPosbiasClassifier
  - batch loop 里加 OpticUNet 推理 (torch.no_grad, 低分辨率加速)
  - 其余 dataloader / optimizer / scheduler / 评估 逻辑完全复用
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os, sys, json, datetime, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.cuda.amp import GradScaler
from torch.amp import autocast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Models.swin_od_pos_bias import ODPosbiasClassifier, ODPolarClassifier
from Models.optic_unet import OpticUNet
from grading_dataloader import create_grading_dataloaders, print_dataset_statistics

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False

try:
    from sklearn.metrics import cohen_kappa_score
    def quadratic_weighted_kappa(y_true, y_pred, n=5):
        if len(np.unique(y_true)) < 2:
            return 0.0
        return cohen_kappa_score(y_true, y_pred, weights='quadratic',
                                 labels=list(range(n)))
except ImportError:
    def quadratic_weighted_kappa(y_true, y_pred, n=5):
        return 0.0


# ============ Loss ============
def focal_loss(logits, labels, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, labels, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def compute_class_weights(dataset, num_classes, device):
    if hasattr(dataset, 'get_labels'):
        labels = np.array(dataset.get_labels())
    elif hasattr(dataset, 'datasets'):   # ConcatDataset
        labels = np.concatenate([
            np.array(d.get_labels()) if hasattr(d, 'get_labels')
            else np.array([s[1] for s in d.samples])
            for d in dataset.datasets
        ])
    else:
        labels = np.array([s[1] for s in dataset.samples])
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-8)
    weights /= weights.sum()
    return torch.tensor(weights * num_classes, dtype=torch.float32, device=device)


# ============ OD detection helper ============
OD_DETECT_SIZE = 512   # run OpticUNet at this resolution for speed

@torch.no_grad()
def detect_od_batch(optic_unet, images, device):
    """
    Detect OD center, diameter and Fovea center for a batch.
    images: [B, 3, H, W] (training resolution)
    Returns:
        od_centers:   list of (cx, cy) in original image coords
        od_diameters: list of float
        fovea_centers: list of (fx, fy) or None
    """
    B, _, H, W = images.shape

    # downsample for faster inference
    imgs_small = F.interpolate(images, size=(OD_DETECT_SIZE, OD_DETECT_SIZE),
                               mode='bilinear', align_corners=False)

    optic_out = optic_unet(imgs_small)
    od_mask   = optic_out['od_mask']          # [B, 1, h, w]
    fovea_hm  = optic_out.get('fovea_heatmap', None)  # [B, 1, h, w] or None

    processed = OpticUNet.morphological_process_od(od_mask)
    _, od_infos = OpticUNet.compute_od_confidence(processed)

    scale_x = W / OD_DETECT_SIZE
    scale_y = H / OD_DETECT_SIZE

    od_centers, od_diameters, fovea_centers = [], [], []

    for b in range(B):
        info = od_infos[b]
        if info['detected']:
            cx, cy = info['center']
            cx *= scale_x; cy *= scale_y
            # approximate diameter from OD mask area
            od_np = processed[b, 0].cpu().numpy()
            from Models._paragraph_constants import compute_od_diameter
            diam = compute_od_diameter(od_np) * max(scale_x, scale_y)
            if diam < 1.0:
                diam = H / 10.0
        else:
            cx, cy = W / 2.0, H / 2.0
            diam    = H / 10.0
        od_centers.append((cx, cy))
        od_diameters.append(diam)

        # fovea
        if fovea_hm is not None:
            fh = fovea_hm[b, 0]
            idx = fh.argmax()
            fy_s = (idx // fovea_hm.shape[-1]).item()
            fx_s = (idx %  fovea_hm.shape[-1]).item()
            fovea_centers.append((fx_s * scale_x, fy_s * scale_y))
        else:
            fovea_centers.append(None)

    return od_centers, od_diameters, fovea_centers


# ============ Train / Evaluate ============
def train_epoch(model, optic_unet, dataloader, optimizer, device,
                class_weights=None, scheduler=None, scheduler_step='epoch',
                accumulation_steps=1, use_focal=False, focal_gamma=2.0,
                scaler=None, use_amp=False):
    model.train()
    total_loss = total_correct = total_samples = 0
    all_preds, all_labels = [], []
    num_batches = len(dataloader)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        # OD detection (frozen, no grad)
        od_centers, od_diams, fovea_centers = detect_od_batch(optic_unet, images, device)

        with autocast('cuda', enabled=use_amp):
            logits = model(images, od_centers=od_centers,
                           od_diameters=od_diams, fovea_centers=fovea_centers)
            if use_focal:
                loss_val = focal_loss(logits, labels, alpha=class_weights, gamma=focal_gamma)
            else:
                loss_val = F.cross_entropy(logits, labels, weight=class_weights)
            loss = loss_val / accumulation_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if scaler is not None:
                scaler.step(optimizer); scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None and scheduler_step == 'batch':
                scheduler.step()

        preds = logits.argmax(dim=-1)
        total_loss    += loss_val.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if (batch_idx + 1) % max(1, num_batches // 4) == 0 or batch_idx == num_batches - 1:
            print(f"  [{int((batch_idx+1)/num_batches*100):3d}%] "
                  f"loss={total_loss/total_samples:.4f}, acc={total_correct/total_samples:.4f}")

    # flush remaining
    if total_samples % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step(); optimizer.zero_grad()

    num_classes = model.num_classes
    return {
        'loss':     total_loss / total_samples,
        'accuracy': total_correct / total_samples,
        'kappa':    quadratic_weighted_kappa(np.array(all_labels), np.array(all_preds), num_classes),
    }


@torch.no_grad()
def evaluate(model, optic_unet, dataloader, device,
             class_weights=None, use_focal=False, focal_gamma=2.0):
    model.eval()
    total_loss = total_correct = total_samples = 0
    all_preds, all_labels = [], []

    for batch in dataloader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)

        od_centers, od_diams, fovea_centers = detect_od_batch(optic_unet, images, device)
        logits = model(images, od_centers=od_centers,
                       od_diameters=od_diams, fovea_centers=fovea_centers)

        if use_focal:
            loss_val = focal_loss(logits, labels, alpha=class_weights, gamma=focal_gamma)
        else:
            loss_val = F.cross_entropy(logits, labels, weight=class_weights)

        preds = logits.argmax(dim=-1)
        total_loss    += loss_val.item() * images.size(0)
        total_correct += (preds == labels).sum().item()
        total_samples += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    num_classes = model.num_classes
    per_class_acc = {}
    all_preds_np  = np.array(all_preds)
    all_labels_np = np.array(all_labels)
    grade_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'][:num_classes]
    for i, name in enumerate(grade_names):
        mask = (all_labels_np == i)
        per_class_acc[f'Grade_{i}'] = float((all_preds_np[mask] == i).mean()) if mask.sum() > 0 else 0.0

    return {
        'loss':           total_loss / total_samples,
        'accuracy':       total_correct / total_samples,
        'kappa':          quadratic_weighted_kappa(all_labels_np, all_preds_np, num_classes),
        'per_class_acc':  per_class_acc,
    }


# ============ Main ============
def main(args):
    torch.manual_seed(args.seed)
    device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir  = os.path.join(args.save_dir, f'swin_od_{timestamp}')
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    num_classes = 3 if args.use_3_stages else 5
    print(f"\nOD-aware Swin training | {num_classes} classes | device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataloaders = create_grading_dataloaders(
        data_root=args.data_root,
        image_size=args.image_size, batch_size=args.batch_size,
        num_workers=args.num_workers, use_3_stages=args.use_3_stages,
        use_oversample=args.use_oversample, use_strong_aug=args.use_strong_aug,
        split_seed=args.split_seed, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
    )
    # without a validation split the test set takes its place
    if 'val' not in dataloaders or dataloaders['val'] is None:
        dataloaders['val'] = dataloaders['test']
    print_dataset_statistics(dataloaders, use_3_stages=args.use_3_stages)

    # ── OpticUNet (frozen) ────────────────────────────────────────────────────
    print(f"\nLoading OpticUNet from: {args.optic_checkpoint}")
    optic_unet = OpticUNet(encoder_name='resnet50', encoder_pretrained=False).to(device)
    if os.path.exists(args.optic_checkpoint):
        ckpt = torch.load(args.optic_checkpoint, map_location=device, weights_only=False)
        optic_unet.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        print("  Loaded OpticUNet checkpoint")
    optic_unet.eval()
    for p in optic_unet.parameters():
        p.requires_grad = False

    # ── Model ────────────────────────────────────────────────────────────────
    if args.model_type == 'pos_bias':
        print(f"\nCreating ODPosbiasClassifier ({args.swin_model}) [Approach A: positional bias]...")
        model = ODPosbiasClassifier(
            num_classes=num_classes,
            model_name=args.swin_model,
            img_size=args.image_size,
            pretrained=True,
            pos_hidden=args.pos_hidden,
            drop_rate=args.drop_rate,
            backbone_lr_scale=args.backbone_lr / args.lr,
        ).to(device)
    else:  # polar
        print(f"\nCreating ODPolarClassifier ({args.swin_model}) [Approach B: polar resampling]...")
        polar_h = args.polar_size[0]
        polar_w = args.polar_size[1]
        model = ODPolarClassifier(
            num_classes=num_classes,
            model_name=args.swin_model,
            polar_size=(polar_h, polar_w),
            max_r_od=args.max_r_od,
            log_r=args.log_r,
            pretrained=args.polar_pretrained,
            drop_rate=args.drop_rate,
        ).to(device)

    model.num_classes = num_classes

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,}  Trainable: {trainable:,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    if hasattr(model, 'get_param_groups'):
        param_groups = model.get_param_groups(args.lr)
    else:
        param_groups = [
            {'params': model.backbone.parameters(), 'lr': args.backbone_lr},
            {'params': [p for n, p in model.named_parameters()
                        if not n.startswith('backbone')], 'lr': args.lr},
        ]
    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)

    if args.scheduler == 'onecycle':
        max_lrs   = [pg['lr'] for pg in param_groups]
        scheduler = OneCycleLR(optimizer, max_lr=max_lrs, epochs=args.epochs,
                               steps_per_epoch=len(dataloaders['train']),
                               pct_start=0.1, div_factor=25, final_div_factor=1000)
        scheduler_step = 'batch'
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scheduler_step = 'epoch'

    class_weights = None
    if args.use_oversample:
        class_weights = compute_class_weights(dataloaders['train'].dataset, num_classes, device)

    scaler = GradScaler() if args.use_amp else None

    # ── Training loop ─────────────────────────────────────────────────────────
    best_kappa  = -1.0
    best_path   = os.path.join(save_dir, 'best_model.pth')
    writer      = SummaryWriter(os.path.join(save_dir, 'tensorboard')) if HAS_TENSORBOARD else None

    print(f"\n{'='*70}\nStarting training...\n{'='*70}")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}\n{'-'*60}")

        train_metrics = train_epoch(
            model, optic_unet, dataloaders['train'], optimizer, device,
            class_weights=class_weights, scheduler=scheduler,
            scheduler_step=scheduler_step, accumulation_steps=args.accumulation_steps,
            use_focal=args.use_focal, scaler=scaler, use_amp=args.use_amp,
        )
        val_metrics = evaluate(model, optic_unet, dataloaders['val'], device,
                               class_weights=class_weights, use_focal=args.use_focal)

        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"Kappa: {train_metrics['kappa']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f},  Acc: {val_metrics['accuracy']:.4f},  "
              f"Kappa: {val_metrics['kappa']:.4f}")
        print(f"LR (backbone): {lrs[0]:.6f}, LR (head): {lrs[-1]:.6f}")
        print("Val per-class accuracy:")
        for g, a in val_metrics['per_class_acc'].items():
            print(f"  {g}: {a:.4f}")

        if scheduler_step == 'epoch':
            scheduler.step()

        if val_metrics['kappa'] > best_kappa:
            best_kappa = val_metrics['kappa']
            torch.save({'model_state_dict': model.state_dict(),
                        'epoch': epoch, 'kappa': best_kappa,
                        'args': vars(args)}, best_path)
            print(f"  -> Saved best model (Kappa: {best_kappa:.4f})")

        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f'Train/{k}', v, epoch)
            for k, v in val_metrics.items():
                if isinstance(v, float):
                    writer.add_scalar(f'Val/{k}', v, epoch)

    # ── Final test ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}\nFinal Evaluation\n{'='*70}")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    test_metrics = evaluate(model, optic_unet, dataloaders['test'], device,
                            class_weights=class_weights, use_focal=args.use_focal)
    print(f"Test Results:\n  Kappa: {test_metrics['kappa']:.4f}\n  Acc: {test_metrics['accuracy']:.4f}")
    print(f"Best validation Kappa: {best_kappa:.4f}")

    if writer:
        writer.close()


# ============ Args ============
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OD-aware Swin Classifier')

    # model type
    parser.add_argument('--model_type', default='pos_bias', choices=['pos_bias', 'polar'],
                        help='A: pos_bias = OD positional bias on patch tokens; '
                             'B: polar = polar coordinate resampling')
    parser.add_argument('--swin_model', default='swin_tiny_patch4_window7_224')
    parser.add_argument('--pos_hidden', type=int, default=64,
                        help='[pos_bias] Hidden dim in ODRelativePosEncoding MLP')
    parser.add_argument('--drop_rate',  type=float, default=0.0)
    # polar-specific
    parser.add_argument('--polar_size', type=int, nargs=2, default=[512, 1024],
                        help='[polar] Output size (radial, angular)')
    parser.add_argument('--max_r_od',   type=float, default=5.0,
                        help='[polar] Max radius in OD-diameter units')
    parser.add_argument('--log_r',      action='store_true', default=False,
                        help='[polar] Use log-r transform to reduce peripheral undersampling')
    parser.add_argument('--polar_pretrained', action='store_true', default=True,
                        help='[polar] Init from ImageNet pretrained weights')

    # optic unet
    parser.add_argument('--optic_checkpoint', type=str,
                        default='./checkpoints/optic_unet_resnet50/best_model.pth')

    # data
    parser.add_argument('--data_root',       type=str, required=True,
                        help='Dataset folder in the fixed format: images/, train.csv, test.csv (see README)')
    parser.add_argument('--split_seed',      type=int, default=None,
                        help='Seed of the stratified re-split (default: use the CSV files as they are)')
    parser.add_argument('--val_ratio',       type=float, default=0.2)
    parser.add_argument('--test_ratio',      type=float, default=0.2)
    parser.add_argument('--image_size',      type=int, default=1024)
    parser.add_argument('--use_3_stages',    action='store_true', default=False)
    parser.add_argument('--use_strong_aug',  action='store_true', default=False)

    # training
    parser.add_argument('--epochs',            type=int,   default=50)
    parser.add_argument('--batch_size',        type=int,   default=4)
    parser.add_argument('--accumulation_steps',type=int,   default=4)
    parser.add_argument('--lr',                type=float, default=1e-4)
    parser.add_argument('--backbone_lr',       type=float, default=1e-5)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--scheduler',         type=str,   default='cosine',
                        choices=['cosine', 'onecycle'])
    parser.add_argument('--use_oversample',    action='store_true', default=False)
    parser.add_argument('--use_focal',         action='store_true', default=False)
    parser.add_argument('--use_amp',           action='store_true', default=False)
    parser.add_argument('--num_workers',       type=int,   default=4)
    parser.add_argument('--seed',              type=int,   default=42)
    parser.add_argument('--save_dir',          type=str,
                        default='./baseline/swin_od_pos_bias_eyepacs')

    args = parser.parse_args()
    main(args)
