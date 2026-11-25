"""
比较 LesionUNet Swin-T encoder 的注意力区域 vs 分割输出 mask

核心问题: encoder 激活了但分割头没有输出正例的区域有多大?
这个"gap"就是 Residual Node 要捕获的信号。

不需要 GT mask, 可在任意数据集上运行。

用法:
    python visualization/analyze_encoder_vs_segmentation.py --img_dir path/to/images [--ext jpg]
    python visualization/analyze_encoder_vs_segmentation.py --img_dir path/to/images --ext png
    python visualization/analyze_encoder_vs_segmentation.py --img_dir path/to/other/images --ext jpg
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os, sys, argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.lesion_unet import LesionUNet
from Models.optic_unet import OpticUNet

CHECKPOINT       = "./checkpoints/lesion_unet_swint/best_model.pth"
OPTIC_CHECKPOINT = "./checkpoints/optic_unet_resnet50/best_model.pth"
IMG_SIZE    = 1024
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LESION_NAMES = ['MA', 'HE', 'EX', 'SE']
LESION_COLORS = np.array([
    [255, 100, 100],   # MA - red
    [100, 100, 255],   # HE - blue
    [255, 220,  50],   # EX - yellow
    [100, 220, 100],   # SE - green
], dtype=np.float32)
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

ATTN_PERCENTILE = 80  # top-20% encoder activation = "encoder notices here"
SEG_THRESH      = 0.5


def load_model():
    model = LesionUNet(
        num_lesions=4, encoder_name='swin',
        swin_model_name='swin_tiny_patch4_window7_224',
        img_size=IMG_SIZE, encoder_pretrained=False,
    ).to(DEVICE)
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded LesionUNet: {CHECKPOINT}")
    return model


def load_optic_model():
    model = OpticUNet(encoder_name='resnet50', encoder_pretrained=False).to(DEVICE)
    ckpt  = torch.load(OPTIC_CHECKPOINT, map_location=DEVICE, weights_only=False)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"Loaded OpticUNet: {OPTIC_CHECKPOINT}")
    return model


def get_od_mask(optic_model, img_bgr, od_dilation_px=30):
    """Run OpticUNet and return binary OD mask at original resolution (with dilation margin)."""
    H, W = img_bgr.shape[:2]
    tensor = preprocess(img_bgr)
    with torch.no_grad():
        out = optic_model(tensor)
    od_logit = out['od_mask'][0, 0].cpu().numpy()       # [h, w]
    od_prob   = 1 / (1 + np.exp(-od_logit))             # sigmoid
    od_resized = cv2.resize(od_prob, (W, H), interpolation=cv2.INTER_LINEAR)
    od_binary  = (od_resized > 0.5).astype(np.uint8)
    # Dilate to give a margin around OD
    if od_dilation_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (od_dilation_px * 2, od_dilation_px * 2))
        od_binary = cv2.dilate(od_binary, kernel)
    return od_binary


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)


def run_model(model, img_bgr):
    """
    Returns:
        feat_mag_full: [H, W] normalized encoder stage4 magnitude, upsampled to original size
        seg_prob_full: [4, H, W] sigmoid segmentation probabilities, upsampled to original size
    """
    H, W = img_bgr.shape[:2]
    tensor = preprocess(img_bgr)

    feat_store = {}
    def hook_fn(m, inp, out):
        feat_store['feat'] = out.detach()
    hook = model.swin_encoder.stage2_proj.register_forward_hook(hook_fn)

    with torch.no_grad():
        output = model(tensor, mode='full')
    hook.remove()

    # Seg output
    if isinstance(output, dict):
        seg_logits = output.get('masks', output.get('logits'))
    else:
        seg_logits = output
    seg_prob = torch.sigmoid(seg_logits[0]).cpu().numpy()  # [4, h, w]

    # Encoder feature magnitude (stage2: H/8 x W/8 = 128x128 for 1024 input)
    feat = feat_store['feat'][0]                          # [512, h8, w8]
    feat_mag = feat.abs().mean(dim=0).cpu().numpy()
    feat_mag = (feat_mag - feat_mag.min()) / (feat_mag.max() - feat_mag.min() + 1e-8)

    # Upsample both to original resolution
    feat_mag_full = cv2.resize(feat_mag, (W, H), interpolation=cv2.INTER_LINEAR)
    seg_prob_full = np.stack([
        cv2.resize(seg_prob[i], (W, H), interpolation=cv2.INTER_LINEAR)
        for i in range(4)
    ])
    return feat_mag_full, seg_prob_full  # feat_mag_full: raw, before any OD masking


def compute_metrics(feat_mag_full, seg_prob_full, od_mask=None):
    """
    Compare encoder attention region vs segmentation output (no GT needed).

    Definitions:
      A = encoder attention region = top-20% of feat_mag  (OD region zeroed if od_mask given)
      S = segmentation positive = any lesion predicted > threshold

    od_mask: binary [H,W], 1 = optic disc region to suppress
    """
    H, W  = feat_mag_full.shape

    # Suppress OD region in feature magnitude before thresholding
    feat_mag_masked = feat_mag_full.copy()
    if od_mask is not None:
        feat_mag_masked[od_mask > 0] = 0.0

    attn_thresh = np.percentile(feat_mag_masked, ATTN_PERCENTILE)
    A = (feat_mag_masked >= attn_thresh)                 # [H, W] bool
    attn_coverage = A.mean()

    per_lesion = {}
    for i, name in enumerate(LESION_NAMES):
        S = seg_prob_full[i] > SEG_THRESH

        overlap   = (A & S).sum()
        union     = (A | S).sum()
        iou       = overlap / (union + 1e-8)

        # Encoder-only: encoder activates but seg is negative
        enc_only  = (A & ~S).mean()  # fraction of image
        # Seg-only: seg positive but encoder low
        seg_only  = (~A & S).mean()

        # Of the "encoder-only" area: what fraction of the full seg area is missed?
        seg_area  = S.mean()
        seg_recall_by_attn = (A & S).sum() / (S.sum() + 1e-8)  # how much of seg is inside encoder attn

        per_lesion[name] = {
            'seg_coverage':       float(seg_area),
            'attn_coverage':      float(attn_coverage),
            'iou_attn_seg':       float(iou),
            'enc_only_area':      float(enc_only),    # encoder sees but seg misses
            'seg_only_area':      float(seg_only),    # seg sees but encoder misses
            'seg_recall_by_attn': float(seg_recall_by_attn),  # how much of seg output is inside attn region
        }

    # Combined: seg = any lesion positive
    S_any = (seg_prob_full > SEG_THRESH).any(axis=0)
    overlap_any = (A & S_any).sum()
    combined = {
        'attn_coverage':      float(attn_coverage),
        'seg_coverage':       float(S_any.mean()),
        'iou_attn_seg':       float(overlap_any / ((A | S_any).sum() + 1e-8)),
        'enc_only_area':      float((A & ~S_any).mean()),
        'seg_only_area':      float((~A & S_any).mean()),
        'seg_recall_by_attn': float(overlap_any / (S_any.sum() + 1e-8)),
    }

    return per_lesion, combined, A, S_any


def visualize(img_id, img_bgr, feat_mag, seg_prob_full, A, S_any, per_lesion, out_path, od_mask=None):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'{img_id}  —  Encoder Attention vs Segmentation Output', fontsize=13)

    # (0,0) Original
    axes[0, 0].imshow(img_rgb); axes[0, 0].set_title('Original'); axes[0, 0].axis('off')

    # (0,1) Encoder attention heatmap
    feat_mag_disp = feat_mag.copy()
    if od_mask is not None:
        feat_mag_disp[od_mask > 0] = 0.0   # zero out OD for display too
    axes[0, 1].imshow(img_rgb, alpha=0.4)
    axes[0, 1].imshow(feat_mag_disp, cmap='hot', alpha=0.65, vmin=0, vmax=1)
    axes[0, 1].contour(A, colors='white', linewidths=1)
    od_label = ' (OD masked)' if od_mask is not None else ''
    axes[0, 1].set_title(f'Encoder Attention (stage4){od_label}\ntop-{100-ATTN_PERCENTILE}% boundary = white')
    axes[0, 1].axis('off')

    # (0,2) Segmentation output overlay (all 4 lesions)
    vis_seg = img_rgb.copy().astype(np.float32)
    for i, name in enumerate(LESION_NAMES):
        S = seg_prob_full[i] > SEG_THRESH
        if S.any():
            vis_seg[S] = vis_seg[S] * 0.4 + LESION_COLORS[i] * 0.6
    axes[0, 2].imshow(vis_seg.astype(np.uint8))
    axes[0, 2].set_title('Segmentation Output\n(MA=red HE=blue EX=yellow SE=green)')
    axes[0, 2].axis('off')

    # (1,0) Gap map: encoder-only (blue), seg-only (red), overlap (green)
    gap_map = np.zeros((H, W, 3), dtype=np.uint8)
    gap_map[A & ~S_any]  = [0,   100, 255]  # encoder only → blue
    gap_map[~A & S_any]  = [255,  50,  50]  # seg only → red
    gap_map[A  &  S_any] = [0,   220, 100]  # both → green
    vis_gap = (img_rgb.astype(np.float32) * 0.35 + gap_map.astype(np.float32) * 0.65).clip(0,255).astype(np.uint8)
    axes[1, 0].imshow(vis_gap)
    patches = [
        mpatches.Patch(color=[0, 100/255, 1],     label='Encoder-only (missed by seg)'),
        mpatches.Patch(color=[1, 50/255, 50/255], label='Seg-only (low encoder activ.)'),
        mpatches.Patch(color=[0, 220/255, 100/255], label='Both'),
    ]
    axes[1, 0].legend(handles=patches, loc='lower right', fontsize=7)
    axes[1, 0].set_title('Attention vs Seg Overlap Map')
    axes[1, 0].axis('off')

    # (1,1) Per-lesion bar chart
    ax = axes[1, 1]
    names = LESION_NAMES
    attn_rec = [per_lesion[n]['seg_recall_by_attn'] for n in names]
    enc_only = [per_lesion[n]['enc_only_area'] * 100 for n in names]
    x = np.arange(len(names))
    b1 = ax.bar(x - 0.2, [v*100 for v in attn_rec], 0.35, label='Seg covered by Attn (%)', color='steelblue')
    b2 = ax.bar(x + 0.2, enc_only, 0.35, label='Encoder-only area (% of image)', color='tomato')
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel('%'); ax.set_title('Per-lesion Metrics')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1,
                f'{b.get_height():.1f}', ha='center', va='bottom', fontsize=7)

    # (1,2) Per-lesion seg coverage vs attn coverage
    ax2 = axes[1, 2]
    seg_cov  = [per_lesion[n]['seg_coverage'] * 100 for n in names]
    attn_cov = [per_lesion[n]['attn_coverage'] * 100 for n in names]
    iou_vals = [per_lesion[n]['iou_attn_seg'] for n in names]
    x2 = np.arange(len(names))
    ax2.bar(x2 - 0.2, attn_cov, 0.35, label=f'Attn coverage (top-{100-ATTN_PERCENTILE}%)', color='orange')
    ax2.bar(x2 + 0.2, seg_cov,  0.35, label='Seg coverage',  color='mediumseagreen')
    for xi, iou in zip(x2, iou_vals):
        ax2.text(xi, max(attn_cov[xi], seg_cov[xi]) + 1, f'IoU={iou:.2f}', ha='center', fontsize=7)
    ax2.set_xticks(x2); ax2.set_xticklabels(names)
    ax2.set_ylabel('% of image'); ax2.set_title('Coverage: Encoder Attn vs Seg Output')
    ax2.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',  required=True, help='Folder with fundus images')
    parser.add_argument('--ext',      default='png')
    parser.add_argument('--max_vis',  type=int, default=15)
    parser.add_argument('--out_dir',  default=None)
    parser.add_argument('--mask_od',  action='store_true', default=False,
                        help='Suppress optic disc region from encoder attention before thresholding')
    parser.add_argument('--od_dilation', type=int, default=40,
                        help='OD mask dilation in pixels (original resolution)')
    args = parser.parse_args()

    suffix  = '_no_od' if args.mask_od else ''
    out_dir = args.out_dir or f"./analysis_attn_vs_seg_{Path(args.img_dir).parent.name}{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    model       = load_model()
    optic_model = load_optic_model() if args.mask_od else None
    img_files = sorted(Path(args.img_dir).glob(f'*.{args.ext}'))
    print(f"\nFound {len(img_files)} images in {args.img_dir}")

    accum = {n: {k: [] for k in ['seg_coverage','iou_attn_seg','enc_only_area','seg_only_area','seg_recall_by_attn']}
             for n in LESION_NAMES}
    accum['_combined'] = {k: [] for k in ['seg_coverage','iou_attn_seg','enc_only_area','seg_only_area','seg_recall_by_attn','attn_coverage']}

    vis_count = 0
    for img_path in img_files:
        img_id  = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        try:
            feat_mag, seg_prob_full = run_model(model, img_bgr)
            od_mask = get_od_mask(optic_model, img_bgr, args.od_dilation) if optic_model else None
            per_lesion, combined, A, S_any = compute_metrics(feat_mag, seg_prob_full, od_mask)
        except Exception as e:
            print(f"  ERROR {img_id}: {e}"); continue

        # Skip images with near-zero segmentation (likely normal images)
        if combined['seg_coverage'] < 0.0005:
            continue

        for name in LESION_NAMES:
            for k in accum[name]:
                accum[name][k].append(per_lesion[name][k])
        for k in accum['_combined']:
            accum['_combined'][k].append(combined[k])

        print(f"{img_id}: attn_cov={combined['attn_coverage']:.1%}  "
              f"seg_cov={combined['seg_coverage']:.1%}  "
              f"IoU(A,S)={combined['iou_attn_seg']:.3f}  "
              f"enc_only={combined['enc_only_area']:.1%}  "
              f"seg_in_attn={combined['seg_recall_by_attn']:.1%}")

        if vis_count < args.max_vis:
            visualize(img_id, img_bgr, feat_mag, seg_prob_full, A, S_any, per_lesion,
                      os.path.join(out_dir, f"{img_id}.png"), od_mask=od_mask)
            vis_count += 1

    # ============ Summary ============
    n_imgs = len(accum['_combined']['iou_attn_seg'])
    print(f"\n{'='*72}")
    print(f"SUMMARY — {Path(args.img_dir).parent.name} ({n_imgs} images with seg output)")
    print(f"{'='*72}")
    c = accum['_combined']
    print(f"  Encoder attention coverage (top-20%): {np.mean(c['attn_coverage']):.2%}")
    print(f"  Segmentation coverage (any lesion>0.5): {np.mean(c['seg_coverage']):.2%}")
    print(f"  IoU(Attn, Seg):  {np.mean(c['iou_attn_seg']):.4f}")
    print(f"  Encoder-only area: {np.mean(c['enc_only_area']):.2%} of image  ← Residual Node target")
    print(f"  Seg-only area:     {np.mean(c['seg_only_area']):.2%} of image")
    print(f"  Seg output covered by Attn: {np.mean(c['seg_recall_by_attn']):.2%}")

    print(f"\n{'Lesion':<6} {'Seg Cov':>9} {'IoU(A,S)':>10} {'Enc-only':>10} {'Seg-in-Attn':>13}")
    print('-'*52)
    summary = {}
    for name in LESION_NAMES:
        d = accum[name]
        if not d['iou_attn_seg']:
            continue
        sc  = np.mean(d['seg_coverage'])
        iou = np.mean(d['iou_attn_seg'])
        eo  = np.mean(d['enc_only_area'])
        sia = np.mean(d['seg_recall_by_attn'])
        print(f"{name:<6} {sc:>9.2%} {iou:>10.4f} {eo:>10.2%} {sia:>13.2%}")
        summary[name] = {'seg_coverage': float(sc), 'iou': float(iou),
                         'enc_only': float(eo), 'seg_in_attn': float(sia)}

    with open(os.path.join(out_dir, 'summary.json'), 'w') as f:
        json.dump({'combined': {k: float(np.mean(v)) for k, v in c.items()},
                   'per_lesion': summary}, f, indent=2)
    print(f"\nSaved to {out_dir}/")


if __name__ == '__main__':
    main()
