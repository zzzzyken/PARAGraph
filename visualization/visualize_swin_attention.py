"""
可视化 LesionUNet Swin-T encoder 各层的 window attention 热力图

对每张图，提取所有 12 个 attention block 的权重，
重建成全图注意力热力图，看 Swin 实际在关注哪些区域。

用法:
    python visualization/visualize_swin_attention.py --img_dir path/to/images --ext png --n_imgs 10
"""

import _bootstrap  # noqa: F401  (adds the repository root to sys.path)
import os, sys, argparse
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models.lesion_unet import LesionUNet

CHECKPOINT = "./checkpoints/lesion_unet_swint/best_model.pth"
IMG_SIZE   = 1024
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])
WIN_SIZE   = 7   # Swin-T default window size


def load_model(imagenet_only=False):
    if imagenet_only:
        # 纯 ImageNet pretrained Swin，不加载分割 checkpoint
        model = LesionUNet(
            num_lesions=4, encoder_name='swin',
            swin_model_name='swin_tiny_patch4_window7_224',
            img_size=IMG_SIZE, encoder_pretrained=True,
        ).to(DEVICE)
        print("Loaded: ImageNet pretrained Swin-T (no segmentation fine-tuning)")
    else:
        model = LesionUNet(
            num_lesions=4, encoder_name='swin',
            swin_model_name='swin_tiny_patch4_window7_224',
            img_size=IMG_SIZE, encoder_pretrained=False,
        ).to(DEVICE)
        ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded: Segmentation pretrained LesionUNet ({CHECKPOINT})")
    model.eval()
    return model


def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)


def reconstruct_attention_map(attn_tensor, feat_h, feat_w, win_size=WIN_SIZE):
    """
    将 window attention 权重重建为全图热力图。

    attn_tensor: [B*nW, nH, N, N]  N = win_size^2
    feat_h/feat_w: 实际 feature map 尺寸（可能不被 win_size 整除）

    Returns: [feat_h, feat_w] numpy array (for batch_idx=0)
    """
    nW_total, nH, N, _ = attn_tensor.shape
    win_h = win_w = win_size

    # 实际 window 网格大小（ceil 保证覆盖全图，与 timm 一致）
    import math
    nW_h = math.ceil(feat_h / win_h)
    nW_w = math.ceil(feat_w / win_w)
    nW   = nW_h * nW_w

    # 只取 batch 0 的 nW 个窗口
    if nW > nW_total:
        raise ValueError(f"Need {nW} windows but only {nW_total} available")
    attn = attn_tensor[:nW]              # [nW, nH, N, N]
    attn = attn.mean(dim=1)              # [nW, N, N] 平均多头
    attn = attn.mean(dim=1)              # [nW, N]    每个 key token 的平均被关注度

    # reshape: [nW, N] → [nW_h, nW_w, win_h, win_w] → [padded_h, padded_w]
    attn = attn.view(nW_h, nW_w, win_h, win_w)
    attn = attn.permute(0, 2, 1, 3).contiguous()
    padded_h, padded_w = nW_h * win_h, nW_w * win_w
    attn_map = attn.reshape(padded_h, padded_w).cpu().numpy()

    # 裁剪回实际 feature 尺寸
    attn_map = attn_map[:feat_h, :feat_w]

    # 归一化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


def get_swin_attention_maps(model, tensor):
    """
    注册 hook，运行 forward，返回所有 attn block 的热力图。

    Returns:
        maps: list of (layer_name, attn_map [H_feat, W_feat])
    """
    swin_backbone = model.swin_encoder.backbone

    captured = {}   # name → (attn_weights, feat_h, feat_w)

    hooks = []
    for name, mod in swin_backbone.named_modules():
        if 'attn' in name and hasattr(mod, 'num_heads'):
            def make_hook(n):
                def hook_fn(m, inp, out):
                    # out: attention output tensor — we need the weights
                    # timm WindowAttention returns (x,) not (x, attn)
                    # Use the stored attn via forward re-hook on softmax
                    pass
                return hook_fn
            # timm's WindowAttention stores attn internally when attn_drop forward runs
            # Better: hook on the softmax output inside forward
            # Simplest: monkey-patch to return attn weights
            hooks.append((name, mod))

    # Cleaner approach: patch each WindowAttention.forward to also save attn
    original_forwards = {}
    attn_store = {}

    for name, mod in swin_backbone.named_modules():
        if 'attn' in name and hasattr(mod, 'num_heads') and 'attn' == name.split('.')[-1]:
            original_forwards[name] = mod.forward
            def make_patched(n, m, orig_fwd):
                def patched_forward(x, mask=None):
                    B_, N, C = x.shape
                    qkv = m.qkv(x).reshape(B_, N, 3, m.num_heads, C // m.num_heads).permute(2, 0, 3, 1, 4)
                    q, k, v = qkv.unbind(0)
                    q = q * m.scale
                    attn = q @ k.transpose(-2, -1)
                    relative_position_bias = m.relative_position_bias_table[
                        m.relative_position_index.view(-1)
                    ].view(m.window_size[0] * m.window_size[1],
                           m.window_size[0] * m.window_size[1], -1)
                    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
                    attn = attn + relative_position_bias.unsqueeze(0)
                    if mask is not None:
                        nW = mask.shape[0]
                        attn = attn.view(B_ // nW, nW, m.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
                        attn = attn.view(-1, m.num_heads, N, N)
                    attn_softmax = attn.softmax(dim=-1)
                    attn_store[n] = attn_softmax.detach().cpu()   # save
                    attn_drop = m.attn_drop(attn_softmax)
                    x = (attn_drop @ v).transpose(1, 2).reshape(B_, N, C)
                    x = m.proj(x)
                    x = m.proj_drop(x)
                    return x
                return patched_forward
            mod.forward = make_patched(name, mod, original_forwards[name])

    # Forward pass
    H, W = tensor.shape[-2:]
    with torch.no_grad():
        swin_feats = swin_backbone(tensor)

    # Restore original forwards
    for name, mod in swin_backbone.named_modules():
        if name in original_forwards:
            mod.forward = original_forwards[name]

    # Compute feature spatial sizes per stage
    # Swin-T: patch_embed → H/4, stage0 → H/4, stage1 → H/8, stage2 → H/16, stage3 → H/32
    stage_sizes = {
        'layers_0': (H // 4,  W // 4),
        'layers_1': (H // 8,  W // 8),
        'layers_2': (H // 16, W // 16),
        'layers_3': (H // 32, W // 32),
    }

    maps = []
    for key in sorted(attn_store.keys()):
        stage = key.split('.')[0]  # e.g. 'layers_0'
        if stage not in stage_sizes:
            continue
        fh, fw = stage_sizes[stage]
        attn_t = attn_store[key]  # [B*nW, nH, N, N]
        try:
            amap = reconstruct_attention_map(attn_t, fh, fw)
            maps.append((key, amap, fh, fw))
        except Exception as e:
            print(f"  skip {key}: {e}")

    return maps


def visualize_attention(img_id, img_bgr, attn_maps, out_path, title_suffix=''):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W = img_rgb.shape[:2]

    n = len(attn_maps)
    cols = 4
    rows = (n + 1 + cols - 1) // cols  # +1 for original

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = axes.flatten()

    # Original
    axes[0].imshow(img_rgb)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for i, (name, amap, fh, fw) in enumerate(attn_maps):
        ax = axes[i + 1]
        amap_up = cv2.resize(amap, (W, H), interpolation=cv2.INTER_LINEAR)
        ax.imshow(img_rgb, alpha=0.4)
        ax.imshow(amap_up, cmap='hot', alpha=0.65, vmin=0, vmax=1)
        stage = name.split('.')[0].replace('layers_', 'S')
        block = name.split('.')[2]
        ax.set_title(f'{stage}-B{block}  [{fh}×{fw}]', fontsize=9)
        ax.axis('off')

    for j in range(len(attn_maps) + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'{img_id} — Swin-T Window Attention Maps{title_suffix}', fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=90, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',    required=True, help='Folder with fundus images')
    parser.add_argument('--ext',        default='png')
    parser.add_argument('--n_imgs',     type=int, default=10)
    parser.add_argument('--out_dir',    default='./swin_attention_maps')
    parser.add_argument('--imagenet',   action='store_true',
                        help='Use ImageNet pretrained Swin (no segmentation fine-tuning)')
    args = parser.parse_args()

    suffix = '_imagenet' if args.imagenet else '_seg'
    out_dir = args.out_dir + suffix
    args.out_dir = out_dir
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(imagenet_only=args.imagenet)
    print("Model loaded.")

    img_files = sorted(Path(args.img_dir).glob(f'*.{args.ext}'))[:args.n_imgs]
    print(f"Processing {len(img_files)} images...")

    for img_path in img_files:
        img_id  = img_path.stem
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        tensor = preprocess(img_bgr)
        print(f"  {img_id}...")
        try:
            attn_maps = get_swin_attention_maps(model, tensor)
            out_path  = os.path.join(args.out_dir, f"{img_id}_swin_attn.png")
            title_suffix = ' [ImageNet]' if args.imagenet else ' [Seg-pretrained]'
            visualize_attention(img_id, img_bgr, attn_maps, out_path, title_suffix=title_suffix)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()


if __name__ == '__main__':
    main()
