"""
HACDRLesionUNet: HACDR-Net backbone (AAAI 2024) + lightweight decoder

Standalone PyTorch port — no mmcv/mmseg dependency.

DCNv2 (ModulatedDeformConv2dPack) replaced with torchvision.ops.DeformConv2d
+ learned offset/mask prediction conv.

Output interface identical to LesionUNet.encode():
    features = {
        'stage2': [B, 512,  H/8,  W/8]
        'stage3': [B, 1024, H/16, W/16]
        'stage4': [B, 2048, H/32, W/32]
        'stem':   [B, 64,   H/4,  W/4]   (from stage0, for decoder skip)
        'stage1': [B, 256,  H/4,  W/4]   (same as stem, for decoder skip)
    }
HACDR native dims [32, 64, 160, 256] projected to [512, 1024, 2048] via 1x1 Conv.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as tvops

try:
    from timm.layers import DropPath
except ImportError:
    from timm.models.layers import DropPath


# ---------------------------------------------------------------------------
# Deformable Conv (standalone, replaces mmcv ModulatedDeformConv2dPack)
# ---------------------------------------------------------------------------

class DeformConv2dPack(nn.Module):
    """Modulated deformable conv with self-predicted offset + mask."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, deform_groups=1, bias=True):
        super().__init__()
        if isinstance(kernel_size, int):
            kh = kw = kernel_size
        else:
            kh, kw = kernel_size
        self.offset_mask_conv = nn.Conv2d(
            in_channels,
            3 * deform_groups * kh * kw,
            kernel_size=(kh, kw), stride=stride, padding=padding
        )
        self.dcn = tvops.DeformConv2d(
            in_channels, out_channels,
            kernel_size=(kh, kw), stride=stride, padding=padding,
            groups=groups, bias=bias
        )
        nn.init.zeros_(self.offset_mask_conv.weight)
        nn.init.zeros_(self.offset_mask_conv.bias)

    def forward(self, x):
        out = self.offset_mask_conv(x)
        n = out.shape[1] // 3
        offset = out[:, :2 * n]
        mask = torch.sigmoid(out[:, 2 * n:])
        return self.dcn(x, offset, mask)


# ---------------------------------------------------------------------------
# HACDR building blocks (ported from mmseg/models/backbones/hacdr.py)
# ---------------------------------------------------------------------------

class DFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1,
                                bias=True, groups=hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class StemConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU(),
            nn.Conv2d(out_channels // 2, out_channels, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.size()
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class HAAAttention(nn.Module):
    def __init__(self, channels,
                 kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        # conv0: depthwise DCNv2 (deform_groups=1, groups=channels)
        self.conv0 = DeformConv2dPack(
            channels, channels,
            kernel_size=kernel_sizes[0],
            padding=paddings[0],
            groups=channels,
            deform_groups=1,
        )
        for i, (ks, pad) in enumerate(zip(kernel_sizes[1:], paddings[1:])):
            for suffix, k, p in [('_1', ks, pad), ('_2', ks[::-1], pad[::-1])]:
                self.add_module(
                    f'conv{i}{suffix}',
                    nn.Conv2d(channels, channels, tuple(k), padding=p, groups=channels)
                )
        self.conv3 = nn.Conv2d(channels, channels, 1)
        # conv3_1/2: deform_groups=channels (each channel has its own offset field)
        self.conv3_1 = DeformConv2dPack(channels, channels, (1, 3), padding=(0, 1),
                                         deform_groups=channels)
        self.conv3_2 = DeformConv2dPack(channels, channels, (3, 1), padding=(1, 0),
                                         deform_groups=channels)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn_0 = self.conv0_2(self.conv0_1(attn))
        attn_1 = self.conv1_2(self.conv1_1(attn))
        attn_2 = self.conv2_2(self.conv2_1(attn))
        attn_3 = self.conv3_2(self.conv3_1(attn))
        attn = attn + attn_0 + attn_1 + attn_2 + attn_3
        attn = self.conv3(attn)
        return attn * u


class HAASpatialAttention(nn.Module):
    def __init__(self, in_channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.proj_1 = nn.Conv2d(in_channels, in_channels, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = HAAAttention(
            in_channels, attention_kernel_sizes, attention_kernel_paddings)
        self.proj_2 = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        return x + shortcut


class HCABlock(nn.Module):
    def __init__(self, channels,
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]],
                 dffn_ratio=4., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(channels)
        self.attn = HAASpatialAttention(channels, attention_kernel_sizes,
                                        attention_kernel_paddings)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(channels)
        self.dffn = DFFN(channels, int(channels * dffn_ratio), drop=drop)
        self.layer_scale_1 = nn.Parameter(1e-2 * torch.ones(channels), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(1e-2 * torch.ones(channels), requires_grad=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.permute(0, 2, 1).view(B, C, H, W)
        x = x + self.drop_path(
            self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(
            self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.dffn(self.norm2(x)))
        x = x.view(B, C, N).permute(0, 2, 1)
        return x


class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=2, in_channels=3, embed_dim=64):
        super().__init__()
        self.proj = DeformConv2dPack(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride,
            padding=patch_size // 2, deform_groups=2,
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        x = x.flatten(2).transpose(1, 2)
        return x, H, W


class HACDRBackbone(nn.Module):
    """
    HACDR backbone — outputs 4-scale features.

    Default config matches HACDR_idrid.py:
        embed_dims=[32, 64, 160, 256], depths=[3, 3, 5, 2]
    """

    def __init__(self,
                 embed_dims=[32, 64, 160, 256],
                 dffn_ratios=[8, 8, 4, 4],
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 depths=[3, 3, 5, 2],
                 attention_kernel_sizes=[5, [1, 7], [1, 11], [1, 21]],
                 attention_kernel_paddings=[2, [0, 3], [0, 5], [0, 10]]):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims
        num_stages = len(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0

        for i in range(num_stages):
            if i == 0:
                patch_embed = StemConv(3, embed_dims[0])
            else:
                patch_embed = OverlapPatchEmbed(
                    patch_size=3, stride=2,
                    in_channels=embed_dims[i - 1],
                    embed_dim=embed_dims[i],
                )
            block = nn.ModuleList([
                HCABlock(
                    channels=embed_dims[i],
                    attention_kernel_sizes=attention_kernel_sizes,
                    attention_kernel_paddings=attention_kernel_paddings,
                    dffn_ratio=dffn_ratios[i],
                    drop=drop_rate,
                    drop_path=dpr[cur + j],
                ) for j in range(depths[i])
            ])
            norm = nn.LayerNorm(embed_dims[i])
            cur += depths[i]
            setattr(self, f'patch_embed{i + 1}', patch_embed)
            setattr(self, f'block{i + 1}', block)
            setattr(self, f'norm{i + 1}', norm)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                nn.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]
        outs = []
        for i in range(len(self.depths)):
            patch_embed = getattr(self, f'patch_embed{i + 1}')
            block = getattr(self, f'block{i + 1}')
            norm = getattr(self, f'norm{i + 1}')
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)
        return outs  # [stage0, stage1, stage2, stage3]


# ---------------------------------------------------------------------------
# Decoder (lightweight, mirrors LesionUNet's slim decoder)
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=True):
        super().__init__()
        layers = []
        if upsample:
            layers += [nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
                       nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers += [nn.Conv2d(in_channels, out_channels, 1),
                       nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(out_channels, out_channels, 3, padding=1),
                   nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ---------------------------------------------------------------------------
# HACDRLesionUNet — drop-in replacement for LesionUNet
# ---------------------------------------------------------------------------

class HACDRLesionUNet(nn.Module):
    """
    HACDR backbone + lightweight UNet decoder for DR lesion segmentation.

    Interface identical to LesionUNet:
        encode(x) → (features_dict, bottleneck)
        decode(features) → masks [B, num_lesions, H, W]
        forward(x, mode='full') → dict

    features_dict keys match PARAGraph expectations:
        'stem':   [B, 64,   H/4,  W/4]
        'stage1': [B, 256,  H/4,  W/4]
        'stage2': [B, 512,  H/8,  W/8]
        'stage3': [B, 1024, H/16, W/16]
        'stage4': [B, 2048, H/32, W/32]
    """

    LESION_NAMES = ['MA', 'HE', 'EX', 'SE']

    # HACDR native output dims per stage (default config)
    HACDR_DIMS = [32, 64, 160, 256]

    # Target dims expected by PARAGraph (matches old LesionUNet)
    TARGET_CHANNELS = [64, 256, 512, 1024, 2048]

    def __init__(self,
                 num_lesions=4,
                 encoder_name='hacdr',   # kept for API compat, ignored
                 embed_dims=[32, 64, 160, 256],
                 dffn_ratios=[8, 8, 4, 4],
                 depths=[3, 3, 5, 2],
                 drop_rate=0.1,
                 drop_path_rate=0.1,
                 **kwargs):
        super().__init__()
        self.num_lesions = num_lesions
        self.encoder_name = 'hacdr'
        self.encoder_channels = self.TARGET_CHANNELS

        # ---- Backbone ----
        self.hacdr = HACDRBackbone(
            embed_dims=embed_dims,
            dffn_ratios=dffn_ratios,
            depths=depths,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        # ---- Projection layers to match PARAGraph-expected feature dims ----
        # stage0 (embed_dims[0], H/4) → stem (64) + stage1 (256)
        self.stem_proj = nn.Sequential(
            nn.Conv2d(embed_dims[0], self.TARGET_CHANNELS[0], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[0]), nn.ReLU(inplace=True))
        self.stage1_proj = nn.Sequential(
            nn.Conv2d(embed_dims[0], self.TARGET_CHANNELS[1], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[1]), nn.ReLU(inplace=True))
        # stage1 (embed_dims[1], H/8) → stage2 (512)
        self.stage2_proj = nn.Sequential(
            nn.Conv2d(embed_dims[1], self.TARGET_CHANNELS[2], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[2]), nn.ReLU(inplace=True))
        # stage2 (embed_dims[2], H/16) → stage3 (1024)
        self.stage3_proj = nn.Sequential(
            nn.Conv2d(embed_dims[2], self.TARGET_CHANNELS[3], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[3]), nn.ReLU(inplace=True))
        # stage3 (embed_dims[3], H/32) → stage4 (2048)
        self.stage4_proj = nn.Sequential(
            nn.Conv2d(embed_dims[3], self.TARGET_CHANNELS[4], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[4]), nn.ReLU(inplace=True))

        # ---- Decoder (lightweight [256, 128, 64, 32]) ----
        self.decoder_stage4 = DecoderBlock(embed_dims[3], 256)                    # /32→/16
        self.decoder_stage3 = DecoderBlock(256 + embed_dims[2], 128)              # /16→/8
        self.decoder_stage2 = DecoderBlock(128 + embed_dims[1], 64)               # /8→/4
        self.decoder_stage1 = DecoderBlock(64 + embed_dims[0], 32, upsample=False)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.segmentation_head = nn.Conv2d(32, num_lesions, 1)

        # Bottleneck projection (for downstream use, mirrors LesionUNet)
        self.bottleneck_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Conv2d(embed_dims[3], 512, 1),
            nn.BatchNorm2d(512), nn.ReLU(inplace=True))

    def encode(self, x):
        """
        Returns:
            features: dict matching PARAGraph expectations
            bottleneck: [B, 2048, H/32, W/32]
        """
        s0, s1, s2, s3 = self.hacdr(x)  # native HACDR dims

        features = {
            'stem':   self.stem_proj(s0),    # [B, 64,   H/4,  W/4]
            'stage1': self.stage1_proj(s0),  # [B, 256,  H/4,  W/4]
            'stage2': self.stage2_proj(s1),  # [B, 512,  H/8,  W/8]
            'stage3': self.stage3_proj(s2),  # [B, 1024, H/16, W/16]
            'stage4': self.stage4_proj(s3),  # [B, 2048, H/32, W/32]
        }
        bottleneck = features['stage4']
        # store native stages for decoder skip connections
        self._native = (s0, s1, s2, s3)
        return features, bottleneck

    def decode(self, features):
        s0, s1, s2, s3 = self._native
        d4 = self.decoder_stage4(s3)                               # [B, 256, H/16]
        d3 = self.decoder_stage3(torch.cat([d4, s2], dim=1))      # [B, 128, H/8]
        d2 = self.decoder_stage2(torch.cat([d3, s1], dim=1))      # [B, 64,  H/4]
        d1 = self.decoder_stage1(torch.cat([d2, s0], dim=1))      # [B, 32,  H/4]
        out = self.final_upsample(d1)                              # [B, 32,  H]
        return self.segmentation_head(out)                         # [B, num_lesions, H, W]

    def forward(self, x, mode='full'):
        features, bottleneck = self.encode(x)
        if mode == 'full':
            masks = self.decode(features)
            return {'masks': masks, 'bottleneck': bottleneck, 'features': features}
        elif mode == 'encode':
            patch_features = self.bottleneck_proj(bottleneck)
            B, C, H, W = patch_features.shape
            patch_features = patch_features.flatten(2).transpose(1, 2)
            return {'patch_features': patch_features, 'bottleneck': bottleneck,
                    'features': features}
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_segmentation_loss(self, pred_masks, gt_masks, class_weights=None):
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(gt_masks.float(), size=pred_masks.shape[-2:], mode='nearest')
        bce = F.binary_cross_entropy_with_logits(pred_masks, gt_masks, reduction='none')
        if class_weights is not None:
            bce = bce * class_weights.view(1, -1, 1, 1).to(bce.device)
        bce = bce.mean()
        pred_probs = torch.sigmoid(pred_masks)
        p = pred_probs.flatten(2)
        t = gt_masks.flatten(2)
        dice = 1 - ((2 * (p * t).sum(2) + 1e-6) / (p.sum(2) + t.sum(2) + 1e-6)).mean()
        return {'total': bce + dice, 'bce': bce, 'dice': dice}

    def _dice_loss(self, pred, target, smooth=1e-6):
        pred = pred.flatten(2)
        target = target.flatten(2)
        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)
        return (1 - ((2 * intersection + smooth) / (union + smooth))).mean()

    def _freeze_encoder(self):
        for p in self.hacdr.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.hacdr.parameters():
            p.requires_grad = True
