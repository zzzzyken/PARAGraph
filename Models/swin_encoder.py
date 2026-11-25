"""
SwinEncoder: Swin Transformer encoder for UNet

将 timm 的 Swin-S 封装为 UNet encoder，通过 1x1 Conv 投影层
匹配 ResNet50 的通道维度，使 decoder 和 PARAGraph 无需修改。

Swin-S stages:
  Stage 0: 96ch,  H/4   → 投影到 stem (64ch) + stage1 (256ch)
  Stage 1: 192ch, H/8   → 投影到 stage2 (512ch)
  Stage 2: 384ch, H/16  → 投影到 stage3 (1024ch)
  Stage 3: 768ch, H/32  → 投影到 stage4 (2048ch)
"""

import torch
import torch.nn as nn
import timm


class SwinEncoder(nn.Module):
    """
    Swin Transformer encoder with channel projections to match ResNet50 outputs.

    Output interface identical to ResNet50 encoder:
        features = {
            'stem':   [B, 64,   H/4,  W/4],
            'stage1': [B, 256,  H/4,  W/4],
            'stage2': [B, 512,  H/8,  W/8],
            'stage3': [B, 1024, H/16, W/16],
            'stage4': [B, 2048, H/32, W/32],
        }
        bottleneck = features['stage4']
    """

    # ResNet50 target channel dimensions
    TARGET_CHANNELS = [64, 256, 512, 1024, 2048]

    def __init__(
        self,
        model_name: str = 'swin_tiny_patch4_window7_224',
        pretrained: bool = True,
        img_size: int = 224,
    ):
        """
        Args:
            model_name: timm model name (default: swin_tiny_patch4_window7_224)
            pretrained: load ImageNet pretrained weights
            img_size: input image size
        """
        super().__init__()

        # timm Swin backbone with multi-scale feature extraction
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )

        # Swin-S feature channels: [96, 192, 384, 768]
        swin_channels = self.backbone.feature_info.channels()

        # 1x1 Conv projections to match ResNet50 encoder channels
        # stem (64ch) 和 stage1 (256ch) 都从 Swin Stage 0 投影
        self.stem_proj = nn.Sequential(
            nn.Conv2d(swin_channels[0], self.TARGET_CHANNELS[0], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[0]),
            nn.ReLU(inplace=True),
        )
        self.stage1_proj = nn.Sequential(
            nn.Conv2d(swin_channels[0], self.TARGET_CHANNELS[1], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[1]),
            nn.ReLU(inplace=True),
        )
        self.stage2_proj = nn.Sequential(
            nn.Conv2d(swin_channels[1], self.TARGET_CHANNELS[2], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[2]),
            nn.ReLU(inplace=True),
        )
        self.stage3_proj = nn.Sequential(
            nn.Conv2d(swin_channels[2], self.TARGET_CHANNELS[3], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[3]),
            nn.ReLU(inplace=True),
        )
        self.stage4_proj = nn.Sequential(
            nn.Conv2d(swin_channels[3], self.TARGET_CHANNELS[4], 1),
            nn.BatchNorm2d(self.TARGET_CHANNELS[4]),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 3, H, W] input image

        Returns:
            features: dict with keys 'stem', 'stage1', 'stage2', 'stage3', 'stage4'
            bottleneck: [B, 2048, H/32, W/32]
        """
        # Swin multi-scale features: list of 4 tensors
        # timm Swin 可能输出 NHWC 格式，需要转为 NCHW
        swin_feats = self.backbone(x)
        s0, s1, s2, s3 = swin_feats

        # 如果是 NHWC (channels last)，转换为 NCHW
        if s0.ndim == 4 and s0.shape[-1] != s0.shape[1]:
            s0 = s0.permute(0, 3, 1, 2).contiguous()
            s1 = s1.permute(0, 3, 1, 2).contiguous()
            s2 = s2.permute(0, 3, 1, 2).contiguous()
            s3 = s3.permute(0, 3, 1, 2).contiguous()

        # Project to ResNet50 channel dimensions
        features = {
            'stem':   self.stem_proj(s0),    # [B, 64,   H/4,  W/4]
            'stage1': self.stage1_proj(s0),  # [B, 256,  H/4,  W/4]
            'stage2': self.stage2_proj(s1),  # [B, 512,  H/8,  W/8]
            'stage3': self.stage3_proj(s2),  # [B, 1024, H/16, W/16]
            'stage4': self.stage4_proj(s3),  # [B, 2048, H/32, W/32]
        }

        return features, features['stage4']
