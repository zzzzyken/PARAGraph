"""
LesionUNet: UNet结构用于病变分割

预训练阶段: 完整UNet做病变分割
下游阶段: 只用Encoder提取特征

支持 ResNet50 或 Swin Transformer encoder (通过 encoder_name 参数切换)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class LesionUNet(nn.Module):
    """
    UNet结构

    预训练阶段: 完整UNet做病变分割
    下游阶段: 只用Encoder提取特征
    """

    LESION_NAMES = ['MA', 'HE', 'EX', 'SE']

    def __init__(
        self,
        num_lesions=4,
        encoder_pretrained=True,
        encoder_freeze=False,
        encoder_name='swin',  # 'resnet50' or 'swin'
        swin_model_name='swin_tiny_patch4_window7_224',
        img_size=1024,
        large_decoder=False,  # True = Jan checkpoint layout (1024/512/256/64)
    ):
        super().__init__()

        self.num_lesions = num_lesions
        self.encoder_name = encoder_name

        # Encoder通道数 (两种 encoder 投影后维度一致)
        self.encoder_channels = [64, 256, 512, 1024, 2048]

        # ============ Encoder ============
        if encoder_name == 'swin':
            from .swin_encoder import SwinEncoder
            self.swin_encoder = SwinEncoder(
                model_name=swin_model_name,
                pretrained=encoder_pretrained,
                img_size=img_size,
            )
        else:
            # ResNet50 encoder
            weights = ResNet50_Weights.IMAGENET1K_V1 if encoder_pretrained else None
            resnet = models.resnet50(weights=weights)

            # 分解ResNet为多个stage，方便skip connection
            self.encoder_stem = nn.Sequential(
                resnet.conv1,      # 3 -> 64, /2
                resnet.bn1,
                resnet.relu,
                resnet.maxpool     # /2, 总共/4
            )  # Output: [B, 64, H/4, W/4]

            self.encoder_stage1 = resnet.layer1  # 64 -> 256, /4
            self.encoder_stage2 = resnet.layer2  # 256 -> 512, /8
            self.encoder_stage3 = resnet.layer3  # 512 -> 1024, /16
            self.encoder_stage4 = resnet.layer4  # 1024 -> 2048, /32

        if encoder_freeze:
            self._freeze_encoder()

        # ============ Decoder ============
        if large_decoder:
            # Jan 2026 checkpoint layout: wider decoder
            self.decoder_stage4 = self._make_decoder_block(2048, 1024)
            self.decoder_stage3 = self._make_decoder_block(1024 + 1024, 512)
            self.decoder_stage2 = self._make_decoder_block(512 + 512, 256)
            self.decoder_stage1 = self._make_decoder_block(256 + 256, 64, up=False)
            self.final_upsample = nn.Sequential(
                nn.ConvTranspose2d(64 + 64, 64, kernel_size=4, stride=4),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            )
            self.segmentation_head = nn.Conv2d(64, num_lesions, kernel_size=1)
        else:
            # Current (smaller) decoder layout
            self.decoder_stage4 = self._make_decoder_block(2048, 256)
            self.decoder_stage3 = self._make_decoder_block(256 + 1024, 128)
            self.decoder_stage2 = self._make_decoder_block(128 + 512, 64)
            self.decoder_stage1 = self._make_decoder_block(64 + 256, 32, up=False)
            self.final_upsample = nn.Sequential(
                nn.ConvTranspose2d(32 + 64, 32, kernel_size=4, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
            self.segmentation_head = nn.Conv2d(32, num_lesions, kernel_size=1)

        # ============ Bottleneck Projection (用于下游) ============
        # 将bottleneck特征投影为patch-level特征
        self.bottleneck_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),  # 确保输出7x7
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def _make_decoder_block(self, in_channels, out_channels, up=True):
        """解码器块: 上采样 + 卷积"""
        layers = []
        if up:
            layers += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        else:
            # 不改变空间尺寸，只做通道融合
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]

        layers += [
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        return nn.Sequential(*layers)


    def _freeze_encoder(self):
        """冻结encoder参数"""
        if hasattr(self, 'swin_encoder'):
            for param in self.swin_encoder.parameters():
                param.requires_grad = False
        else:
            for module in [self.encoder_stem, self.encoder_stage1,
                           self.encoder_stage2, self.encoder_stage3,
                           self.encoder_stage4]:
                for param in module.parameters():
                    param.requires_grad = False

    def unfreeze_encoder(self):
        """解冻encoder参数"""
        if hasattr(self, 'swin_encoder'):
            for param in self.swin_encoder.parameters():
                param.requires_grad = True
        else:
            for module in [self.encoder_stem, self.encoder_stage1,
                           self.encoder_stage2, self.encoder_stage3,
                           self.encoder_stage4]:
                for param in module.parameters():
                    param.requires_grad = True

    def encode(self, x):
        """
        只运行encoder部分，返回多尺度特征

        Returns:
            features: dict of multi-scale features
                - 'stem':   [B, 64,   H/4,  W/4]
                - 'stage1': [B, 256,  H/4,  W/4]
                - 'stage2': [B, 512,  H/8,  W/8]
                - 'stage3': [B, 1024, H/16, W/16]
                - 'stage4': [B, 2048, H/32, W/32]
            bottleneck: [B, 2048, H/32, W/32]
        """
        if hasattr(self, 'swin_encoder'):
            return self.swin_encoder(x)

        # Encoder forward with skip connections
        e0 = self.encoder_stem(x)      # [B, 64, H/4, W/4]
        e1 = self.encoder_stage1(e0)   # [B, 256, H/4, W/4]
        e2 = self.encoder_stage2(e1)   # [B, 512, H/8, W/8]
        e3 = self.encoder_stage3(e2)   # [B, 1024, H/16, W/16]
        e4 = self.encoder_stage4(e3)   # [B, 2048, H/32, W/32]

        features = {
            'stem': e0,
            'stage1': e1,
            'stage2': e2,
            'stage3': e3,
            'stage4': e4  # bottleneck
        }

        return features, e4

    def decode(self, features, return_decoder_feats=False):
        """
        运行decoder部分，生成分割mask

        Args:
            features: encoder feature dict
            return_decoder_feats: 若 True，同时返回 decoder 中间特征
                - 'dec_stage2': [B, 64,  H/4,  W/4]  ← MA 用
                - 'dec_stage3': [B, 128, H/8,  W/8]  ← HE/EX/SE 用

        Returns:
            masks: [B, num_lesions, H, W]
            decoder_feats (optional): dict with 'dec_stage2', 'dec_stage3'
        """
        e0, e1, e2, e3, e4 = (
            features['stem'], features['stage1'],
            features['stage2'], features['stage3'], features['stage4']
        )

        # Decoder with skip connections
        d4 = self.decoder_stage4(e4)                          # [B, 256, H/16, W/16]
        d3 = self.decoder_stage3(torch.cat([d4, e3], dim=1))  # [B, 128, H/8,  W/8]
        d2 = self.decoder_stage2(torch.cat([d3, e2], dim=1))  # [B, 64,  H/4,  W/4]
        d1 = self.decoder_stage1(torch.cat([d2, e1], dim=1))  # [B, 32,  H/4,  W/4]

        # Final upsample
        out = self.final_upsample(torch.cat([d1, e0], dim=1))  # [B, 32, H, W]

        # Segmentation head
        masks = self.segmentation_head(out)  # [B, num_lesions, H, W]

        if return_decoder_feats:
            return masks, {'dec_stage2': d2, 'dec_stage3': d3}
        return masks

    def forward(self, x, mode='full'):
        """
        Args:
            x: [B, 3, H, W] input image
            mode:
                'full' - 完整UNet (用于预训练)
                'encode' - 只encoder (用于下游任务)

        Returns:
            根据mode返回不同输出
        """
        features, bottleneck = self.encode(x)

        if mode == 'full':
            # 预训练模式: 返回分割mask
            masks = self.decode(features)
            return {
                'masks': masks,                    # [B, 4, H, W]
                'bottleneck': bottleneck,          # [B, 2048, H/32, W/32]
                'features': features
            }

        elif mode == 'encode':
            # 下游模式: 返回patch特征
            patch_features = self.bottleneck_proj(bottleneck)  # [B, 512, 7, 7]

            # 展平为patch序列
            B, C, H, W = patch_features.shape
            patch_features = patch_features.flatten(2).transpose(1, 2)  # [B, 49, 512]

            return {
                'patch_features': patch_features,  # [B, 49, 512]
                'bottleneck': bottleneck,
                'features': features
            }

        else:
            raise ValueError(f"Unknown mode: {mode}")

    def compute_segmentation_loss(self, pred_masks, gt_masks, class_weights=None):
        """
        计算分割损失

        Args:
            pred_masks: [B, 4, H, W] 预测的mask logits
            gt_masks: [B, 4, H, W] ground truth masks
            class_weights: [4] 每种lesion的权重 (处理类别不平衡)
        """
        # Resize gt to match pred if needed
        if pred_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = F.interpolate(
                gt_masks.float(),
                size=pred_masks.shape[-2:],
                mode='nearest'
            )

        # BCE + Dice Loss 组合
        bce_loss = F.binary_cross_entropy_with_logits(
            pred_masks, gt_masks, reduction='none'
        )

        if class_weights is not None:
            # [4] -> [1, 4, 1, 1]
            weights = class_weights.view(1, -1, 1, 1).to(bce_loss.device)
            bce_loss = bce_loss * weights

        bce_loss = bce_loss.mean()

        # Dice loss
        pred_probs = torch.sigmoid(pred_masks)
        dice_loss = self._dice_loss(pred_probs, gt_masks)

        total_loss = bce_loss + dice_loss

        return {
            'total': total_loss,
            'bce': bce_loss,
            'dice': dice_loss
        }

    def _dice_loss(self, pred, target, smooth=1e-6):
        """Dice Loss for segmentation"""
        pred = pred.flatten(2)    # [B, C, H*W]
        target = target.flatten(2)

        intersection = (pred * target).sum(dim=2)
        union = pred.sum(dim=2) + target.sum(dim=2)

        dice = (2 * intersection + smooth) / (union + smooth)

        return 1 - dice.mean()
