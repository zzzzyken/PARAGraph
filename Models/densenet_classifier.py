"""
DenseNet-121 Baseline for DR Classification

使用 timm 预训练的 DenseNet-121 做 DR 5分类 (或 3分类) baseline。
接口与 PARAGraph / SwinClassifier 兼容: forward(images) -> {'logits', 'probs', 'predictions'}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class DenseNet121Classifier(nn.Module):
    """
    DenseNet-121 for DR grading (5-class or 3-class).

    Architecture:
        DenseNet-121 (ImageNet pretrained) -> Global Average Pooling -> Dropout -> Linear -> num_classes
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        # 创建 DenseNet-121 backbone (去掉原始分类头)
        self.backbone = timm.create_model(
            'densenet121',
            pretrained=pretrained,
            num_classes=0,       # 移除原始分类头, 输出 pooled features
            drop_rate=drop_rate,
        )

        # 获取 backbone 输出维度
        num_features = self.backbone.num_features  # densenet121: 1024

        # 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(drop_rate),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images: [B, 3, H, W] 归一化后的图像

        Returns:
            dict with:
                logits: [B, num_classes]
                probs: [B, num_classes]
                predictions: [B]
        """
        features = self.backbone(images)        # [B, num_features]
        logits = self.classifier(features)      # [B, num_classes]
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
        }
