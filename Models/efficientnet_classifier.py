"""
EfficientNet Baseline for DR Classification

使用 timm 预训练的 EfficientNet 做 DR 5分类 (或 3分类) baseline。
接口与 PARAGraph / SwinClassifier / DenseNet121Classifier 兼容:
    forward(images) -> {'logits', 'probs', 'predictions'}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetClassifier(nn.Module):
    """
    EfficientNet for DR grading (5-class or 3-class).

    Architecture:
        EfficientNet (ImageNet pretrained) -> Global Average Pooling -> Dropout -> Linear -> num_classes
    """

    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = 'efficientnet_b3',
        pretrained: bool = True,
        drop_rate: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            drop_rate=drop_rate,
        )

        num_features = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(drop_rate),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, images: torch.Tensor) -> dict:
        """
        Args:
            images: [B, 3, H, W]

        Returns:
            dict with logits, probs, predictions
        """
        features = self.backbone(images)
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
        }
