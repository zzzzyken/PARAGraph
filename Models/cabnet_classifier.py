"""
CABNet (Category Attention Block Network) Baseline for DR Classification

Based on: He et al., "CABNet: Category Attention Block for Imbalanced Diabetic Retinopathy Grading"
Uses ResNet-50 backbone + CBAM + Category Attention Block

接口与 PARAGraph / SwinClassifier / DenseNet121Classifier 兼容:
    forward(images) -> {'logits', 'probs', 'predictions'}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ============ CBAM (Convolutional Block Attention Module) ============

class _BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=not bn)
        self.bn = nn.BatchNorm2d(out_planes) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class _ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )

    def forward(self, x):
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), -1)
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), -1)
        scale = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool))
        return x * scale.unsqueeze(2).unsqueeze(3)


class _SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial = _BasicConv(2, 1, kernel_size=7, stride=1, padding=3, relu=False)

    def forward(self, x):
        x_compress = torch.cat([
            torch.max(x, 1)[0].unsqueeze(1),
            torch.mean(x, 1).unsqueeze(1)
        ], dim=1)
        scale = torch.sigmoid(self.spatial(x_compress))
        return x * scale


class _CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super().__init__()
        self.channel_gate = _ChannelGate(gate_channels, reduction_ratio)
        self.spatial_gate = _SpatialGate()

    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x


# ============ Category Attention Block ============

class _CABlock(nn.Module):
    """Category Attention Block — 学习类别特定的注意力权重"""
    def __init__(self, in_channels, num_classes, k=5):
        super().__init__()
        self.num_classes = num_classes
        self.k = k
        self.conv1 = nn.Conv2d(in_channels, num_classes * k, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_classes * k)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(0.1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        # x: [B, C, H, W]
        x1 = self.relu(self.bn(self.conv1(x)))     # [B, classes*k, H, W]

        x2 = self.dropout(x1)
        x2 = self.maxpool(x2).flatten(1)            # [B, classes*k]
        x2 = x2.view(x2.shape[0], self.num_classes, self.k)  # [B, classes, k]
        weight = torch.mean(x2, dim=-1)             # [B, classes]
        weight = weight.unsqueeze(-1).unsqueeze(-1)  # [B, classes, 1, 1]

        n, c, h, w = x1.size()
        res = x1.reshape(n, self.num_classes, self.k, h, w).sum(2)   # [B, classes, H, W]
        new = weight * res                                            # [B, classes, H, W]
        attention = new.reshape(n, 1, self.num_classes, h, w).sum(2)  # [B, 1, H, W]

        return attention * x  # [B, C, H, W]


# ============ CABNet Head (CBAM + Global Attention + CABlock + Linear) ============

class _CABNetHead(nn.Module):
    """CABNet 的完整分类头，包含 CBAM、全局注意力、类别注意力和最终分类器"""
    def __init__(self, in_features, num_classes, drop_rate, ca_k):
        super().__init__()
        hidden_dim = in_features // 2

        self.cbam = _CBAM(in_features)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Linear(in_features, hidden_dim)
        self.ca_block = _CABlock(hidden_dim, num_classes, k=ca_k)
        self.final_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, feat):
        # CBAM attention on backbone feature maps
        feat = self.cbam(feat)                                    # [B, C, h, w]

        # Global attention: pool → reduce dim
        pooled = self.global_pool(feat).flatten(1)                # [B, C]
        global_feat = self.global_fc(pooled)                      # [B, C//2]
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)     # [B, C//2, 1, 1]

        # Category attention
        ca_feat = self.ca_block(global_feat)                      # [B, C//2, 1, 1]
        ca_feat = self.final_pool(ca_feat).flatten(1)             # [B, C//2]

        # Classification
        logits = self.head(ca_feat)                               # [B, num_classes]
        return logits


# ============ CABNet Classifier ============

_BACKBONE_CONFIGS = {
    'resnet18':  (models.resnet18,  models.ResNet18_Weights.DEFAULT,  512),
    'resnet50':  (models.resnet50,  models.ResNet50_Weights.DEFAULT,  2048),
    'resnet101': (models.resnet101, models.ResNet101_Weights.DEFAULT, 2048),
}


class CABNetClassifier(nn.Module):
    """
    CABNet for DR grading (5-class or 3-class).

    Architecture:
        ResNet (ImageNet pretrained) -> CBAM -> Global Attention -> CABlock -> Classifier

    Reference:
        He et al., "CABNet: Category Attention Block for Imbalanced Diabetic Retinopathy Grading"
    """

    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = 'resnet50',
        pretrained: bool = True,
        drop_rate: float = 0.3,
        ca_k: int = 5,
    ):
        super().__init__()
        self.num_classes = num_classes

        if backbone_name not in _BACKBONE_CONFIGS:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose from {list(_BACKBONE_CONFIGS.keys())}")

        model_fn, weights, in_features = _BACKBONE_CONFIGS[backbone_name]
        resnet = model_fn(weights=weights if pretrained else None)

        # ResNet feature extractor (remove avgpool and fc)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )

        # CBAM + Global Attention + CABlock + Classifier (all under self.classifier)
        self.classifier = _CABNetHead(in_features, num_classes, drop_rate, ca_k)

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
        feat = self.backbone(images)                # [B, C, h, w]
        logits = self.classifier(feat)              # [B, num_classes]
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)

        return {
            'logits': logits,
            'probs': probs,
            'predictions': predictions,
        }
