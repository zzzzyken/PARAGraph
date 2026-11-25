"""
OpticUNet: UNet结构用于Optic Disc分割和Fovea定位

双任务模型:
- Optic Disc 分割 (输出 mask)
- Fovea Center 定位 (输出 heatmap)

Encoder: ResNet-50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet50_Weights


class OpticUNet(nn.Module):
    """
    双任务UNet结构

    共享 Encoder，两个独立 Decoder:
    - Optic Disc Decoder: 输出分割 mask
    - Fovea Decoder: 输出定位 heatmap
    """

    def __init__(
        self,
        encoder_pretrained=True,
        encoder_freeze=False,
        predict_fovea=True,  # 是否预测 Fovea
        encoder_name='resnet50',  # ResNet-50 is the only encoder
        img_size=1024,  # kept for call compatibility; the ResNet encoder is size-agnostic
    ):
        super().__init__()
        if encoder_name != 'resnet50':
            raise ValueError(
                f"{type(self).__name__} supports only the ResNet-50 encoder "
                f"(got encoder_name={encoder_name!r}).")

        self.predict_fovea = predict_fovea
        self.encoder_name = encoder_name

        # Encoder通道数
        self.encoder_channels = [64, 256, 512, 1024, 2048]

        # ============ Encoder ============
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

        # ============ Optic Disc Decoder ============
        # Decoder widths [256, 128, 64, 32] — encoder skip channels [1024, 512, 256, 64]
        self.od_decoder_stage4 = self._make_decoder_block(2048, 256)           # /32 -> /16
        self.od_decoder_stage3 = self._make_decoder_block(256 + 1024, 128)     # /16 -> /8
        self.od_decoder_stage2 = self._make_decoder_block(128 + 512, 64)       # /8  -> /4
        self.od_decoder_stage1 = self._make_decoder_block(64 + 256, 32, up=False)  # /4 -> /4

        self.od_final_upsample = nn.Sequential(
            nn.ConvTranspose2d(32 + 64, 32, kernel_size=4, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # Optic Disc 分割头
        self.od_head = nn.Conv2d(32, 1, kernel_size=1)

        # ============ Fovea Decoder ============
        if self.predict_fovea:
            self.fovea_decoder_stage4 = self._make_decoder_block(2048, 256)          # /32 -> /16
            self.fovea_decoder_stage3 = self._make_decoder_block(256 + 1024, 128)    # /16 -> /8
            self.fovea_decoder_stage2 = self._make_decoder_block(128 + 512, 64)      # /8  -> /4
            self.fovea_decoder_stage1 = self._make_decoder_block(64 + 256, 32, up=False)  # /4 -> /4

            self.fovea_final_upsample = nn.Sequential(
                nn.ConvTranspose2d(32 + 64, 32, kernel_size=4, stride=4),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )

            # Fovea heatmap 头
            self.fovea_head = nn.Conv2d(32, 1, kernel_size=1)

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
        for module in [self.encoder_stem, self.encoder_stage1,
                       self.encoder_stage2, self.encoder_stage3,
                       self.encoder_stage4]:
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_encoder(self):
        """解冻encoder参数"""
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

    def decode_optic_disc(self, features):
        """
        Optic Disc decoder，生成分割 mask
        """
        e0, e1, e2, e3, e4 = (
            features['stem'], features['stage1'],
            features['stage2'], features['stage3'], features['stage4']
        )

        d4 = self.od_decoder_stage4(e4)                          # [B, 256, H/16, W/16]
        d3 = self.od_decoder_stage3(torch.cat([d4, e3], dim=1))  # [B, 128, H/8, W/8]
        d2 = self.od_decoder_stage2(torch.cat([d3, e2], dim=1))  # [B, 64,  H/4, W/4]
        d1 = self.od_decoder_stage1(torch.cat([d2, e1], dim=1))  # [B, 32,  H/4, W/4]

        out = self.od_final_upsample(torch.cat([d1, e0], dim=1))  # [B, 32, H, W]
        mask = self.od_head(out)  # [B, 1, H, W]

        return mask

    def decode_fovea(self, features):
        """
        Fovea decoder，生成定位 heatmap
        """
        e0, e1, e2, e3, e4 = (
            features['stem'], features['stage1'],
            features['stage2'], features['stage3'], features['stage4']
        )

        d4 = self.fovea_decoder_stage4(e4)                          # [B, 256, H/16, W/16]
        d3 = self.fovea_decoder_stage3(torch.cat([d4, e3], dim=1))  # [B, 128, H/8, W/8]
        d2 = self.fovea_decoder_stage2(torch.cat([d3, e2], dim=1))  # [B, 64,  H/4, W/4]
        d1 = self.fovea_decoder_stage1(torch.cat([d2, e1], dim=1))  # [B, 32,  H/4, W/4]

        out = self.fovea_final_upsample(torch.cat([d1, e0], dim=1))  # [B, 32, H, W]
        heatmap = self.fovea_head(out)  # [B, 1, H, W]

        return heatmap

    def forward(self, x):
        """
        前向传播

        Args:
            x: [B, 3, H, W] input image

        Returns:
            dict:
                'od_mask': [B, 1, H, W] Optic Disc 分割 logits
                'fovea_heatmap': [B, 1, H, W] Fovea 定位 heatmap (if predict_fovea=True)
        """
        features, _ = self.encode(x)

        output = {}

        # Optic Disc 分割
        output['od_mask'] = self.decode_optic_disc(features)

        # Fovea 定位
        if self.predict_fovea:
            output['fovea_heatmap'] = self.decode_fovea(features)

        return output

    def get_fovea_coordinates(self, heatmap, method='argmax'):
        """
        从 heatmap 中提取 Fovea 坐标

        Args:
            heatmap: [B, 1, H, W] 预测的 heatmap
            method: 'argmax' 或 'weighted_mean'

        Returns:
            coords: [B, 2] (x, y) 坐标
        """
        B, _, H, W = heatmap.shape
        heatmap_prob = torch.sigmoid(heatmap)  # 转为概率

        heatmap_flat = heatmap_prob.view(B, H * W)

        if method == 'argmax':
            # 找最大值位置
            max_idx = heatmap_flat.argmax(dim=1)
            y = max_idx // W
            x = max_idx % W
        else:
            # 加权平均
            heatmap_flat = heatmap_flat / (heatmap_flat.sum(dim=1, keepdim=True) + 1e-6)

            # 创建坐标网格
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=heatmap.device, dtype=torch.float32),
                torch.arange(W, device=heatmap.device, dtype=torch.float32),
                indexing='ij'
            )
            grid_x = grid_x.flatten().unsqueeze(0)  # [1, H*W]
            grid_y = grid_y.flatten().unsqueeze(0)  # [1, H*W]

            x = (heatmap_flat * grid_x).sum(dim=1)
            y = (heatmap_flat * grid_y).sum(dim=1)

        return torch.stack([x, y], dim=1)  # [B, 2]

    @staticmethod
    def morphological_process_od(od_mask, threshold=0.5, kernel_size=15, min_area=500):
        """
        对 OD mask 进行形态学处理

        Args:
            od_mask: [B, 1, H, W] OD 分割 logits
            threshold: 二值化阈值
            kernel_size: 形态学操作的核大小
            min_area: 最小连通域面积，小于此值的会被过滤

        Returns:
            processed_mask: [B, 1, H, W] 处理后的二值 mask (0-1 float)
        """
        import cv2
        import numpy as np

        B, _, H, W = od_mask.shape
        device = od_mask.device

        # 转为概率并二值化
        od_prob = torch.sigmoid(od_mask)
        od_binary = (od_prob > threshold).float()

        processed_masks = []

        # 形态学核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for b in range(B):
            mask = od_binary[b, 0].cpu().numpy().astype(np.uint8)

            if mask.sum() == 0:
                processed_masks.append(torch.zeros(1, H, W, device=device))
                continue

            # 1. 闭运算：填充内部孔洞
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # 2. 开运算：去除小噪声
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel)

            # 3. 只保留最大连通域
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

            if num_labels > 1:
                # 找最大的连通域（排除背景 label=0）
                largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                mask = (labels == largest_label).astype(np.uint8)

                # 检查面积是否足够
                if stats[largest_label, cv2.CC_STAT_AREA] < min_area:
                    mask = np.zeros_like(mask)

            # 4. 再次闭运算，使边缘更平滑
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, small_kernel)

            processed_masks.append(torch.from_numpy(mask).float().unsqueeze(0).to(device))

        return torch.stack(processed_masks, dim=0)  # [B, 1, H, W]

    @staticmethod
    def compute_od_confidence(od_mask, threshold=0.5, min_area=500):
        """
        基于 solidity（实心度）计算 OD 检测置信度

        Solidity = Area / Convex Hull Area
        理想的 OD 应该是接近圆形/椭圆形，solidity 接近 1

        Args:
            od_mask: [B, 1, H, W] OD 分割 logits 或处理后的二值 mask
            threshold: 二值化阈值（如果输入是 logits）
            min_area: 最小有效面积

        Returns:
            confidence: [B] 每个样本的置信度 (0-1)
            info: List[dict] 每个样本的详细信息
        """
        import cv2
        import numpy as np

        B, _, H, W = od_mask.shape
        device = od_mask.device

        # 判断输入是 logits 还是已经是二值 mask
        if od_mask.max() > 1 or od_mask.min() < 0:
            # 是 logits，需要 sigmoid
            od_prob = torch.sigmoid(od_mask)
            od_binary = (od_prob > threshold).float()
        else:
            # 已经是 0-1 的 mask
            od_binary = (od_mask > threshold).float()

        confidences = []
        infos = []

        for b in range(B):
            mask = od_binary[b, 0].cpu().numpy().astype(np.uint8)

            info = {
                'detected': False,
                'area': 0,
                'convex_area': 0,
                'solidity': 0.0,
                'circularity': 0.0,
                'center': (0, 0),
                'confidence': 0.0
            }

            if mask.sum() == 0:
                confidences.append(0.0)
                infos.append(info)
                continue

            # 找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) == 0:
                confidences.append(0.0)
                infos.append(info)
                continue

            # 取最大轮廓
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)

            if area < min_area:
                confidences.append(0.0)
                infos.append(info)
                continue

            # 计算凸包
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)

            # Solidity: 面积 / 凸包面积
            solidity = area / (hull_area + 1e-6)

            # Circularity: 4 * pi * area / perimeter^2
            perimeter = cv2.arcLength(largest_contour, True)
            circularity = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)

            # 计算中心
            M = cv2.moments(largest_contour)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = 0, 0

            # 综合置信度: solidity 和 circularity 的加权平均
            # OD 应该是圆形且实心的
            confidence = 0.6 * solidity + 0.4 * min(circularity, 1.0)

            info = {
                'detected': True,
                'area': int(area),
                'convex_area': int(hull_area),
                'solidity': float(solidity),
                'circularity': float(circularity),
                'center': (cx, cy),
                'confidence': float(confidence)
            }

            confidences.append(confidence)
            infos.append(info)

        return torch.tensor(confidences, device=device, dtype=torch.float32), infos

    @staticmethod
    def compute_fovea_confidence(fovea_heatmap, top_k=100):
        """
        基于 heatmap 峰值比例计算 Fovea 检测置信度

        r = max(H) / mean(top-k(H))

        如果预测很集中（尖锐的峰），r 接近 1
        如果预测分散，r 会较小

        Args:
            fovea_heatmap: [B, 1, H, W] Fovea heatmap (logits)
            top_k: 用于计算平均值的 top-k 个像素数

        Returns:
            confidence: [B] 每个样本的置信度
            info: List[dict] 每个样本的详细信息
        """
        B, _, H, W = fovea_heatmap.shape
        device = fovea_heatmap.device

        # 转为概率
        heatmap_prob = torch.sigmoid(fovea_heatmap)

        confidences = []
        infos = []

        for b in range(B):
            prob = heatmap_prob[b, 0]  # [H, W]
            prob_flat = prob.flatten()  # [H*W]

            # 最大值
            max_val = prob_flat.max().item()

            # Top-k 平均值
            k = min(top_k, prob_flat.numel())
            topk_vals, topk_indices = prob_flat.topk(k)
            topk_mean = topk_vals.mean().item()

            # 置信度: max / mean(top-k)
            if topk_mean > 1e-6:
                ratio = max_val / topk_mean
            else:
                ratio = 0.0

            # 获取最大值位置
            max_idx = prob_flat.argmax()
            max_y = (max_idx // W).item()
            max_x = (max_idx % W).item()

            info = {
                'max_value': float(max_val),
                'topk_mean': float(topk_mean),
                'ratio': float(ratio),
                'peak_location': (max_x, max_y),
                'confidence': float(ratio)
            }

            confidences.append(ratio)
            infos.append(info)

        return torch.tensor(confidences, device=device, dtype=torch.float32), infos
