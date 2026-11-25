"""
OD-Aware Swin Classifier

Two OD-centric representations:
  A. ODPosbiasClassifier  — adds an OD-relative (r, sinθ, cosθ) positional bias
                            onto patch tokens. Keeps ImageNet pretrained weights;
                            a lightweight modification.
  B. PolarResampleTransform — polar resampling: OD centered, the annular region
                            unrolled into a rectangle, + ODPolarClassifier: a Swin
                            trained on the polar image. ImageNet pretrained weights
                            no longer apply; needs training from scratch.

Common requirement: the OD center (cx, cy) and OD diameter (od_diam) are provided
externally (from OpticUNet inference). If the OD is not detected, it falls back to
the image center.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import timm
from typing import Optional, Tuple, List


# ============================================================
#  A. OD Positional Bias  (lightweight, keeps pretrained weights)
# ============================================================

class ODRelativePosEncoding(nn.Module):
    """
    Computes an OD-relative positional encoding for each patch token and projects
    it to embed_dim.

    Encoding: [r/od_d, sin(θ), cos(θ)]  (3D)  → MLP → embed_dim
    θ is measured relative to the OD→Fovea direction (falls back to the horizontal
    rightward direction = 0° if no fovea information is available).

    Zero-init output layer: initially equivalent to the original Swin; the OD prior
    is learned gradually during training.
    """

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        # zero-init: starts as a no-op
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def forward(
        self,
        patch_size: int,
        img_h: int,
        img_w: int,
        od_center: Tuple[float, float],   # (cx, cy) pixel coords
        od_diameter: float,
        fovea_center: Optional[Tuple[float, float]] = None,
        device: torch.device = torch.device('cpu'),
    ) -> torch.Tensor:
        """
        Returns: [num_patches, embed_dim]   num_patches = (H/P) * (W/P)
        """
        nH = img_h // patch_size
        nW = img_w // patch_size

        # patch center coordinates
        cy_grid = (torch.arange(nH, device=device).float() + 0.5) * patch_size  # [nH]
        cx_grid = (torch.arange(nW, device=device).float() + 0.5) * patch_size  # [nW]
        grid_y, grid_x = torch.meshgrid(cy_grid, cx_grid, indexing='ij')  # [nH, nW]

        dx = grid_x - od_center[0]   # [nH, nW]
        dy = grid_y - od_center[1]

        dist = torch.sqrt(dx**2 + dy**2 + 1e-8)
        r_norm = dist / (od_diameter + 1e-8)   # normalized radius

        # reference axis: OD→Fovea direction; fallback = (1, 0)
        if fovea_center is not None:
            ref_dx = fovea_center[0] - od_center[0]
            ref_dy = fovea_center[1] - od_center[1]
            ref_len = math.sqrt(ref_dx**2 + ref_dy**2) + 1e-8
            ref_x, ref_y = ref_dx / ref_len, ref_dy / ref_len
        else:
            ref_x, ref_y = 1.0, 0.0

        # angle relative to reference axis
        cos_t = (dx * ref_x + dy * ref_y) / dist
        sin_t = (dx * ref_y - dy * ref_x) / dist  # cross product (signed)

        # [nH, nW, 3] → [nH*nW, 3]
        pos_input = torch.stack([r_norm, sin_t, cos_t], dim=-1).view(-1, 3)
        return self.proj(pos_input)   # [nH*nW, embed_dim]


class ODPosbiasClassifier(nn.Module):
    """
    OD-aware Swin Classifier (Approach A).

    After patch embedding and before the transformer blocks, adds an OD-relative
    (r, sinθ, cosθ) positional bias onto each patch token.

    - Keeps ImageNet pretrained weights
    - The only new parameters are ODRelativePosEncoding (~10K params)
    - Without OD information, falls back to the image center (does not affect
      pretrained weights)
    """

    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = 'swin_tiny_patch4_window7_224',
        img_size: int = 1024,
        pretrained: bool = True,
        pos_hidden: int = 64,
        drop_rate: float = 0.0,
        backbone_lr_scale: float = 0.1,   # smaller lr for the backbone
    ):
        super().__init__()
        self.img_size   = img_size
        self.num_classes = num_classes

        # Swin backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # remove default head
            img_size=img_size,
            drop_rate=drop_rate,
        )
        embed_dim  = self.backbone.embed_dim
        patch_size = self.backbone.patch_embed.patch_size[0]
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        # OD positional bias
        self.od_pos_enc = ODRelativePosEncoding(embed_dim, hidden_dim=pos_hidden)

        # Classification head
        num_features = self.backbone.num_features  # final embed dim after all stages
        self.head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Linear(num_features, num_classes),
        )

        self.backbone_lr_scale = backbone_lr_scale

    def get_param_groups(self, lr: float):
        """
        Return param groups with separate lr for backbone vs new modules.
        Use in optimizer: optimizer = Adam(model.get_param_groups(lr))
        """
        return [
            {'params': self.backbone.parameters(),  'lr': lr * self.backbone_lr_scale},
            {'params': self.od_pos_enc.parameters(), 'lr': lr},
            {'params': self.head.parameters(),       'lr': lr},
        ]

    def _make_patch_embed_hook(self, bias):
        """
        Returns a forward hook that adds OD positional bias to patch_embed output.
        Handles both NHWC [B,H',W',C] and NLC [B,N,C] output formats.
        """
        def hook(module, input, output):
            if output.dim() == 4:          # NHWC (newer timm)
                B, pH, pW, C = output.shape
                flat = output.reshape(B, pH * pW, C) + bias
                return flat.reshape(B, pH, pW, C)
            else:                           # NLC (older timm)
                return output + bias
        return hook

    def _run_backbone_with_od_bias(
        self,
        x: torch.Tensor,
        od_centers: Optional[List[Tuple[float, float]]] = None,
        od_diameters: Optional[List[float]] = None,
        fovea_centers: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> torch.Tensor:
        """
        Run backbone with OD positional bias injected at patch_embed.
        Returns pooled features [B, num_features].
        """
        B, _, H, W = x.shape
        device = x.device

        biases = []
        for b in range(B):
            od_c  = od_centers[b]    if od_centers   is not None else (W / 2.0, H / 2.0)
            od_d  = od_diameters[b]  if od_diameters  is not None else H / 10.0
            fov_c = fovea_centers[b] if fovea_centers is not None else None
            biases.append(self.od_pos_enc(
                patch_size=self.patch_size, img_h=H, img_w=W,
                od_center=od_c, od_diameter=od_d,
                fovea_center=fov_c, device=device,
            ))
        bias_batch = torch.stack(biases, dim=0)  # [B, N, embed_dim]

        hook = self.backbone.patch_embed.register_forward_hook(
            self._make_patch_embed_hook(bias_batch)
        )
        features = self.backbone.forward_features(x)
        hook.remove()

        if features.dim() == 4:
            features = features.mean(dim=[1, 2])
        elif features.dim() == 3:
            features = features.mean(dim=1)
        return features  # [B, num_features]

    def forward_features(
        self,
        x: torch.Tensor,
        od_centers: Optional[List[Tuple[float, float]]] = None,
        od_diameters: Optional[List[float]] = None,
        fovea_centers: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> torch.Tensor:
        """Returns pooled backbone features [B, num_features] before the head."""
        return self._run_backbone_with_od_bias(x, od_centers, od_diameters, fovea_centers)

    def forward_features_with_stages(
        self,
        x: torch.Tensor,
        od_centers: Optional[List[Tuple[float, float]]] = None,
        od_diameters: Optional[List[float]] = None,
        fovea_centers: Optional[List[Optional[Tuple[float, float]]]] = None,
        add_od_bias: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Run backbone with OD bias, also capturing intermediate stage outputs.

        Args:
            add_od_bias: if False, skip the OD positional bias entirely (ablation).

        Returns:
            final_features: [B, num_features] pooled
            stage_dict: {'stage2': [B,C2,H2,W2], 'stage3': [B,C3,H3,W3], 'stage4': [B,C4,H4,W4]}
                        NCHW tensors from Swin-T layers[1/2/3].
                        stage4 is also the final spatial output (before global pool).
        """
        B, _, H, W = x.shape
        device = x.device

        bias_batch = None
        if add_od_bias:
            biases = []
            for b in range(B):
                od_c  = od_centers[b]    if od_centers   is not None else (W / 2.0, H / 2.0)
                od_d  = od_diameters[b]  if od_diameters  is not None else H / 10.0
                fov_c = fovea_centers[b] if fovea_centers is not None else None
                biases.append(self.od_pos_enc(
                    patch_size=self.patch_size, img_h=H, img_w=W,
                    od_center=od_c, od_diameter=od_d,
                    fovea_center=fov_c, device=device,
                ))
            bias_batch = torch.stack(biases, dim=0)

        stage_outputs = {}

        def _make_stage_hook(idx):
            def _h(module, inp, out):
                # out: NHWC [B,H',W',C] (timm ≥0.9) or NLC [B,N,C]
                if out.dim() == 4:          # NHWC → NCHW
                    stage_outputs[f'stage{idx}'] = out.permute(0, 3, 1, 2).contiguous()
                elif out.dim() == 3:        # NLC: infer spatial from input
                    _, N, C = out.shape
                    side = int(N ** 0.5)
                    stage_outputs[f'stage{idx}'] = out.transpose(1, 2).reshape(B, C, side, side)
            return _h

        hooks = []
        # layers[0]=stage1, [1]=stage2, [2]=stage3, [3]=stage4 — capture 2,3,4
        for stage_idx, layer_idx in [(2, 1), (3, 2), (4, 3)]:
            if layer_idx < len(self.backbone.layers):
                hooks.append(
                    self.backbone.layers[layer_idx].register_forward_hook(
                        _make_stage_hook(stage_idx)
                    )
                )

        od_hook = None
        if bias_batch is not None:
            od_hook = self.backbone.patch_embed.register_forward_hook(
                self._make_patch_embed_hook(bias_batch)
            )
        features = self.backbone.forward_features(x)
        if od_hook is not None:
            od_hook.remove()
        for h in hooks:
            h.remove()

        if features.dim() == 4:
            features = features.mean(dim=[1, 2])
        elif features.dim() == 3:
            features = features.mean(dim=1)

        return features, stage_outputs

    def forward(
        self,
        x: torch.Tensor,
        od_centers: Optional[List[Tuple[float, float]]] = None,
        od_diameters: Optional[List[float]] = None,
        fovea_centers: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> torch.Tensor:
        features = self._run_backbone_with_od_bias(x, od_centers, od_diameters, fovea_centers)
        return self.head(features)


# ============================================================
#  B. Polar Resample Transform  (aggressive, needs training from scratch)
# ============================================================

class PolarResampleTransform:
    """
    Resamples a fundus image into an OD-centered polar representation.

    Output image:
      rows (height): polar radius r ∈ [0, max_r_od_units * od_diameter]
      cols (width):  polar angle  θ ∈ [0, 2π], with 0° along the OD→Fovea direction

    Properties:
      - The OD is always on the first row (r≈0), becoming Swin's first patches
      - Patches at equal r form the same annular band, corresponding to the
        "4-quadrant" distribution
      - Uses bilinear interpolation; invalid regions outside the retina are filled black
      - Optional log-r transform to mitigate undersampling at the periphery

    Usage:
        transform = PolarResampleTransform(output_size=(512, 1024), max_r_od=5.0)
        polar_img = transform(image, od_center=(cx, cy), od_diameter=d)
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (512, 1024),  # (radial, angular)
        max_r_od: float = 5.0,                        # max radius in OD-diameter units
        log_r: bool = False,                           # log-polar (mitigate peripheral undersampling)
    ):
        self.out_h, self.out_w = output_size
        self.max_r_od = max_r_od
        self.log_r = log_r

    def __call__(
        self,
        image: torch.Tensor,        # [3, H, W] or [B, 3, H, W]
        od_center: Tuple[float, float],
        od_diameter: float,
        fovea_center: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """Returns polar-resampled image, same dtype and device as input."""
        batched = image.dim() == 4
        if not batched:
            image = image.unsqueeze(0)
        B, C, H, W = image.shape
        device = image.device

        cx, cy = od_center
        max_r_px = self.max_r_od * od_diameter

        # reference direction OD→Fovea (fallback: rightward)
        if fovea_center is not None:
            ref_dx = fovea_center[0] - cx
            ref_dy = fovea_center[1] - cy
            ref_len = math.sqrt(ref_dx**2 + ref_dy**2) + 1e-8
            ref_angle = math.atan2(ref_dy, ref_dx)
        else:
            ref_angle = 0.0

        # build sampling grid
        if self.log_r:
            r_vals = torch.exp(
                torch.linspace(math.log(1.0), math.log(max_r_px + 1), self.out_h, device=device)
            ) - 1.0
        else:
            r_vals = torch.linspace(0.0, max_r_px, self.out_h, device=device)

        theta_vals = torch.linspace(0.0, 2 * math.pi, self.out_w, device=device) + ref_angle

        r_grid, t_grid = torch.meshgrid(r_vals, theta_vals, indexing='ij')  # [out_h, out_w]

        # polar → Cartesian
        x_cart = cx + r_grid * torch.cos(t_grid)   # [out_h, out_w]
        y_cart = cy + r_grid * torch.sin(t_grid)

        # normalize to [-1, 1] for grid_sample
        x_norm = (x_cart / (W - 1)) * 2 - 1
        y_norm = (y_cart / (H - 1)) * 2 - 1

        grid = torch.stack([x_norm, y_norm], dim=-1)          # [out_h, out_w, 2]
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)        # [B, out_h, out_w, 2]

        polar = F.grid_sample(
            image.float(), grid,
            mode='bilinear', padding_mode='zeros', align_corners=True,
        )  # [B, 3, out_h, out_w]

        return polar.squeeze(0) if not batched else polar


class ODPolarClassifier(nn.Module):
    """
    OD-Polar Swin Classifier (Approach B).

    First polar-resamples the input (PolarResampleTransform), then classifies with a
    standard Swin. Needs training from scratch (ImageNet pretrained weights have
    little meaning on polar images).

    Note: to speed up convergence, one could pretrain on an ImageNet polar version,
          but that is engineering-heavy. Practical suggestion: fine-tune from
          ImageNet weights and check whether it converges.
    """

    def __init__(
        self,
        num_classes: int = 5,
        model_name: str = 'swin_tiny_patch4_window7_224',
        polar_size: Tuple[int, int] = (512, 1024),
        max_r_od: float = 5.0,
        log_r: bool = False,
        pretrained: bool = True,   # fine-tune from ImageNet weights (practical starting point)
        drop_rate: float = 0.1,
    ):
        super().__init__()
        self.polar_transform = PolarResampleTransform(
            output_size=polar_size, max_r_od=max_r_od, log_r=log_r
        )
        self.polar_h, self.polar_w = polar_size

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=max(polar_size),
            drop_rate=drop_rate,
        )

    def forward(
        self,
        x: torch.Tensor,
        od_centers: Optional[List[Tuple[float, float]]] = None,
        od_diameters: Optional[List[float]] = None,
        fovea_centers: Optional[List[Optional[Tuple[float, float]]]] = None,
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        device = x.device

        polar_imgs = []
        for b in range(B):
            od_c  = od_centers[b]   if od_centers   is not None else (W / 2, H / 2)
            od_d  = od_diameters[b] if od_diameters  is not None else H / 10.0
            fov_c = fovea_centers[b] if fovea_centers is not None else None
            p = self.polar_transform(x[b], od_c, od_d, fov_c)   # [3, out_h, out_w]
            # resize to model's expected input size
            p = F.interpolate(p.unsqueeze(0),
                              size=(max(self.polar_h, self.polar_w),
                                    max(self.polar_h, self.polar_w)),
                              mode='bilinear', align_corners=False).squeeze(0)
            polar_imgs.append(p)

        polar_batch = torch.stack(polar_imgs, dim=0)   # [B, 3, S, S]
        return self.backbone(polar_batch)
