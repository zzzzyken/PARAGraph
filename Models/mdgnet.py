"""
MDGNet reimplementation (baseline for PARAGraph comparison).

Paper: Wang et al., "A graph convolutional network with dynamic weight fusion
of multi-scale local features for diabetic retinopathy grading",
Scientific Reports 14, 2024 (s41598-024-56389-4).

There is NO official public code and the paper is the only reference, so the
details below are reimplemented from the paper's text. Components that the paper
does not fully specify are marked `INFERRED` and are the natural / standard
choice; they are the places to revisit if the numbers do not line up.

Architecture (from the paper):
  - Conv stem -> isotropic/pyramid body of [MSDF + ViG] blocks -> GAP -> FC head.
  - MSDF (Multi-scale Dynamic Fusion) = MFE + MFF:
      MFE (Multi-scale Feature Extraction): parallel depthwise convs with
          kernels {1,3,5,7} capturing lesions of different shape/size.
      MFF (Multi-scale Feature Fusion): learn a *dynamic* weight per scale from
          global context (avg-pool + max-pool -> shared MLP -> softmax over the
          4 scales), then fuse as the softmax-weighted sum of the 4 branches.
  - ViG block (Vision GNN): build a KNN graph over the feature-map nodes and
      apply a max-relative graph convolution
          y_i = FC( concat[ x_i , max_{j in N(i)} (x_j - x_i) ] )
      followed by an FFN, each with a residual connection (Grapher + FFN,
      Han et al. "Vision GNN", NeurIPS 2022).

Interface matches EfficientNetClassifier / SwinClassifier / PARAGraph:
    forward(images) -> {'logits', 'probs', 'predictions'}
and exposes `.num_classes`, `.backbone`, `.classifier` so the shared
train loop (train_mdgnet.py, cloned from train_efficientnet.py) can build the
differential-LR optimizer.

Resolution note: the paper uses 224x224 (stem /16 -> 14x14 = 196 graph nodes).
A dense KNN is O(N^2) in nodes, so to let the input resolution scale up to our
other baselines' scale WITHOUT blowing up the graph, the stem adaptively pools
to a fixed graph grid `graph_hw` (default 14) before the ViG body. Larger input
images therefore feed more detail through the conv stem while the graph stays
bounded. Set `graph_hw` higher to make the graph denser.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  MSDF: multi-scale extraction + dynamic weighted fusion
# --------------------------------------------------------------------------- #
class MSDF(nn.Module):
    """Multi-scale Dynamic Fusion module (MFE parallel convs + MFF dynamic weights)."""

    def __init__(self, channels: int, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.kernels = kernels
        self.n_scales = len(kernels)

        # MFE: one depthwise conv per scale (depthwise keeps it light, as in the
        # paper's "local feature" extraction), followed by a pointwise 1x1.
        self.branches = nn.ModuleList()
        for k in kernels:
            self.branches.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=k,
                              padding=k // 2, groups=channels, bias=False),
                    nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(channels),
                    nn.GELU(),
                )
            )

        # MFF: global context (avg + max) -> shared MLP -> per-scale logits.
        reduced = max(channels // 4, 8)
        self.mff_mlp = nn.Sequential(
            nn.Linear(channels, reduced),
            nn.GELU(),
            nn.Linear(reduced, self.n_scales),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        feats = [branch(x) for branch in self.branches]           # n_scales * [B,C,H,W]
        stacked = torch.stack(feats, dim=1)                       # [B, S, C, H, W]

        # Dynamic weights from global context (avg + max pooled descriptors).
        gap = x.mean(dim=(2, 3))                                  # [B, C]
        gmp = x.amax(dim=(2, 3))                                  # [B, C]
        logits = self.mff_mlp(gap) + self.mff_mlp(gmp)            # [B, S]
        weights = F.softmax(logits, dim=1)                        # [B, S]

        fused = (stacked * weights[:, :, None, None, None]).sum(dim=1)  # [B,C,H,W]
        return x + fused                                          # residual


# --------------------------------------------------------------------------- #
#  ViG block: KNN graph + max-relative graph convolution + FFN
# --------------------------------------------------------------------------- #
def _knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of k nearest neighbours (by L2 in feature space) per node.

    x: [B, N, C] -> idx: [B, N, k]  (excludes self).
    """
    with torch.no_grad():
        dist = torch.cdist(x, x)                                  # [B, N, N]
        # exclude self by pushing the diagonal to +inf
        n = x.size(1)
        diag = torch.arange(n, device=x.device)
        dist[:, diag, diag] = float('inf')
        idx = dist.topk(k, dim=-1, largest=False).indices         # [B, N, k]
    return idx


class MaxRelGraphConv(nn.Module):
    """Max-relative graph convolution (ViG)."""

    def __init__(self, in_ch: int, out_ch: int, k: int = 9):
        super().__init__()
        self.k = k
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W
        nodes = x.flatten(2).transpose(1, 2)                      # [B, N, C]
        idx = _knn(nodes, min(self.k, N - 1))                     # [B, N, k]

        # gather neighbour features and take max of (neighbour - self)
        k = idx.size(-1)
        batch_idx = torch.arange(B, device=x.device)[:, None, None]
        neigh = nodes[batch_idx, idx]                             # [B, N, k, C]
        rel = neigh - nodes[:, :, None, :]                        # [B, N, k, C]
        agg = rel.amax(dim=2)                                     # [B, N, C]

        feat = torch.cat([nodes, agg], dim=-1)                    # [B, N, 2C]
        feat = feat.transpose(1, 2).reshape(B, 2 * C, H, W)       # [B, 2C, H, W]
        return self.fc(feat)                                      # [B, out, H, W]


class FFN(nn.Module):
    def __init__(self, ch: int, hidden_ratio: int = 4):
        super().__init__()
        hidden = ch * hidden_ratio
        self.net = nn.Sequential(
            nn.Conv2d(ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.GELU(),
            nn.Conv2d(hidden, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return x + self.net(x)


class ViGBlock(nn.Module):
    """Grapher (max-relative graph conv, residual) + FFN (residual)."""

    def __init__(self, ch: int, k: int = 9):
        super().__init__()
        self.graph_in = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch))
        self.graph = MaxRelGraphConv(ch, ch, k=k)
        self.graph_out = nn.Sequential(nn.Conv2d(ch, ch, 1, bias=False), nn.BatchNorm2d(ch))
        self.ffn = FFN(ch)

    def forward(self, x):
        shortcut = x
        x = self.graph_in(x)
        x = self.graph(x)
        x = self.graph_out(x)
        x = x + shortcut
        x = self.ffn(x)
        return x


# --------------------------------------------------------------------------- #
#  Full MDGNet
# --------------------------------------------------------------------------- #
class MDGNet(nn.Module):
    """MDGNet: conv stem -> [MSDF + ViG] body -> GAP -> FC head."""

    def __init__(
        self,
        num_classes: int = 5,
        embed_dims: List[int] = (96, 192, 384),
        blocks_per_stage: int = 1,
        knn_k: int = 9,
        graph_hw: int = 14,
        drop_rate: float = 0.3,
        in_ch: int = 3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.graph_hw = graph_hw

        # --- conv stem: downsample input, embed to embed_dims[0] (INFERRED: a
        #     4-conv /16 stem, as in ViG's Pyramid stem). ---
        c0 = embed_dims[0]
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, c0 // 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0 // 2), nn.GELU(),
            nn.Conv2d(c0 // 2, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.GELU(),
            nn.Conv2d(c0, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.GELU(),
            nn.Conv2d(c0, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0), nn.GELU(),
        )

        # --- body: a stage per embed_dim; each stage = MSDF + ViG x blocks,
        #     with a strided downsample between stages (pyramid). ---
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for s, dim in enumerate(embed_dims):
            stage = nn.ModuleList()
            for _ in range(blocks_per_stage):
                stage.append(nn.ModuleList([MSDF(dim), ViGBlock(dim, k=knn_k)]))
            self.stages.append(stage)
            if s < len(embed_dims) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.Conv2d(dim, embed_dims[s + 1], 3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(embed_dims[s + 1]),
                    )
                )

        self.backbone = nn.ModuleList([self.stem, self.stages, self.downsamples])

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dims[-1]),
            nn.Dropout(drop_rate),
            nn.Linear(embed_dims[-1], num_classes),
        )

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        x = self.stem(images)                                     # [B, c0, h, w]
        # Pool to the fixed graph grid so the KNN graph stays bounded regardless
        # of input resolution (see module docstring).
        if x.shape[-1] != self.graph_hw or x.shape[-2] != self.graph_hw:
            x = F.adaptive_avg_pool2d(x, self.graph_hw)
        for s, stage in enumerate(self.stages):
            for msdf, vig in stage:
                x = msdf(x)
                x = vig(x)
            if s < len(self.downsamples):
                x = self.downsamples[s](x)
        return x

    def forward(self, images: torch.Tensor) -> dict:
        x = self.forward_features(images)
        pooled = x.mean(dim=(2, 3))                               # GAP -> [B, C]
        logits = self.classifier(pooled)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return {'logits': logits, 'probs': probs, 'predictions': predictions}
