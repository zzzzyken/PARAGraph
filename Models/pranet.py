"""
PRANet reimplementation (baseline for PARAGraph comparison).

Paper: "PRANet: Pathological relationship perception and dual attention guided
network for diabetic retinopathy grading", Applied Intelligence, 2025
(DOI 10.1007/s10489-025-07036-6).

There is NO official public code and the paper is paywalled, so this is a
reimplementation from the abstract / figures / available description. Every
component the source does not pin down is marked `INFERRED`; those are the
knobs to revisit if numbers diverge. Reported targets: DDR ACC 0.8821 /
QWK 0.9039 / AUC 0.9409; datasets DDR, APTOS2019, FGADR.

Pipeline (two stage):
  Stage 1 -- PRANetLesionDetector: PRANet's OWN lesion-detection network
      (a compact encoder-decoder segmenter, trained from scratch on DDR's
      pixel lesion masks: MA/HE/EX/SE). It is NOT our production LesionUNet.
      Trained once via train_pranet_detector.py, then frozen.
  Stage 2 -- PRANet grading network:
      * Global branch: EfficientNetV2-M (ImageNet pretrained) feature map +
        Dual Attention Module (DAM = channel attention + spatial attention)
        emphasising suspicious lesion regions.
      * Pathological-relationship branch (PRFM): the frozen detector locates
        rough lesion regions; lesion-region descriptors from the global
        feature map are K-means clustered into lesion *nodes*; a co-occurrence
        adjacency links the nodes; a GCN propagates relationships.
      * The global (DAM) vector and the graph vector are fused and classified.

Interface matches the other baselines:
    forward(images) -> {'logits', 'probs', 'predictions'}
exposing `.num_classes`, `.backbone` (EfficientNetV2-M) so the train loop can
build a differential-LR optimizer; all other submodules train at head LR.

Resolution: EfficientNetV2-M is convolutional, so the input can scale up to our
other baselines' resolution (`--image_size`, default 1024) with no architectural
change -- unlike the paper's smaller default. The detector runs at the same
resolution. The graph is bounded by K (number of K-means nodes), not resolution.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# --------------------------------------------------------------------------- #
#  Stage 1: PRANet's own lesion detector (compact U-Net, 4 lesion channels)
# --------------------------------------------------------------------------- #
class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PRANetLesionDetector(nn.Module):
    """PRANet's own lesion-detection network.

    A compact U-Net segmenter producing rough lesion regions for the 4 lesion
    types (MA/HE/EX/SE). Deliberately separate from the project LesionUNet so
    PRANet owns its detector end-to-end (trained from scratch on DDR masks).
    Output: per-pixel lesion logits [B, 4, H, W].
    """

    LESION_NAMES = ['MA', 'HE', 'EX', 'SE']

    def __init__(self, n_lesions: int = 4, base: int = 32):
        super().__init__()
        self.n_lesions = n_lesions
        self.enc1 = _DoubleConv(3, base)
        self.enc2 = _DoubleConv(base, base * 2)
        self.enc3 = _DoubleConv(base * 2, base * 4)
        self.enc4 = _DoubleConv(base * 4, base * 8)
        self.bottleneck = _DoubleConv(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = _DoubleConv(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = _DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = _DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = _DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, n_lesions, 1)

    def forward(self, x: torch.Tensor) -> dict:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        masks = self.head(d1)
        return {'masks': masks}


# --------------------------------------------------------------------------- #
#  Dual Attention Module (channel + spatial), CBAM-style
# --------------------------------------------------------------------------- #
class DualAttention(nn.Module):
    """DAM: channel attention then spatial attention over the global feature map."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        r = max(channels // reduction, 8)
        self.ch_mlp = nn.Sequential(
            nn.Linear(channels, r), nn.ReLU(inplace=True), nn.Linear(r, channels)
        )
        self.sp_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # channel attention
        gap = x.mean(dim=(2, 3))                                  # [B, C]
        gmp = x.amax(dim=(2, 3))                                  # [B, C]
        ch = torch.sigmoid(self.ch_mlp(gap) + self.ch_mlp(gmp))   # [B, C]
        x = x * ch[:, :, None, None]
        # spatial attention
        avg = x.mean(dim=1, keepdim=True)                         # [B,1,H,W]
        mx = x.amax(dim=1, keepdim=True)                          # [B,1,H,W]
        sp = torch.sigmoid(self.sp_conv(torch.cat([avg, mx], dim=1)))
        x = x * sp
        return x


# --------------------------------------------------------------------------- #
#  Lesion-node construction + co-occurrence GCN (PRFM)
# --------------------------------------------------------------------------- #
def torch_kmeans(feats: torch.Tensor, k: int, iters: int = 10) -> torch.Tensor:
    """Lightweight K-means (Lloyd) on GPU. feats: [M, C] -> centroids [k, C].

    If M < k the feats are padded by repetition so exactly k nodes are returned.
    """
    M = feats.size(0)
    if M == 0:
        return feats.new_zeros(k, feats.size(1))
    if M < k:
        reps = (k + M - 1) // M
        feats = feats.repeat(reps, 1)[:k]
        M = k
    # init: k random distinct rows
    idx = torch.randperm(M, device=feats.device)[:k]
    centroids = feats[idx].clone()
    for _ in range(iters):
        d = torch.cdist(feats, centroids)                        # [M, k]
        assign = d.argmin(dim=1)                                 # [M]
        new_c = centroids.clone()
        for c in range(k):
            m = assign == c
            if m.any():
                new_c[c] = feats[m].mean(dim=0)
        if torch.allclose(new_c, centroids, atol=1e-5):
            centroids = new_c
            break
        centroids = new_c
    return centroids                                             # [k, C]


class GCNLayer(nn.Module):
    """Dense GCN layer: H' = act( A_hat @ H @ W )."""

    def __init__(self, cin, cout, act=True):
        super().__init__()
        self.lin = nn.Linear(cin, cout)
        self.bn = nn.BatchNorm1d(cout)
        self.act = act

    def forward(self, x, a_hat):
        # x: [B, K, Cin], a_hat: [B, K, K]
        x = torch.bmm(a_hat, x)
        x = self.lin(x)
        B, K, C = x.shape
        x = self.bn(x.reshape(B * K, C)).reshape(B, K, C)
        if self.act:
            x = F.relu(x)
        return x


class PRFM(nn.Module):
    """Pathological Relationship Feature perception Module: co-occurrence GCN."""

    def __init__(self, in_ch: int, hidden: int = 256, out_ch: int = 256, layers: int = 2):
        super().__init__()
        dims = [in_ch] + [hidden] * (layers - 1) + [out_ch]
        self.gcns = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1], act=(i < layers - 1)) for i in range(layers)]
        )

    @staticmethod
    def cooccurrence_adjacency(nodes: torch.Tensor) -> torch.Tensor:
        """INFERRED co-occurrence adjacency from node-feature affinity.

        The paper builds an adjacency from co-occurrence relationships between
        lesion nodes; with no formula given, we use symmetric cosine affinity
        (a learnable-feature proxy for how often nodes co-activate), add
        self-loops, and apply the standard symmetric normalisation
        A_hat = D^-1/2 (A + I) D^-1/2.  nodes: [B, K, C] -> [B, K, K].
        """
        x = F.normalize(nodes, dim=-1)
        a = torch.bmm(x, x.transpose(1, 2)).clamp(min=0)          # [B,K,K] cosine>=0
        eye = torch.eye(a.size(1), device=a.device)[None]
        a = a + eye
        d = a.sum(dim=-1).clamp(min=1e-6)
        d_inv_sqrt = d.pow(-0.5)
        a_hat = a * d_inv_sqrt[:, None, :] * d_inv_sqrt[:, :, None]
        return a_hat

    def forward(self, nodes: torch.Tensor) -> torch.Tensor:
        a_hat = self.cooccurrence_adjacency(nodes)
        x = nodes
        for gcn in self.gcns:
            x = gcn(x, a_hat)
        return x.mean(dim=1)                                     # graph readout [B, out_ch]


# --------------------------------------------------------------------------- #
#  Full PRANet grading network
# --------------------------------------------------------------------------- #
class PRANet(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        backbone_name: str = 'efficientnetv2_m',
        pretrained: bool = True,
        detector: Optional[PRANetLesionDetector] = None,
        detector_ckpt: Optional[str] = None,
        num_nodes: int = 12,
        lesion_thresh: float = 0.5,
        gcn_out: int = 256,
        drop_rate: float = 0.3,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.lesion_thresh = lesion_thresh

        # --- global branch: EfficientNetV2-M feature map ---
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(4,)
        )
        feat_ch = self.backbone.feature_info.channels()[-1]
        self.feat_ch = feat_ch
        self.dam = DualAttention(feat_ch)
        self.global_proj = nn.Sequential(
            nn.Linear(feat_ch, gcn_out), nn.LayerNorm(gcn_out), nn.ReLU(inplace=True)
        )

        # --- lesion detector (frozen) ---
        self.detector = detector if detector is not None else PRANetLesionDetector()
        if detector_ckpt is not None:
            sd = torch.load(detector_ckpt, map_location='cpu', weights_only=False)
            sd = sd.get('model_state_dict', sd)
            self.detector.load_state_dict(sd)
            print(f"[PRANet] loaded frozen detector from {detector_ckpt}")
        for p in self.detector.parameters():
            p.requires_grad = False

        # --- pathological-relationship branch (GCN over K-means lesion nodes) ---
        # node feature = global-feature-map vector (feat_ch) ++ lesion-type soft
        # composition (4) so co-occurrence can reflect lesion types.
        self.node_in = feat_ch + 4
        self.prfm = PRFM(self.node_in, hidden=gcn_out, out_ch=gcn_out)

        # --- fusion + head ---
        self.classifier = nn.Sequential(
            nn.LayerNorm(gcn_out * 2),
            nn.Dropout(drop_rate),
            nn.Linear(gcn_out * 2, num_classes),
        )

    @torch.no_grad()
    def _detect(self, images):
        self.detector.eval()
        return torch.sigmoid(self.detector(images)['masks'])      # [B,4,H,W]

    def _build_nodes(self, feat_map, lesion_probs):
        """Per-sample K-means lesion nodes from the global feature map.

        feat_map: [B, Cf, Hf, Wf]; lesion_probs: [B, 4, H, W].
        Returns node tensor [B, K, Cf+4].
        """
        B, Cf, Hf, Wf = feat_map.shape
        lp = F.interpolate(lesion_probs, size=(Hf, Wf), mode='bilinear', align_corners=False)
        fm = feat_map.flatten(2).transpose(1, 2)                  # [B, N, Cf]
        lpf = lp.flatten(2).transpose(1, 2)                       # [B, N, 4]
        any_les = lpf.amax(dim=-1)                                # [B, N]

        out = feat_map.new_zeros(B, self.num_nodes, self.node_in)
        for b in range(B):
            sel = any_les[b] > self.lesion_thresh
            if sel.sum() < 2:
                # fall back: use the top-activated locations so grade-0 / silent
                # detections still yield a (near-empty) graph instead of crashing
                topn = min(self.num_nodes, any_les[b].numel())
                sel = torch.topk(any_les[b], topn).indices
                node_feats = torch.cat([fm[b][sel], lpf[b][sel]], dim=-1)
            else:
                node_feats = torch.cat([fm[b][sel], lpf[b][sel]], dim=-1)
            centroids = torch_kmeans(node_feats, self.num_nodes)
            out[b] = centroids
        return out                                                # [B, K, Cf+4]

    def forward(self, images: torch.Tensor) -> dict:
        feat_map = self.backbone(images)[-1]                      # [B, Cf, Hf, Wf]

        # global (DAM) vector
        g = self.dam(feat_map)
        g = g.mean(dim=(2, 3))                                    # [B, Cf]
        g = self.global_proj(g)                                   # [B, gcn_out]

        # pathological-relationship (graph) vector
        lesion_probs = self._detect(images)
        nodes = self._build_nodes(feat_map, lesion_probs)         # [B, K, Cf+4]
        graph_vec = self.prfm(nodes)                              # [B, gcn_out]

        fused = torch.cat([g, graph_vec], dim=-1)
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=-1)
        predictions = logits.argmax(dim=-1)
        return {'logits': logits, 'probs': probs, 'predictions': predictions}
