"""
PARAGraph: hierarchical graph construction for fundus images.

Builds a graph structure from OpticUNet and LesionUNet outputs:
- OD as the origin; the landmark (fovea) defines the positive x-axis
- distances measured in OD-diameter units
- Lesion nodes carry position, feature vector, and confidence
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional

from .optic_unet import OpticUNet
from .lesion_unet import LesionUNet
from .graph_attention import GraphAttentionLayer

# constants and static functions
from ._paragraph_constants import (
    NODE_GRAPH, NODE_GLOBAL, NODE_MA, NODE_HE, NODE_EX, NODE_SE,
    NODE_MA_CAT, NODE_HE_CAT, NODE_EX_CAT, NODE_SE_CAT,
    NODE_REGION,
    LESION_NAMES, LESION_NODE_TYPES, CATEGORY_NODE_TYPES, NUM_EDGE_TYPES,
    # backward-compat aliases still available
    NODE_OD, NODE_LANDMARK, TOKEN_NODE_TYPES,
    get_edge_type_id, compute_od_diameter, weighted_median,
    compute_landmark_with_prior, get_dr_grade_name
)


# mixins
from ._paragraph_features import PARAGraphFeaturesMixin
from ._paragraph_gnn import PARAGraphGNNMixin
from ._paragraph_training import PARAGraphTrainingMixin
from .swin_od_pos_bias import ODRelativePosEncoding, ODPosbiasClassifier


class PARAGraph(PARAGraphFeaturesMixin, PARAGraphGNNMixin, PARAGraphTrainingMixin, nn.Module):
    """
    Hierarchical fundus lesion-graph builder.

    The graph is built with the OD as origin (0, 0), the landmark (fovea) along
    the positive x-axis, and distances measured in OD-diameter units.

    Node types:
    - NODE_GRAPH  (0): Graph node (anchored at OD; graph-level aggregator)
    - NODE_GLOBAL (1): Global node (anchored at Fovea; global spatial anchor)
    - NODE_MA (2): microaneurysm
    - NODE_HE (3): hemorrhage
    - NODE_EX (4): hard exudate
    - NODE_SE (5): soft exudate
    - NODE_MA_CAT (6): MA category node
    - NODE_HE_CAT (7): HE category node
    - NODE_EX_CAT (8): EX category node
    - NODE_SE_CAT (9): SE category node
    - NODE_REGION (10): lesion-enriched region node

    Edge structure (as actually constructed in forward):
    - Graph ↔ every category node (bidirectional)
    - Global ↔ every category node (bidirectional)
    - each category node ↔ all lesion nodes of that class (bidirectional)
    - Graph ↔ near-OD lesion nodes (direct shortcut; only when the lesion is
      within edge_distance_threshold OD-diameters of the OD center; only Graph,
      not Global)
    - Region ↔ Graph and Region ↔ Global (bidirectional); a region aggregates its
      cluster's lesion features internally via attention (no region↔lesion edges)

    GNN message passing:
    num_gnn_layers GAT layers process all edges (full-graph message passing).
    """

    # node type constants
    NODE_GRAPH    = NODE_GRAPH
    NODE_GLOBAL   = NODE_GLOBAL
    NODE_MA       = NODE_MA
    NODE_HE       = NODE_HE
    NODE_EX       = NODE_EX
    NODE_SE       = NODE_SE
    NODE_MA_CAT   = NODE_MA_CAT
    NODE_HE_CAT   = NODE_HE_CAT
    NODE_EX_CAT   = NODE_EX_CAT
    NODE_SE_CAT   = NODE_SE_CAT
    NODE_REGION   = NODE_REGION
    # backward-compat
    NODE_OD = NODE_GRAPH
    NODE_LANDMARK = NODE_GLOBAL
    NODE_MA_TOKEN = NODE_MA_CAT

    LESION_NAMES        = LESION_NAMES
    LESION_NODE_TYPES   = LESION_NODE_TYPES
    CATEGORY_NODE_TYPES = CATEGORY_NODE_TYPES
    TOKEN_NODE_TYPES    = CATEGORY_NODE_TYPES   # backward-compat
    NUM_EDGE_TYPES      = NUM_EDGE_TYPES

    # static-method delegation (backward-compat)
    get_edge_type_id = staticmethod(get_edge_type_id)
    compute_od_diameter = staticmethod(compute_od_diameter)
    weighted_median = staticmethod(weighted_median)
    compute_landmark_with_prior = staticmethod(compute_landmark_with_prior)
    get_dr_grade_name = staticmethod(get_dr_grade_name)

    def __init__(
        self,
        optic_unet: OpticUNet,
        lesion_unet: Optional[LesionUNet] = None,
        vessel_unet=None,   # deprecated, ignored (kept for API compat)
        default_od_diameter: float = 100.0,
        lesion_threshold: float = 0.5,
        min_lesion_area: int = 5,
        min_ma_area: int = 1,
        max_lesions_per_class: int = 100,
        feature_dim: int = 256,
        num_gnn_layers: int = 3,
        num_classes: int = 5,
        gnn_dropout: float = 0.1,
        num_region_nodes: int = 4,
        region_min_lesions: int = 3,
        topk_lesions_per_cluster: int = None,
        use_prior_fusion: bool = True,
        fusion_mode: str = 'full',
        use_optic_nodes: bool = True,
        use_gating: bool = True,
        pos_proj_ln: bool = False,
        edge_distance_threshold: float = 0.5,
        use_global_logits: bool = True,
        use_global_node: bool = True,
        use_region_nodes: bool = True,
        use_od_posbias: bool = True,
        lesion_direct: bool = False,
    ):
        """
        PARAGraph — hierarchical GNN with dual-branch output.

        Anatomical branch (GNN):
          OD → Lesion nodes → Category tokens → Region nodes → graph pool → anatomical_logits

        Global branch (only used when use_global_logits=True):
          whole-image avg pool of global_swin → head → global_logits

        Final output:
          use_global_logits=True  → logits = anatomical_logits + global_logits (dual-branch)
          use_global_logits=False → logits = anatomical_logits (single graph branch;
                                    global_swin is then only a node feature extractor)

        Confidence gate: each node has its own gate g_i = σ(α·c_i + β), where
                     c_i is that node's reliability — lesion nodes use their
                     segmentation confidence, category/region nodes use the
                     (mean) confidence of their member lesions, and the OD node
                     uses the accumulated category confidence. α, β are shared.
                     Each edge is weighted by g_from · g_to, so a message is
                     trusted only when BOTH endpoints are reliable.
                     Assumption: segmentation is mostly reliable but not error-free;
                     the gate down-weights low-confidence (likely-wrong) nodes.
        Position encoding: lesion encoder features are modulated by (r, sinθ, cosθ)
                           before entering the GNN.
        """
        super().__init__()
        self.optic_unet   = optic_unet
        self.lesion_unet  = lesion_unet
        self.vessel_unet  = None           # vessel removed
        self.feature_dim  = feature_dim
        self.num_classes  = num_classes
        self.num_gnn_layers = num_gnn_layers
        self.gnn_dropout  = gnn_dropout

        self.default_od_diameter     = default_od_diameter
        self.lesion_threshold        = lesion_threshold
        self.min_lesion_area         = min_lesion_area
        self.min_ma_area             = min_ma_area
        self.max_lesions_per_class   = max_lesions_per_class
        self.num_region_nodes        = num_region_nodes
        self.region_min_lesions      = region_min_lesions
        self.topk_lesions_per_cluster = topk_lesions_per_cluster
        self.use_prior_fusion        = use_prior_fusion
        self.use_optic_nodes         = use_optic_nodes
        self.use_gating              = use_gating
        self.edge_distance_threshold = edge_distance_threshold
        # Dual-branch fusion: when True, logits = anatomical_logits + global_logits.
        # When False, the model is single-branch (graph only): logits = anatomical_logits;
        # global_swin then acts purely as a feature extractor for the graph nodes.
        self.use_global_logits       = use_global_logits
        # Ablation switches (all default to the full model):
        #   use_global_node : keep the Global(Fovea) node in the graph
        #   use_region_nodes: build Region (lesion-cluster) nodes
        #   use_od_posbias  : inject OD positional bias into global_swin
        #   lesion_direct   : lesions connect directly to OD/Global hubs, skipping
        #                     Category and Region nodes (hierarchy-ladder L1)
        self.use_global_node         = use_global_node
        self.use_region_nodes        = use_region_nodes
        self.use_od_posbias          = use_od_posbias
        self.lesion_direct           = lesion_direct

        # Node visual features come exclusively from global_swin (see
        # global_lesion_feature_proj/fuse below); OpticUNet/LesionUNet are used
        # only to produce OD/Fovea/lesion masks, not as graph feature extractors.

        # ── Medical prior fusion (for GNN node features) ──────────────────────
        self.fusion_mode = fusion_mode
        if fusion_mode == 'pos_only':
            self.pos_extra_dim = 2
        else:
            self.pos_extra_dim = 7  # [cx, cy, type_onehot(4), area]
        if pos_proj_ln:
            self.pos_proj = nn.Sequential(
                nn.Linear(feature_dim + self.pos_extra_dim, feature_dim),
                nn.LayerNorm(feature_dim),
            )
        else:
            self.pos_proj = nn.Sequential(
                nn.Linear(feature_dim + self.pos_extra_dim, feature_dim),
                nn.GELU(),
                nn.Dropout(gnn_dropout),
                nn.Linear(feature_dim, feature_dim),
            )

        # ── Per-node confidence gate ──────────────────────────────────────────
        # g_i = σ(α · c_i + β), where c_i is each node's own reliability:
        #   lesion → segmentation confidence; category/region → mean member-lesion
        #   confidence; OD → accumulated category confidence.
        # α, β are shared learnable scalars (od_gate_alpha/beta name kept for
        # backward-compat). Each edge is weighted by g_from · g_to in the GNN.
        # Assumption: segmentation is mostly reliable but still makes errors;
        # the gate suppresses messages from low-confidence (likely-wrong) nodes.
        self.od_gate_alpha = nn.Parameter(torch.tensor(1.0))
        self.od_gate_beta  = nn.Parameter(torch.tensor(0.0))

        # ── Category nodes: hardcoded mean aggregation (no learnable transform) ─

        # ── Region nodes ──────────────────────────────────────────────────────
        self.region_tokens = nn.Parameter(torch.randn(1, feature_dim))
        nn.init.normal_(self.region_tokens, std=0.02)
        self.region_attention = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4,
            dropout=gnn_dropout, batch_first=True,
        )

        # ── GNN layers ────────────────────────────────────────────────────────
        self.gnn_layers = nn.ModuleList()
        self.gnn_norms  = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                GraphAttentionLayer(feature_dim, feature_dim, num_heads=4, dropout=gnn_dropout)
            )
            self.gnn_norms.append(nn.LayerNorm(feature_dim))
        self.gat_num_heads = 4
        self.edge_type_attn_bias = nn.Embedding(self.NUM_EDGE_TYPES, self.gat_num_heads)
        nn.init.zeros_(self.edge_type_attn_bias.weight)

        # ── Graph pooling → anatomical branch classifier ──────────────────────
        self.graph_pool_queries = nn.Parameter(torch.randn(num_classes, feature_dim))
        nn.init.normal_(self.graph_pool_queries, std=0.02)
        self.graph_pool_attn = nn.MultiheadAttention(
            embed_dim=feature_dim, num_heads=4,
            dropout=gnn_dropout, batch_first=True,
        )
        # graph branch: pool output → feature_dim
        self.graph_proj = nn.Sequential(
            nn.Linear(feature_dim * num_classes, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(gnn_dropout),
        )

        # ── Global node: trainable Swin-T + OD pos bias → feature_dim ───────────
        # Separate trainable backbone (ODPosbiasClassifier) for the Global node.
        # Used as a feature extractor only — its head is not used for prediction.
        # The frozen LesionUNet is unchanged (lesion segmentation only).
        self.global_img_size = 512
        self.global_swin = ODPosbiasClassifier(
            num_classes=num_classes,
            model_name='swin_tiny_patch4_window7_224',
            img_size=self.global_img_size,
            pretrained=True,
            pos_hidden=64,
            drop_rate=gnn_dropout,
        )
        _global_num_features = self.global_swin.backbone.num_features  # 768 for Swin-T
        self.global_node_proj = nn.Sequential(
            nn.LayerNorm(_global_num_features),
            nn.Linear(_global_num_features, feature_dim),
            nn.GELU(),
            nn.Dropout(gnn_dropout),
        )

        # ── Lesion feature proj from global_swin stages (192/384/768 → feature_dim) ──
        # Swin-T layers[1/2/3] → stage2 192ch, stage3 384ch, stage4 768ch
        _swin_stage_dims = {'stage2': 192, 'stage3': 384, 'stage4': 768}
        self.global_lesion_feature_proj = nn.ModuleDict({
            k: nn.Sequential(
                nn.Conv2d(c, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
            )
            for k, c in _swin_stage_dims.items()
        })
        self.global_lesion_feature_fuse = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
        )

        # ── Classifier ────────────────────────────────────────────────────────
        self.shared_classifier = nn.Linear(feature_dim, num_classes)

        self._use_grade234_finetune = False


    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: str,
        device: str = 'cpu',
        eval_mode: bool = True,
        strict: bool = False,
        verbose: bool = True,
    ) -> 'PARAGraph':
        """
        Load a full PARAGraph model from a checkpoint.

        Rebuilds the model architecture from the args saved in the checkpoint
        (including the OpticUNet and LesionUNet submodels), then loads the
        trained weights.

        Args:
            checkpoint_path: checkpoint file path (.pth)
            device: target device ('cpu', 'cuda', 'cuda:0', ...)
            eval_mode: whether to set eval mode (default True)
            strict: strict state_dict matching (default False; tolerates orphan keys
                    from older checkpoints that contained now-removed modules)
            verbose: print loading info (default True)

        Returns:
            model: PARAGraph model with weights loaded

        Example::

            model = PARAGraph.load_from_checkpoint(
                './checkpoints/paragraph/paragraph_xxx/best_model.pth',
                device='cuda'
            )
            output = model(images)
        """
        import os

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        args = checkpoint.get('args', {})

        # ---- read architecture params from args ----
        # OpticUNet (optic disc + fovea) is ResNet-50 only.
        # LesionUNet: new format (lesion_encoder) and legacy (encoder).
        _legacy_encoder = args.get('encoder', 'swin')
        optic_encoder = 'resnet50'
        lesion_encoder = args.get('lesion_encoder', _legacy_encoder)
        if any(k.startswith('optic_unet.swin_encoder.') for k in state_dict):
            raise ValueError(
                f"{os.path.basename(checkpoint_path)} contains a Swin-based OpticUNet. "
                "OpticUNet supports only the ResNet-50 encoder.")
        image_size = args.get('image_size', 1024)
        feature_dim = args.get('feature_dim', 256)
        num_gnn_layers = args.get('num_gnn_layers', 3)
        gnn_dropout = args.get('gnn_dropout', 0.1)
        topk_lesions_per_cluster = args.get('topk_lesions_per_cluster', None)
        num_region_nodes = args.get('num_region_nodes', 4)
        region_min_lesions = args.get('region_min_lesions', 3)
        use_prior_fusion = args.get('use_prior_fusion', args.get('use_pos_encoding', True))
        # prefer fusion_mode from args; if absent (older checkpoint), infer it from
        # the input dim of pos_proj.0.weight in the state_dict:
        #   feature_dim + 2 → pos_only
        #   feature_dim + 7 → full
        if 'fusion_mode' in args:
            fusion_mode = args['fusion_mode']
        elif 'pos_proj.0.weight' in state_dict:
            extra_dim = state_dict['pos_proj.0.weight'].shape[1] - feature_dim
            fusion_mode = 'pos_only' if extra_dim <= 2 else 'full'
            if verbose:
                print(f"  [load_from_checkpoint] fusion_mode auto-detected: '{fusion_mode}' "
                      f"(pos_proj input extra_dim={extra_dim})")
        else:
            fusion_mode = 'full'
        use_optic_nodes = args.get('use_optic_nodes', True)
        use_gating = args.get('use_gating', True)
        # default True so older checkpoints (trained with the dual branch, no arg
        # saved) reproduce their original behavior
        use_global_logits = args.get('use_global_logits', True)
        # ablation switches: default to the full-model behavior so older checkpoints
        # (without these args) rebuild the original architecture
        use_global_node = args.get('use_global_node', True)
        use_region_nodes = args.get('use_region_nodes', True)
        use_od_posbias = args.get('use_od_posbias', True)
        lesion_direct = args.get('lesion_direct', False)
        num_classes = 3 if args.get('use_3_stages', False) else 5

        # detect pos_proj structure: old structure has pos_proj.1 = LayerNorm
        # (weight shape=[feature_dim]); new structure has pos_proj.1 = GELU
        # (no params), pos_proj.3 = Linear
        pos_proj_ln = ('pos_proj.1.weight' in state_dict and
                       state_dict['pos_proj.1.weight'].dim() == 1)
        if verbose and pos_proj_ln:
            print(f"  [load_from_checkpoint] pos_proj: old structure (Linear+LayerNorm) detected")

        # ---- decide which submodels are needed from the state_dict keys ----
        has_lesion = any(k.startswith('lesion_unet.') for k in state_dict)

        # OpticUNet (always needed)
        optic_unet = OpticUNet(
            encoder_pretrained=False,
            predict_fovea=True,
            encoder_name=optic_encoder,
            img_size=image_size
        )

        # LesionUNet (as needed)
        lesion_unet = None
        if has_lesion:
            # auto-detect large_decoder (January version) directly from this
            # checkpoint's own state_dict, so loading is self-contained (no need
            # for the external lesion_checkpoint file referenced in the saved args).
            _bias = state_dict.get('lesion_unet.decoder_stage4.1.bias', None)
            _large_decoder = bool(_bias is not None and _bias.shape[0] > 256)
            lesion_unet = LesionUNet(
                num_lesions=4,
                encoder_pretrained=False,
                encoder_name=lesion_encoder,
                img_size=image_size,
                large_decoder=_large_decoder,
            )

        # ---- assemble PARAGraph and load weights ----
        model = cls(
            optic_unet=optic_unet,
            lesion_unet=lesion_unet,
            feature_dim=feature_dim,
            num_gnn_layers=num_gnn_layers,
            num_classes=num_classes,
            gnn_dropout=gnn_dropout,
            num_region_nodes=num_region_nodes,
            region_min_lesions=region_min_lesions,
            topk_lesions_per_cluster=topk_lesions_per_cluster,
            use_prior_fusion=use_prior_fusion,
            fusion_mode=fusion_mode,
            use_optic_nodes=use_optic_nodes,
            use_gating=use_gating,
            pos_proj_ln=pos_proj_ln,
            use_global_logits=use_global_logits,
            use_global_node=use_global_node,
            use_region_nodes=use_region_nodes,
            use_od_posbias=use_od_posbias,
            lesion_direct=lesion_direct,
        )

        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

        model = model.to(device)

        if eval_mode:
            model.eval()

        if verbose:
            print(f"[PARAGraph] Loaded from: {os.path.basename(checkpoint_path)}")
            print(f"  OpticEncoder: {optic_encoder}, LesionEncoder: {lesion_encoder}, "
                  f"Classes: {num_classes}, GNN layers: {num_gnn_layers}, Feature dim: {feature_dim}")
            print(f"  num_region_nodes: {num_region_nodes}, "
                  f"region_min_lesions: {region_min_lesions}")
            print(f"  LesionUNet: {'yes' if has_lesion else 'no'}")
            if missing_keys:
                print(f"  [WARN] missing keys ({len(missing_keys)}): "
                      f"{missing_keys[:8]}{'...' if len(missing_keys) > 8 else ''}")
            if unexpected_keys:
                print(f"  [WARN] unexpected keys ({len(unexpected_keys)}): "
                      f"{unexpected_keys[:8]}{'...' if len(unexpected_keys) > 8 else ''}")
            if not missing_keys and not unexpected_keys:
                print(f"  state_dict loaded perfectly (no missing / unexpected keys)")

        return model

    def freeze_backbones(self):
        """Freeze all OpticUNet and LesionUNet parameters."""
        for unet in [self.optic_unet, self.lesion_unet]:
            if unet is not None:
                for param in unet.parameters():
                    param.requires_grad = False
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Backbones frozen. Trainable parameters: {trainable:,}")

    def forward(
        self,
        x: torch.Tensor,
        include_lesions: bool = True,
        od_mask_cached: torch.Tensor = None,
        landmark_heatmap_cached: torch.Tensor = None,
        lesion_masks_cached: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Dual-branch forward pass.

        Args:
            x: [B, 3, H, W] input image
            include_lesions: whether to run the lesion branch
            od_mask_cached: [B, H, W] precomputed OD segmentation logits (optional)
            landmark_heatmap_cached: [B, H, W] precomputed fovea heatmap logits (optional)
            lesion_masks_cached: [B, 4, H, W] precomputed lesion segmentation logits (optional)

        Returns:
            Dict:
                - 'logits':            [B, num_classes]  (anatomical_logits + global_logits)
                - 'anatomical_logits': [B, num_classes]
                - 'global_logits':     [B, num_classes]
                - 'graphs':            list of per-sample graph dicts
                - 'probs':             [B, num_classes]
        """
        B, _, H, W = x.shape
        device = x.device

        _backbone_frozen = not any(p.requires_grad for p in self.optic_unet.parameters())

        # OpticUNet encoder runs only when its decode output is needed (i.e. OD or
        # fovea is not cached). Caching both lets lesion-intervention loops skip it.
        optic_features = None
        if od_mask_cached is None or landmark_heatmap_cached is None:
            if _backbone_frozen:
                with torch.no_grad():
                    optic_features, _ = self.optic_unet.encode(x)
            else:
                optic_features, _ = self.optic_unet.encode(x)

        # OpticUNet decoder: use cached masks or run live inference
        if od_mask_cached is not None:
            od_mask = od_mask_cached.unsqueeze(1).to(device)  # [B, H, W] -> [B, 1, H, W]
        else:
            if _backbone_frozen:
                with torch.no_grad():
                    od_mask = self.optic_unet.decode_optic_disc(optic_features)
            else:
                od_mask = self.optic_unet.decode_optic_disc(optic_features)

        if landmark_heatmap_cached is not None:
            landmark_heatmap = landmark_heatmap_cached.unsqueeze(1).to(device)  # [B, H, W] -> [B, 1, H, W]
        else:
            if _backbone_frozen:
                with torch.no_grad():
                    landmark_heatmap = self.optic_unet.decode_fovea(optic_features)
            else:
                landmark_heatmap = self.optic_unet.decode_fovea(optic_features)

        # morphological processing of the OD
        processed_od = OpticUNet.morphological_process_od(od_mask)

        # OD confidence and center
        od_conf, od_infos = OpticUNet.compute_od_confidence(processed_od)

        # LesionUNet — used only to produce lesion masks (node visual features
        # are sampled from global_swin, not from the LesionUNet encoder).
        lesion_output = None
        if include_lesions and self.lesion_unet is not None:
            _lesion_frozen = not any(p.requires_grad for p in self.lesion_unet.parameters())

            _ctx = torch.no_grad() if _lesion_frozen else torch.enable_grad()
            with _ctx:
                lesion_features_enc, _ = self.lesion_unet.encode(x)
                masks_raw = self.lesion_unet.decode(lesion_features_enc)
                lesion_decoded_masks = lesion_masks_cached.to(device) \
                    if lesion_masks_cached is not None else masks_raw

            lesion_output = {
                'masks': lesion_decoded_masks,   # [B, 4, H, W]
            }

        # ── Global node features: trainable Swin-T + OD pos bias ─────────────────
        import math as _math
        fovea_centers_512 = []
        od_centers_512    = []
        od_diameters_512  = []
        scale_512 = self.global_img_size / max(H, W)

        for _b in range(B):
            _lm = landmark_heatmap[_b]
            if _lm.dim() == 2: _lm = _lm.unsqueeze(0)
            _, _fH, _fW = _lm.shape
            _yy = torch.arange(_fH, device=device, dtype=torch.float32).view(-1,1).expand(_fH,_fW)
            _xx = torch.arange(_fW, device=device, dtype=torch.float32).view(1,-1).expand(_fH,_fW)
            _p  = torch.softmax(_lm[0].flatten(), dim=0).view(_fH, _fW)
            fovea_centers_512.append(
                ((_xx * _p).sum().item() * scale_512, (_yy * _p).sum().item() * scale_512)
            )
            od_info = od_infos[_b]
            od_c = od_info['center'] if od_info['detected'] else (W/2.0, H/2.0)
            area = od_info.get('area', 0)
            od_d = 2.0 * _math.sqrt(area / _math.pi) if area > 0 else H / 10.0
            od_centers_512.append((od_c[0] * scale_512, od_c[1] * scale_512))
            od_diameters_512.append(od_d * scale_512)

        x_512 = F.interpolate(x, size=(self.global_img_size, self.global_img_size),
                               mode='bilinear', align_corners=False)
        global_raw_feats, global_stage_feats = self.global_swin.forward_features_with_stages(
            x_512, od_centers_512, od_diameters_512, fovea_centers_512,
            add_od_bias=self.use_od_posbias,
        )  # [B, num_features], {'stage2':..,'stage3':..,'stage4':..}
        global_logits     = self.global_swin.head(global_raw_feats)   # [B, num_classes]
        global_feat_batch = self.global_node_proj(global_raw_feats)   # [B, feature_dim]

        # ── Anatomical branch: full hierarchical GNN ─────────────────────────
        graphs = []

        for b in range(B):
            od_info = od_infos[b]
            if od_info['detected']:
                od_center = od_info['center']
            else:
                od_center = (W // 2, H // 2)

            # OD diameter
            od_mask_b  = processed_od[b, 0].cpu().numpy()
            od_diameter = self.compute_od_diameter(od_mask_b)
            if od_diameter < 1.0:
                od_diameter = self.default_od_diameter

            # global_swin stage features for this sample (shared by OD node + lesion nodes)
            single_global_f = {k: v[b:b+1] for k, v in global_stage_feats.items()
                               if v is not None}

            # OD feature from global_swin at OD center (same OD-aware representation)
            od_feature = self.extract_lesion_features(
                single_global_f, [{'center': od_center}], (H, W),
                feat_proj=self.global_lesion_feature_proj,
                feat_fuse=self.global_lesion_feature_fuse,
            )[0]

            graph_node = {
                'type': self.NODE_GRAPH,
                'coords': (0.0, 0.0),
                'absolute_coords': od_center,
                'confidence': float(od_conf[b]),
                'detected': od_info['detected'],
                'info': od_info,
                'feature': od_feature,
                'gate': self.compute_gate(od_conf[b]),  # placeholder; updated below
            }

            # Fovea: soft argmax → coordinate system reference
            lm_logits = landmark_heatmap[b]
            if lm_logits.dim() == 2:
                lm_logits = lm_logits.unsqueeze(0)
            _, fH, fW = lm_logits.shape
            yy = torch.arange(fH, device=device, dtype=torch.float32).view(-1,1).expand(fH,fW)
            xx = torch.arange(fW, device=device, dtype=torch.float32).view(1,-1).expand(fH,fW)
            posterior  = torch.softmax(lm_logits[0].flatten(), dim=0).view(fH, fW)
            lm_abs_x   = (xx * posterior).sum()
            lm_abs_y   = (yy * posterior).sum()
            dx_lm = lm_abs_x - od_center[0]
            dy_lm = lm_abs_y - od_center[1]
            lm_dist  = torch.sqrt(dx_lm**2 + dy_lm**2)
            lm_dist_n = (lm_dist / od_diameter).detach().item()

            # Global node feature: from trainable global_swin (OD pos bias)
            lm_feature = global_feat_batch[b]  # [D]

            global_node = {
                'type': self.NODE_GLOBAL,
                'coords': (lm_dist_n, 0.0),
                'absolute_coords': (lm_abs_x.detach().item(), lm_abs_y.detach().item()),
                'confidence': 1.0,
                'detected': True,
                'feature': lm_feature,
                'gate': self.compute_gate(torch.tensor(1.0, device=device, dtype=torch.float32)),
            }

            # OD ↔ Landmark edges
            graph = {
                'nodes': [graph_node, global_node],
                'edges': [
                    {'from': self.NODE_GRAPH,  'to': self.NODE_GLOBAL,
                     'distance': lm_dist_n, 'edge_type': 'global_graph'},
                    {'from': self.NODE_GLOBAL, 'to': self.NODE_GRAPH,
                     'distance': lm_dist_n, 'edge_type': 'graph_global'},
                ],
                'od_center':        np.array(od_center, dtype=np.float32),
                'od_detected':      od_info['detected'],
                'od_confidence':    float(od_conf[b]),
                'od_diameter':      float(od_diameter),
                'landmark_center':  np.array([lm_abs_x.detach().item(),
                                              lm_abs_y.detach().item()], dtype=np.float32),
                'image_size':       (H, W),
                '_device':          device,
            }

            # OD-Fovea reference direction (for polar position encoding)
            ref_dx = float(dx_lm.detach())
            ref_dy = float(dy_lm.detach())
            ref_len = np.sqrt(ref_dx**2 + ref_dy**2) + 1e-8
            ref_x, ref_y = ref_dx / ref_len, ref_dy / ref_len

            # Lesion nodes with position-gated feature modulation
            if include_lesions and lesion_output is not None:
                lesion_masks_b  = lesion_output['masks'][b]
                lesion_probs_b  = torch.sigmoid(lesion_masks_b)

                all_lesion_nodes  = []
                all_lesion_feats  = []
                lesion_edges      = []
                node_idx          = 2

                for cls_idx in range(4):
                    class_mask = (lesion_probs_b[cls_idx] > self.lesion_threshold).cpu().numpy()
                    class_prob_np = lesion_probs_b[cls_idx].detach().cpu().numpy()

                    regions = self.extract_lesion_regions(class_mask, class_prob_np, cls_idx)
                    if not regions:
                        continue

                    enc_feats = self.extract_lesion_features(
                        single_global_f, regions, (H, W),
                        feat_proj=self.global_lesion_feature_proj,
                        feat_fuse=self.global_lesion_feature_fuse,
                    )

                    for i, region in enumerate(regions):
                        cx, cy = region['center']
                        dx = cx - od_center[0]
                        dy = cy - od_center[1]
                        dist = np.sqrt(dx**2 + dy**2)
                        r_od  = dist / (od_diameter + 1e-8)
                        area_n = region.get('area', 1) / (od_diameter**2 + 1e-8)
                        lesion_feat = enc_feats[i]

                        # Medical prior for GNN node (used in pos_proj)
                        dist_n_od = r_od
                        graph_cx  = dx / od_diameter
                        graph_cy  = dy / od_diameter
                        type_oh   = [0.0] * 4; type_oh[cls_idx] = 1.0

                        lesion_conf = float(region.get('confidence', 0.5))
                        lesion_node = {
                            'type':            self.LESION_NODE_TYPES[cls_idx],
                            'coords':          (graph_cx, graph_cy),
                            'absolute_coords': (cx, cy),
                            'lesion_class':    cls_idx,
                            'lesion_name':     LESION_NAMES[cls_idx],
                            'area_normalized': float(area_n),
                            'confidence':      lesion_conf,
                            'feature_idx':     len(all_lesion_feats),
                            'node_idx':        node_idx,
                            'gate':            self.compute_gate(torch.tensor(lesion_conf, device=device, dtype=torch.float32)),
                        }

                        if self.lesion_direct:
                            # Hierarchy-ladder L1: every lesion connects directly to
                            # the OD (and Global) hub; no Category/Region aggregation.
                            lesion_edges += [
                                {'from': self.NODE_GRAPH, 'to': node_idx,
                                 'edge_type': 'graph_to_lesion_direct'},
                                {'from': node_idx, 'to': self.NODE_GRAPH,
                                 'edge_type': 'lesion_to_graph_direct'},
                            ]
                            if self.use_global_node:
                                lesion_edges += [
                                    {'from': self.NODE_GLOBAL, 'to': node_idx,
                                     'edge_type': 'global_to_lesion_direct'},
                                    {'from': node_idx, 'to': self.NODE_GLOBAL,
                                     'edge_type': 'lesion_to_global_direct'},
                                ]
                        elif dist_n_od < self.edge_distance_threshold:
                            # OD → lesion shortcut edge for near-OD lesions
                            lesion_edges += [
                                {'from': self.NODE_GRAPH, 'to': node_idx,
                                 'edge_type': 'graph_to_lesion_direct'},
                                {'from': node_idx, 'to': self.NODE_GRAPH,
                                 'edge_type': 'lesion_to_graph_direct'},
                            ]

                        all_lesion_nodes.append(lesion_node)
                        all_lesion_feats.append(lesion_feat)
                        node_idx += 1

                graph['lesion_nodes']    = all_lesion_nodes
                graph['lesion_features'] = (torch.stack(all_lesion_feats)
                                             if all_lesion_feats else
                                             torch.zeros(0, self.feature_dim, device=device))
                graph['lesion_edges']    = lesion_edges
                graph['num_lesions']     = len(all_lesion_nodes)

                # Category tokens (class tokens, no position — aggregate pos-gated feats)
                # Skipped entirely under lesion_direct (hierarchy-ladder L1).
                category_nodes = []
                for cls_idx in range(4):
                    if self.lesion_direct:
                        break
                    cls_lesions = [n for n in all_lesion_nodes if n['lesion_class'] == cls_idx]
                    if not cls_lesions:
                        continue

                    cls_feats = torch.stack([
                        all_lesion_feats[n['feature_idx']] for n in cls_lesions
                    ])  # [N_cls, D]
                    # Hardcoded: confidence-weighted mean of detected lesion features
                    cls_confs = torch.tensor(
                        [n['confidence'] for n in cls_lesions],
                        device=device, dtype=torch.float32
                    )
                    cls_weights = cls_confs / cls_confs.sum().clamp(min=1e-6)
                    token_feat = (cls_feats * cls_weights.unsqueeze(1)).sum(0)  # [D]

                    # cumulative confidence: mean of child lesion nodes
                    cls_conf = float(np.mean([n['confidence'] for n in cls_lesions]))
                    cls_gate = self.compute_gate(
                        torch.tensor(cls_conf, device=device, dtype=torch.float32)
                    )
                    category_node = {
                        'type':       self.CATEGORY_NODE_TYPES[cls_idx],
                        'node_idx':   node_idx,
                        'feature':    token_feat,
                        'confidence': cls_conf,
                        'gate':       cls_gate,
                    }
                    # Graph/Global ↔ category node edges
                    lname = LESION_NAMES[cls_idx].lower()
                    _cat_hubs = [(self.NODE_GRAPH,  f'graph_to_{lname}_cat', f'{lname}_cat_to_graph')]
                    if self.use_global_node:
                        _cat_hubs.append(
                            (self.NODE_GLOBAL, f'global_to_{lname}_cat', f'{lname}_cat_to_global'))
                    for src, et_fwd, et_rev in _cat_hubs:
                        lesion_edges += [
                            {'from': src,      'to': node_idx,  'edge_type': et_fwd},
                            {'from': node_idx, 'to': src,       'edge_type': et_rev},
                        ]
                    # category ↔ its lesion nodes
                    for n in cls_lesions:
                        lesion_edges += [
                            {'from': node_idx, 'to': n['node_idx'],
                             'edge_type': f'{lname}_cat_to_lesion'},
                            {'from': n['node_idx'], 'to': node_idx,
                             'edge_type': f'lesion_to_{lname}_cat'},
                        ]
                    category_nodes.append(category_node)
                    node_idx += 1

                graph['category_nodes']  = category_nodes
                graph['lesion_edges'] = lesion_edges
                graph['num_categories']   = len(category_nodes)

                # Region nodes (K-means on lesion positions)
                if (self.use_region_nodes and not self.lesion_direct
                        and len(all_lesion_nodes) >= self.region_min_lesions):
                    landmark_center = graph['landmark_center']
                    lesion_feats_tensor = graph['lesion_features']
                    region_nodes, region_feats = self.compute_region_nodes(
                        all_lesion_nodes, lesion_feats_tensor,
                        od_center, tuple(landmark_center), od_diameter
                    )
                    # assign node indices and feature_idx
                    for i, rn in enumerate(region_nodes):
                        rn['node_idx']    = node_idx + i
                        rn['feature_idx'] = i
                        rn_conf = torch.tensor(rn['confidence'], device=device, dtype=torch.float32)
                        rn['gate'] = self.compute_gate(rn_conf)
                    graph['region_nodes']    = region_nodes
                    graph['region_features'] = region_feats
                    graph['num_regions']     = len(region_nodes)
                    node_idx += len(region_nodes)

                    # Region ↔ Graph/Global bidirectional edges
                    for rn in region_nodes:
                        ri = rn['node_idx']
                        lesion_edges += [
                            {'from': self.NODE_GRAPH,  'to': ri, 'edge_type': 'graph_to_region'},
                            {'from': ri, 'to': self.NODE_GRAPH,  'edge_type': 'region_to_graph'},
                        ]
                        if self.use_global_node:
                            lesion_edges += [
                                {'from': self.NODE_GLOBAL, 'to': ri, 'edge_type': 'global_to_region'},
                                {'from': ri, 'to': self.NODE_GLOBAL, 'edge_type': 'region_to_global'},
                            ]
                    graph['lesion_edges'] = lesion_edges
                else:
                    graph['region_nodes'] = []
                    graph['num_regions']  = 0
            else:
                graph['lesion_nodes']    = []
                graph['lesion_features'] = torch.zeros(0, self.feature_dim, device=device)
                graph['lesion_edges']    = []
                graph['category_nodes']     = []
                graph['region_nodes']    = []
                graph['num_lesions']     = 0
                graph['num_categories']      = 0
                graph['num_regions']     = 0

            # ── Cumulative confidence: OD node accumulates from token nodes ──────
            token_confs = [n['confidence'] for n in graph.get('category_nodes', [])]
            if token_confs:
                od_accum_conf = float(np.mean(token_confs))
            else:
                # fallback: accumulate from lesion nodes directly
                lesion_confs = [n['confidence'] for n in graph.get('lesion_nodes', [])]
                od_accum_conf = float(np.mean(lesion_confs)) if lesion_confs else float(od_conf[b])
            od_accum_conf_t = torch.tensor(od_accum_conf, device=device, dtype=torch.float32)
            graph_node['confidence'] = od_accum_conf
            graph_node['gate']       = self.compute_gate(od_accum_conf_t)

            # Gate map: each node uses its own gate
            node_gates = {
                self.NODE_GRAPH:       graph_node['gate'],
                self.NODE_GLOBAL: global_node['gate'],
            }
            for n in graph.get('lesion_nodes', []):
                node_gates[n['node_idx']] = n['gate']
            for n in graph.get('category_nodes', []):
                node_gates[n['node_idx']] = n['gate']
            for n in graph.get('region_nodes', []):
                node_gates[n['node_idx']] = n['gate']
            graph['node_gates'] = node_gates
            graph['graph_node_confidence'] = od_accum_conf   # for fusion weight

            graphs.append(graph)

        # ── GNN message passing ───────────────────────────────────────────────
        batch_anatomical_logits = []
        batch_graph_features    = []

        for graph in graphs:
            node_features, edge_index = self.get_node_features_for_gnn(graph)
            edge_weights   = self.get_edge_weights_for_gnn(graph)
            edge_type_bias = self.get_edge_type_bias_for_gnn(graph)

            h = F.dropout(node_features, p=self.gnn_dropout, training=self.training)
            for gnn_layer, norm in zip(self.gnn_layers, self.gnn_norms):
                if edge_index.shape[1] > 0:
                    h_new = gnn_layer(h, edge_index, edge_weights, edge_type_bias)
                    h     = h + norm(F.gelu(h_new))

            graph['node_features_after_gnn'] = h

            # Graph pool: Graph node + Global node + category nodes + region nodes
            num_lesions    = graph.get('num_lesions', 0)
            num_categories = graph.get('num_categories', 0)
            num_regions    = graph.get('num_regions', 0)
            token_start    = 2 + num_lesions
            region_start   = token_start + num_categories

            if self.use_optic_nodes:
                key_idx = [0, 1] if self.use_global_node else [0]
            else:
                key_idx = []
            if num_categories > 0: key_idx += list(range(token_start,  token_start  + num_categories))
            if num_regions    > 0: key_idx += list(range(region_start, region_start + num_regions))
            if not key_idx:        key_idx  = list(range(len(h)))

            key_feats = h[key_idx].unsqueeze(0)                       # [1, K, D]
            queries   = self.graph_pool_queries.unsqueeze(0)          # [1, C, D]
            pooled, _ = self.graph_pool_attn(queries, key_feats, key_feats)
            pooled    = pooled.squeeze(0)                              # [C, D]

            pooled_concat = pooled.flatten()                           # [C*D]
            graph_feat    = self.graph_proj(pooled_concat)            # [D]
            anatomical_logits = self.shared_classifier(graph_feat)    # [num_classes]

            batch_anatomical_logits.append(anatomical_logits)
            batch_graph_features.append(pooled_concat)

        # ── Final output ─────────────────────────────────────────────────────────
        # Dual-branch (use_global_logits=True): logits = anatomical + global.
        # Single-branch (False): logits = anatomical only (global_swin is just a
        # feature extractor; global_logits is still returned for analysis).
        anatomical_logits = torch.stack(batch_anatomical_logits, dim=0)  # [B, C]
        if self.use_global_logits:
            logits = anatomical_logits + global_logits
        else:
            logits = anatomical_logits

        return {
            'logits':            logits,
            'anatomical_logits': anatomical_logits,
            'global_logits':     global_logits,
            'graphs':            graphs,
            'probs':             F.softmax(logits, dim=-1),
            # graph-level pooled features (before graph_proj + classifier), for t-SNE
            'graph_features':    torch.stack(batch_graph_features, dim=0),
        }

