"""
PARAGraph Features Mixin

Methods for feature extraction, node construction, and coordinate transforms.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from ._paragraph_constants import (
    NODE_GRAPH, NODE_GLOBAL, NODE_REGION,
    NODE_OD, NODE_LANDMARK,   # backward-compat aliases
    NODE_MA, NODE_HE, NODE_EX, NODE_SE,
    NODE_MA_CAT, NODE_HE_CAT, NODE_EX_CAT, NODE_SE_CAT,
    NODE_MA_TOKEN, NODE_HE_TOKEN, NODE_EX_TOKEN, NODE_SE_TOKEN,  # aliases
    LESION_NAMES
)

if TYPE_CHECKING:
    from .paragraph import PARAGraph


class PARAGraphFeaturesMixin:
    """Mixin: feature extraction, node construction, coordinate transforms."""

    def compute_gate(self, confidence, node_type: int = None):
        """
        Compute the soft gate value g = σ(α · c + β).

        Each node uses its own reliability c to compute the gate:
          - lesion node: that lesion's segmentation confidence
          - category / region node: the (mean) confidence of its member lesions
          - OD (graph) node: the accumulated category confidence
        α, β are shared learnable scalars. The edge weight = g_from · g_to, so a
        message is trusted only when both endpoint nodes are reliable.

        Assumption: segmentation is mostly reliable but still makes errors; the gate
        suppresses low-confidence (likely-wrong) nodes so they contribute less.

        Args:
            confidence: this node's reliability confidence (scalar, float or tensor)
            node_type: node type (kept for API, currently unused)

        Returns:
            gate: gate value tensor (0-1)
        """
        if not self.use_gating:
            device = next(self.parameters()).device
            return torch.ones((), device=device)

        alpha = self.od_gate_alpha
        beta  = self.od_gate_beta

        if not torch.is_tensor(confidence):
            confidence = torch.tensor(confidence, device=alpha.device, dtype=torch.float32)

        return torch.sigmoid(alpha * confidence + beta)

    def extract_lesion_regions(
        self,
        lesion_mask: np.ndarray,
        lesion_probs: np.ndarray,
        lesion_class_idx: int
    ) -> List[Dict]:
        """
        Extract lesion regions from a segmentation mask.

        Args:
            lesion_mask: [H, W] binary mask (0 or 1)
            lesion_probs: [H, W] segmentation probability map
            lesion_class_idx: lesion class index (0=MA, 1=HE, 2=EX, 3=SE)

        Returns:
            regions: List[Dict], info for each region
        """
        # ensure uint8 type
        binary_mask = (lesion_mask > 0).astype(np.uint8)

        # connected-component analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        regions = []

        # area threshold depends on lesion type
        # MA (lesion_class_idx=0) is usually very small, use a separate threshold
        min_area = self.min_ma_area if lesion_class_idx == 0 else self.min_lesion_area

        # skip background (label 0)
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]

            # filter out too-small regions (MA uses a smaller threshold)
            if area < min_area:
                continue

            # region mask
            region_mask = (labels == label_id)

            # centroid
            cx, cy = centroids[label_id]

            # mean confidence over the region
            region_probs = lesion_probs[region_mask]
            confidence = float(np.mean(region_probs))

            # bounding box
            x, y, w, h = (
                stats[label_id, cv2.CC_STAT_LEFT],
                stats[label_id, cv2.CC_STAT_TOP],
                stats[label_id, cv2.CC_STAT_WIDTH],
                stats[label_id, cv2.CC_STAT_HEIGHT]
            )

            regions.append({
                'center': (float(cx), float(cy)),
                'area': int(area),
                'confidence': confidence,
                'bbox': (x, y, w, h),
                'mask': region_mask,
                'lesion_class': lesion_class_idx,
                'lesion_name': LESION_NAMES[lesion_class_idx]
            })

        # sort by area, keep the largest N
        regions = sorted(regions, key=lambda r: r['area'], reverse=True)
        regions = regions[:self.max_lesions_per_class]

        return regions

    def extract_lesion_features(
        self,
        features: Dict[str, torch.Tensor],
        regions: List[Dict],
        image_size: Tuple[int, int],
        feat_proj,
        feat_fuse,
    ) -> torch.Tensor:
        """
        Extract a feature vector for each lesion region from encoder features.

        Features are sampled from the provided multi-scale stage feature maps
        (global_swin stage2/3/4) at each region center via grid_sample, then fused.

        Args:
            features: multi-scale stage features (stage2/stage3/stage4)
            regions: list of lesion regions
            image_size: original image size (H, W)
            feat_proj: ModuleDict of per-stage 1x1 conv projections
            feat_fuse: module fusing the concatenated multi-scale features

        Returns:
            lesion_features: [N, feature_dim] feature vector per region
        """
        if len(regions) == 0:
            return torch.zeros(0, self.feature_dim, device=next(self.parameters()).device)

        H, W = image_size
        device = features['stage2'].device

        proj = feat_proj
        fuse = feat_fuse

        # project each stage's features
        feat2 = proj['stage2'](features['stage2'])  # [1, D, H/8, W/8]
        feat3 = proj['stage3'](features['stage3'])  # [1, D, H/16, W/16]
        feat4 = proj['stage4'](features['stage4'])  # [1, D, H/32, W/32]

        lesion_features = []

        for region in regions:
            cx, cy = region['center']

            # normalize coordinates to [-1, 1]
            nx = (cx / W) * 2 - 1
            ny = (cy / H) * 2 - 1
            grid = torch.tensor([[[[nx, ny]]]], device=device, dtype=torch.float32)

            # sample features from each stage (grid_sample)
            f2 = F.grid_sample(feat2, grid, mode='bilinear', align_corners=True)  # [1, D, 1, 1]
            f3 = F.grid_sample(feat3, grid, mode='bilinear', align_corners=True)
            f4 = F.grid_sample(feat4, grid, mode='bilinear', align_corners=True)

            # concatenate multi-scale features
            f2 = f2.view(-1)  # [D]
            f3 = f3.view(-1)
            f4 = f4.view(-1)
            multi_scale_feat = torch.cat([f2, f3, f4], dim=0)  # [3*D]

            # fuse
            fused_feat = fuse(multi_scale_feat)  # [D]
            lesion_features.append(fused_feat)

        return torch.stack(lesion_features, dim=0)  # [N, D]

    def compute_region_nodes(
        self,
        lesion_nodes: List[Dict],
        lesion_features: torch.Tensor,
        od_center: Tuple[float, float],
        landmark_center: Tuple[float, float],
        od_diameter: float
    ) -> Tuple[List[Dict], torch.Tensor]:
        """
        Compute lesion-enriched Region nodes via K-means clustering.

        Region nodes are built as follows:
        1. K-means clustering over all lesion nodes' graph coordinates
        2. Each cluster center's coordinates use the mean distance within the cluster
        3. Region features aggregate the cluster's lesion features via attention

        Args:
            lesion_nodes: list of lesion nodes
            lesion_features: [N, feature_dim] lesion feature matrix
            od_center: OD center pixel coordinates
            landmark_center: landmark (fovea) center pixel coordinates
            od_diameter: OD diameter (pixels)

        Returns:
            region_nodes: List[Dict] of region nodes
            region_features: [K, feature_dim] region feature matrix
        """
        if len(lesion_nodes) < self.region_min_lesions:
            # too few lesions, do not create Region nodes
            device = lesion_features.device if lesion_features is not None and len(lesion_features) > 0 else next(self.parameters()).device
            return [], torch.zeros(0, self.feature_dim, device=device)

        device = lesion_features.device

        # collect all lesion coordinates
        lesion_coords = np.array([node['coords'] for node in lesion_nodes], dtype=np.float32)  # [N, 2] graph coords
        lesion_abs_coords = np.array([node['absolute_coords'] for node in lesion_nodes], dtype=np.float32)  # [N, 2] pixel coords
        lesion_confidences = np.array([node['confidence'] for node in lesion_nodes], dtype=np.float32)  # [N]

        # ablation: use pixel coords for clustering when use_prior_fusion=False
        clustering_coords = lesion_coords if self.use_prior_fusion else lesion_abs_coords

        # determine the actual number of clusters (cannot exceed #lesions or #unique coords)
        n_lesions = len(lesion_nodes)
        # round to 4 decimals before counting unique coords, to avoid float error
        n_unique = len(np.unique(np.round(clustering_coords, 4), axis=0))
        k = min(self.num_region_nodes, n_lesions, n_unique)

        if k < 1:
            return [], torch.zeros(0, self.feature_dim, device=device)

        # K-means clustering
        import warnings
        from sklearn.cluster import KMeans
        from sklearn.exceptions import ConvergenceWarning

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(clustering_coords)
        cluster_centers = kmeans.cluster_centers_  # [K, 2]

        region_nodes = []
        region_features_list = []

        for cluster_idx in range(k):
            # get this cluster's lesion nodes
            cluster_mask = (cluster_labels == cluster_idx)
            cluster_lesion_indices = np.where(cluster_mask)[0]

            if len(cluster_lesion_indices) < 1:
                continue

            # coords and confidences of lesions in this cluster
            cluster_coords = lesion_coords[cluster_mask]  # [M, 2] graph coords
            cluster_abs_coords = lesion_abs_coords[cluster_mask]  # [M, 2] pixel coords
            cluster_confs = lesion_confidences[cluster_mask]  # [M]
            # ablation: use pixel coords when use_prior_fusion=False
            cluster_spatial_coords = cluster_coords if self.use_prior_fusion else cluster_abs_coords

            # build coords via confidence-weighted mean
            if cluster_confs.sum() > 0:
                weights = cluster_confs / cluster_confs.sum()
                # graph coords: weighted mean
                region_graph_x = float(np.sum(cluster_coords[:, 0] * weights))
                region_graph_y = float(np.sum(cluster_coords[:, 1] * weights))
                # absolute coords: weighted mean
                region_abs_x = float(np.sum(cluster_abs_coords[:, 0] * weights))
                region_abs_y = float(np.sum(cluster_abs_coords[:, 1] * weights))
            else:
                # no confidence info, use simple mean
                region_graph_x = float(np.mean(cluster_coords[:, 0]))
                region_graph_y = float(np.mean(cluster_coords[:, 1]))
                region_abs_x = float(np.mean(cluster_abs_coords[:, 0]))
                region_abs_y = float(np.mean(cluster_abs_coords[:, 1]))

            # ablation: region does not store graph coords when use_prior_fusion=False
            if self.use_prior_fusion:
                region_coords = (region_graph_x, region_graph_y)
            else:
                region_coords = (0.0, 0.0)
            region_abs_coords_tuple = (region_abs_x, region_abs_y)

            # distances from this region to OD and landmark
            if self.use_prior_fusion:
                dist_to_od = float(np.sqrt(region_graph_x**2 + region_graph_y**2))
                landmark_dist_norm = np.sqrt((landmark_center[0] - od_center[0])**2 + (landmark_center[1] - od_center[1])**2) / od_diameter
                dist_to_landmark = float(np.sqrt((region_graph_x - landmark_dist_norm)**2 + region_graph_y**2))
            else:
                dist_to_od = 0.0
                dist_to_landmark = 0.0

            # mean radius of this cluster (mean distance to the cluster center)
            center_x, center_y = cluster_centers[cluster_idx]
            distances_to_center = np.sqrt((cluster_spatial_coords[:, 0] - center_x)**2 + (cluster_spatial_coords[:, 1] - center_y)**2)
            avg_radius = float(np.mean(distances_to_center))

            # lesion density (#lesions in this cluster / area)
            # circular approximation: area = π * r²
            area = np.pi * (avg_radius + 0.1)**2  # +0.1 avoids divide-by-zero
            density = len(cluster_lesion_indices) / area

            # aggregate the cluster's lesion features
            cluster_lesion_features = lesion_features[cluster_lesion_indices]  # [M, feature_dim]

            # Top-k selection: keep the k highest-confidence lesion nodes
            if self.topk_lesions_per_cluster is not None and len(cluster_lesion_indices) > self.topk_lesions_per_cluster:
                topk_confs = cluster_confs  # [M]
                topk_indices = np.argsort(topk_confs)[-self.topk_lesions_per_cluster:]  # pick top-k
                cluster_lesion_features = cluster_lesion_features[topk_indices]  # [k, feature_dim]

            # aggregate via the region token (attention)
            region_token = self.region_tokens.to(device)  # [1, feature_dim], shared query across clusters

            # attention aggregation: region_token attends to lesion features
            # query: [1, 1, feature_dim], key/value: [1, M, feature_dim] or [1, k, feature_dim]
            query = region_token.unsqueeze(0)  # [1, 1, D]
            kv = cluster_lesion_features.unsqueeze(0)  # [1, M or k, D]

            attended, attn_weights = self.region_attention(query, kv, kv)  # [1, 1, D], [1, 1, M or k]
            attended_feature = attended.squeeze(0).squeeze(0)  # [D]

            # use the attention-aggregated feature directly as the region feature
            # prior fusion is added uniformly at the GNN input stage
            region_feature = attended_feature

            region_features_list.append(region_feature)

            # count lesions of each type within this cluster
            lesion_type_counts = {name: 0 for name in LESION_NAMES}
            for idx in cluster_lesion_indices:
                lesion_name = lesion_nodes[idx]['lesion_name']
                lesion_type_counts[lesion_name] += 1

            region_node = {
                'type': NODE_REGION,
                'region_idx': cluster_idx,
                'coords': region_coords,
                'absolute_coords': region_abs_coords_tuple,
                'confidence': float(np.mean(cluster_confs)),
                'num_lesions': len(cluster_lesion_indices),
                'lesion_indices': cluster_lesion_indices.tolist(),
                'lesion_type_counts': lesion_type_counts,
                'avg_radius': avg_radius,
                'density': density,
                'dist_to_od': dist_to_od,
                'dist_to_landmark': dist_to_landmark,
                'attn_weights': attn_weights.squeeze().detach().cpu().numpy()  # for visualization
            }
            region_nodes.append(region_node)

        if len(region_features_list) > 0:
            region_features = torch.stack(region_features_list, dim=0)  # [K, D]
        else:
            region_features = torch.zeros(0, self.feature_dim, device=device)

        return region_nodes, region_features
