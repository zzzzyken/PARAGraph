"""
PARAGraph GNN Mixin

GNN message passing, graph-level representation, prediction and loss computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING

from ._paragraph_constants import get_edge_type_id

if TYPE_CHECKING:
    from .paragraph import PARAGraph


class PARAGraphGNNMixin:
    """Mixin: GNN input preparation, graph-level representation, prediction and loss."""

    def get_node_features_for_gnn(self, graph: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build node features and edge indices for the GNN (torch tensors).

        Node features = encoder features + medical prior fusion
        (coords + lesion type + area). Includes: OD, Landmark, Lesion, Category,
        and Region nodes.

        Args:
            graph: a single graph returned by forward

        Returns:
            node_features: [N, feature_dim] all node features (with prior fusion)
            edge_index: [2, E] edge indices in COO format
        """
        # GNN input stage: get device from saved _device or from node features
        device = graph.get('_device', graph['nodes'][0]['feature'].device)
        nodes = graph['nodes']
        lesion_nodes = graph.get('lesion_nodes', [])
        lesion_features = graph.get('lesion_features', None)
        category_nodes = graph.get('category_nodes', [])
        region_nodes = graph.get('region_nodes', [])
        region_features = graph.get('region_features', None)

        num_nodes = len(nodes) + len(lesion_nodes) + len(category_nodes) + len(region_nodes)

        # init node features and prior features
        node_features = torch.zeros(num_nodes, self.feature_dim, device=device)
        # prior features: pos_only=2D, full=7D
        node_prior = torch.zeros(num_nodes, self.pos_extra_dim, device=device)

        # OD and Landmark nodes (coords only, other prior features zero)
        for i, node in enumerate(nodes):
            if 'feature' in node:
                node_features[i] = node['feature']
            coords = node.get('coords', (0.0, 0.0))
            node_prior[i, 0] = coords[0]
            node_prior[i, 1] = coords[1]

        # Lesion nodes
        if lesion_features is not None and len(lesion_features) > 0:
            for lesion_node in lesion_nodes:
                feat_idx = lesion_node['feature_idx']
                node_idx = lesion_node['node_idx']
                node_features[node_idx] = lesion_features[feat_idx]
                cx, cy = lesion_node.get('coords', (0.0, 0.0))
                if self.fusion_mode == 'pos_only':
                    prior = [cx, cy]
                else:
                    # [cx, cy, onehot(4), area_n] = 7D
                    lesion_class = lesion_node.get('lesion_class', 0)
                    area_n = lesion_node.get('area_normalized', 0.0)
                    prior = [cx, cy, 0.0, 0.0, 0.0, 0.0, area_n]
                    prior[2 + lesion_class] = 1.0  # type one-hot
                node_prior[node_idx] = torch.tensor(prior, dtype=torch.float32, device=device)

        # Category (class-token) nodes (no prior fusion, all zero)
        category_node_idx_set = set()
        for category_node in category_nodes:
            node_idx = category_node['node_idx']
            if 'feature' in category_node:
                node_features[node_idx] = category_node['feature']
            category_node_idx_set.add(node_idx)

        # Region nodes (coords only, other prior features zero)
        if region_features is not None and len(region_features) > 0:
            for region_node in region_nodes:
                feat_idx = region_node['feature_idx']
                node_idx = region_node['node_idx']
                node_features[node_idx] = region_features[feat_idx]
                coords = region_node.get('coords', (0.0, 0.0))
                node_prior[node_idx, 0] = coords[0]
                node_prior[node_idx, 1] = coords[1]

        # Medical prior fusion: concat extra features (coords + lesion type + area),
        # project back to feature_dim. Applied to OD/Landmark/Lesion/Region nodes,
        # but not to Category (class-token) nodes.
        # Ablation: skip prior fusion when use_prior_fusion=False
        if self.use_prior_fusion:
            # for Category nodes, zero out the prior features (no prior info)
            if len(category_node_idx_set) > 0:
                for idx in category_node_idx_set:
                    node_prior[idx] = 0.0
            # concat features and prior: [N, feature_dim + pos_extra_dim]
            node_features_with_prior = torch.cat([node_features, node_prior], dim=-1)
            # project back to feature_dim: [N, feature_dim]
            node_features = self.pos_proj(node_features_with_prior)

        # build edge index
        # ablation: exclude OD and Landmark when use_optic_nodes=False
        od_node_idx = 0
        landmark_node_idx = 1  # nodes = [od_node, landmark_node]
        if not self.use_optic_nodes:
            excluded_nodes = {od_node_idx, landmark_node_idx}  # exclude OD and Global
        elif not self.use_global_node:
            excluded_nodes = {landmark_node_idx}  # exclude only the Global node
        else:
            excluded_nodes = set()  # exclude nothing

        edges = graph['edges'] + graph.get('lesion_edges', [])
        if excluded_nodes:
            edges = [e for e in edges if e['from'] not in excluded_nodes and e['to'] not in excluded_nodes]

        if len(edges) > 0:
            edge_index = torch.tensor(
                [[e['from'], e['to']] for e in edges],
                dtype=torch.long,
                device=device
            ).t()  # [2, E]
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)

        return node_features, edge_index

    def get_edge_weights_for_gnn(self, graph: Dict) -> torch.Tensor:
        """
        Build edge weights for the GNN: edge_gate = gate_from * gate_to.

        Each node has its own gate g_i = σ(α·c_i + β); the edge weight is the
        product of both endpoints' gates, reflecting their reliability.

        Args:
            graph: a single graph returned by forward

        Returns:
            edge_weights: [E, 1] edge weights (gate product, broadcast to num_heads)
        """
        device = graph.get('_device', graph['nodes'][0]['feature'].device)
        # nodes to exclude (consistent with get_node_features_for_gnn)
        od_node_idx = 0
        landmark_node_idx = 1
        if not self.use_optic_nodes:
            excluded_nodes = {od_node_idx, landmark_node_idx}
        elif not self.use_global_node:
            excluded_nodes = {landmark_node_idx}
        else:
            excluded_nodes = set()

        edges = graph['edges'] + graph.get('lesion_edges', [])
        if excluded_nodes:
            edges = [e for e in edges if e['from'] not in excluded_nodes and e['to'] not in excluded_nodes]

        if len(edges) == 0:
            return torch.zeros(0, 1, device=device)

        # node_idx → gate map
        node_gates = graph['node_gates']
        default_gate = torch.ones((), device=device)

        # edge weight = gate_from * gate_to
        gate_list = []
        for e in edges:
            g_from = node_gates.get(e['from'], default_gate)
            g_to = node_gates.get(e['to'], default_gate)
            if not torch.is_tensor(g_from):
                g_from = torch.tensor(g_from, device=device, dtype=torch.float32)
            if not torch.is_tensor(g_to):
                g_to = torch.tensor(g_to, device=device, dtype=torch.float32)
            gate_list.append(g_from * g_to)
        gate_tensor = torch.stack(gate_list).float().unsqueeze(-1)  # [E, 1]

        return gate_tensor

    def get_edge_type_bias_for_gnn(self, graph: Dict) -> torch.Tensor:
        """
        Build per-edge-type attention bias for the GNN (added to attention logits
        before softmax).

        Returns:
            edge_type_bias: [E, num_heads]
        """
        device = graph.get('_device', graph['nodes'][0]['feature'].device)
        # nodes to exclude (consistent with get_node_features_for_gnn)
        od_node_idx = 0
        landmark_node_idx = 1
        if not self.use_optic_nodes:
            excluded_nodes = {od_node_idx, landmark_node_idx}
        elif not self.use_global_node:
            excluded_nodes = {landmark_node_idx}
        else:
            excluded_nodes = set()

        edges = graph['edges'] + graph.get('lesion_edges', [])
        if excluded_nodes:
            edges = [e for e in edges if e['from'] not in excluded_nodes and e['to'] not in excluded_nodes]

        if len(edges) == 0:
            return torch.zeros(0, self.gat_num_heads, device=device)

        type_ids = torch.tensor(
            [get_edge_type_id(e.get('edge_type', '')) for e in edges],
            dtype=torch.long, device=device
        )
        return self.edge_type_attn_bias(type_ids)  # [E, num_heads]

    def predict(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Predict DR grade.

        Args:
            x: [B, 3, H, W] input image
            **kwargs: other forward arguments

        Returns:
            Dict:
                - 'predictions': [B] predicted classes
                - 'probs': [B, num_classes] class probabilities
                - 'logits': [B, num_classes] class logits
        """
        output = self.forward(x, **kwargs)
        predictions = output['logits'].argmax(dim=-1)

        return {
            'predictions': predictions,
            'probs': output['probs'],
            'logits': output['logits']
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the DR classification loss.

        Args:
            x: [B, 3, H, W] input image
            labels: [B] ground-truth labels
            class_weights: [num_classes] class weights (optional, for class imbalance)
            **kwargs: other forward arguments

        Returns:
            Dict:
                - 'loss': classification loss
                - 'logits': [B, num_classes] class logits
                - 'probs': [B, num_classes] class probabilities
                - 'predictions': [B] predicted classes
                - 'accuracy': accuracy
        """
        output = self.forward(x, **kwargs)
        logits = output['logits']

        # cross-entropy loss
        if class_weights is not None:
            loss = F.cross_entropy(logits, labels, weight=class_weights)
        else:
            loss = F.cross_entropy(logits, labels)

        # prediction
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == labels).float().mean()

        return {
            'loss': loss,
            'logits': logits,
            'probs': output['probs'],
            'predictions': predictions,
            'accuracy': accuracy,
            'graphs': output['graphs']
        }
