"""
PARAGraph constants and pure static functions.

Contains node types, edge-type mapping, OD diameter computation, coordinate
computation, and other self-free utility functions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Dict, Tuple


# ============ Node type constants ============

NODE_GRAPH    = 0   # Graph node (anchored at OD; graph-level aggregator)
NODE_GLOBAL   = 1   # Global node (anchored at Fovea; global spatial anchor)
NODE_MA       = 2
NODE_HE       = 3
NODE_EX       = 4
NODE_SE       = 5
# Category node types (one per lesion class)
NODE_MA_CAT   = 6
NODE_HE_CAT   = 7
NODE_EX_CAT   = 8
NODE_SE_CAT   = 9
# Region node type (lesion-enriched region)
NODE_REGION   = 10

# Backward-compat aliases
NODE_OD       = NODE_GRAPH
NODE_LANDMARK = NODE_GLOBAL
NODE_MA_TOKEN = NODE_MA_CAT
NODE_HE_TOKEN = NODE_HE_CAT
NODE_EX_TOKEN = NODE_EX_CAT
NODE_SE_TOKEN = NODE_SE_CAT

# Lesion type names
LESION_NAMES        = ['MA', 'HE', 'EX', 'SE']
LESION_NODE_TYPES   = [NODE_MA,     NODE_HE,     NODE_EX,     NODE_SE    ]
CATEGORY_NODE_TYPES = [NODE_MA_CAT, NODE_HE_CAT, NODE_EX_CAT, NODE_SE_CAT]
TOKEN_NODE_TYPES    = CATEGORY_NODE_TYPES   # backward-compat alias

# Number of edge types (used for edge-type embedding)
NUM_EDGE_TYPES = 22


# ============ Pure static functions ============

def get_edge_type_id(edge_type_str: str) -> int:
    """
    Map an edge-type string to an integer ID (22 types total).

    0: global_graph              1: graph_global
    2: graph_to_*_cat         3: *_cat_to_graph
    4: global_to_*_cat           5: *_cat_to_global
    6: *_cat_to_lesion        7: lesion_to_*_cat
    8: graph_to_lesion_direct  9: lesion_to_graph_direct
    10: global_to_lesion_direct  11: lesion_to_global_direct
    12: intra_class (reserved)   13: inter_class (reserved)
    14: graph_to_region        15: region_to_graph
    16: global_to_region          17: region_to_global
    18: region_to_lesion       19: lesion_to_region
    20: region_to_*_cat        21: *_cat_to_region
    """
    if edge_type_str in ('global_graph', 'landmark_od'):
        return 0
    elif edge_type_str in ('graph_global', 'od_landmark'):
        return 1
    elif edge_type_str.startswith('graph_to_') and edge_type_str.endswith('_cat'):
        return 2
    elif edge_type_str.startswith('od_to_') and edge_type_str.endswith('_token'):
        return 2  # legacy
    elif edge_type_str.endswith('_cat_to_graph'):
        return 3
    elif edge_type_str.endswith('_token_to_od'):
        return 3  # legacy
    elif edge_type_str.startswith('global_to_') and edge_type_str.endswith('_cat'):
        return 4
    elif edge_type_str.startswith('landmark_to_') and edge_type_str.endswith('_token'):
        return 4  # legacy
    elif edge_type_str.endswith('_cat_to_global'):
        return 5
    elif edge_type_str.endswith('_token_to_landmark'):
        return 5  # legacy
    elif edge_type_str.endswith('_cat_to_lesion') or edge_type_str.endswith('_token_to_lesion'):
        return 6
    elif edge_type_str.startswith('lesion_to_') and (edge_type_str.endswith('_cat') or edge_type_str.endswith('_token')):
        return 7
    elif edge_type_str in ('graph_to_lesion_direct', 'od_to_lesion_direct'):
        return 8
    elif edge_type_str in ('lesion_to_graph_direct', 'lesion_to_od_direct'):
        return 9
    elif edge_type_str in ('global_to_lesion_direct', 'landmark_to_lesion_direct'):
        return 10
    elif edge_type_str in ('lesion_to_global_direct', 'lesion_to_landmark_direct'):
        return 11
    elif edge_type_str.startswith('intra_'):  # reserved
        return 12
    elif edge_type_str.startswith('inter_'):  # reserved
        return 13
    elif edge_type_str in ('graph_to_region', 'od_to_region'):
        return 14
    elif edge_type_str in ('region_to_graph', 'region_to_od'):
        return 15
    elif edge_type_str in ('global_to_region', 'landmark_to_region'):
        return 16
    elif edge_type_str in ('region_to_global', 'region_to_landmark'):
        return 17
    elif edge_type_str == 'region_to_lesion':
        return 18
    elif edge_type_str == 'lesion_to_region':
        return 19
    elif edge_type_str.startswith('region_to_') and (edge_type_str.endswith('_cat') or edge_type_str.endswith('_token')):
        return 20
    elif edge_type_str.endswith('_cat_to_region') or edge_type_str.endswith('_token_to_region'):
        return 21
    else:
        return 0  # fallback


def compute_od_diameter(od_mask: np.ndarray) -> float:
    """
    Compute the equivalent-circle diameter of the OD.

    Args:
        od_mask: [H, W] binary OD mask (0 or 255)

    Returns:
        diameter: OD diameter in pixels, 0 on failure
    """
    contours, _ = cv2.findContours(
        od_mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if len(contours) == 0:
        return 0.0

    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)

    # equivalent-circle diameter: d = 2 * sqrt(area / pi)
    diameter = 2 * np.sqrt(area / np.pi)

    return diameter


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute weighted median.

    Args:
        values: [N] array of values
        weights: [N] array of weights (must be positive)

    Returns:
        Weighted median value
    """
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    # Sort by values
    sorted_indices = np.argsort(values)
    sorted_values = values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    # Cumulative weights
    cumsum = np.cumsum(sorted_weights)
    total = cumsum[-1]

    # Find median: first index where cumsum >= total / 2
    median_idx = np.searchsorted(cumsum, total / 2.0)
    median_idx = min(median_idx, len(sorted_values) - 1)

    return float(sorted_values[median_idx])


def compute_landmark_with_prior(
    landmark_logits: torch.Tensor,
    od_center: Tuple[float, float],
    od_diameter: float,
    mu: float = 2.7,
    sigma: float = 0.4,
    eps: float = 1e-6,
    differentiable: bool = True
) -> Dict:
    """
    Compute landmark (fovea) coordinates using a medical prior.

    NOTE: utility function, currently NOT used in the forward pass (the fovea is
    obtained via a plain soft-argmax over the landmark heatmap). Kept as an
    available alternative.

    Prior: the landmark typically lies on an annulus about 2.5-3.0 OD diameters
    away from the OD.

    Args:
        landmark_logits: [1, H, W] landmark logits from the network
        od_center: OD center coordinates (x0, y0)
        od_diameter: OD diameter (pixels)
        mu: annulus center distance multiplier (default 2.7)
        sigma: annulus width multiplier (default 0.4)
        eps: numerical stability
        differentiable: whether to use differentiable soft argmax (default True)

    Returns:
        dict: {
            'coords': (x, y) or (x_tensor, y_tensor) (tensors when differentiable=True)
            'coords_tensor': (x_tensor, y_tensor)  # always tensor coords for gradients
            'confidence_peak': float,  # posterior peak
            'confidence_entropy': float,  # posterior entropy (lower = more certain)
            'posterior': Tensor  # [H, W] posterior distribution
        }
    """
    _, H, W = landmark_logits.shape
    device = landmark_logits.device

    x0, y0 = od_center

    # build coordinate grid
    y_coords = torch.arange(H, device=device, dtype=torch.float32)
    x_coords = torch.arange(W, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

    # distance from each pixel to the OD center
    r = torch.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    # medical prior: annular distribution
    # P_ring(x,y) = exp(-(r - μD)² / (2(σD)²))
    mu_d = mu * od_diameter
    sigma_d = sigma * od_diameter
    sigma_d = max(sigma_d, 1e-6)
    p_ring = torch.exp(-((r - mu_d) ** 2) / (2.0 * (sigma_d ** 2)))

    # Posterior = softmax(S + log(P_ring + ε))
    log_prior = torch.log(p_ring + eps)
    logits_with_prior = landmark_logits[0] + log_prior  # [H, W]

    # Softmax over entire spatial dimension
    logits_flat = logits_with_prior.flatten()  # [H*W]
    posterior_flat = torch.softmax(logits_flat, dim=0)
    posterior = posterior_flat.view(H, W)

    # Soft argmax (spatial expectation): differentiable coordinate extraction
    x_soft = (xx * posterior).sum()  # scalar tensor
    y_soft = (yy * posterior).sum()  # scalar tensor

    # confidence 1: posterior peak
    confidence_peak = posterior_flat.max().detach().item()

    # confidence 2: posterior entropy H = -sum(p * log(p))
    log_posterior = torch.log(posterior_flat + eps)
    entropy = -(posterior_flat * log_posterior).sum().detach().item()

    if differentiable:
        return {
            'coords': (x_soft, y_soft),
            'coords_tensor': (x_soft, y_soft),
            'confidence_peak': confidence_peak,
            'confidence_entropy': entropy,
            'posterior': posterior
        }
    else:
        return {
            'coords': (x_soft.detach().item(), y_soft.detach().item()),
            'coords_tensor': (x_soft, y_soft),
            'confidence_peak': confidence_peak,
            'confidence_entropy': entropy,
            'posterior': posterior
        }


def get_dr_grade_name(grade: int, num_classes: int = 5) -> str:
    """
    Get the DR grade name.

    Args:
        grade: grade (0-4 for 5-class, 0-2 for 3-class)
        num_classes: number of classes

    Returns:
        name: grade name
    """
    if num_classes == 5:
        names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
    elif num_classes == 3:
        names = ['No DR', 'Non-Proliferative DR', 'Proliferative DR']
    else:
        return f'Grade {grade}'

    if 0 <= grade < len(names):
        return names[grade]
    return f'Unknown Grade {grade}'
