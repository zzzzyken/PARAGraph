"""
Graph Attention Layer (GAT) for PARAGraph

Multi-head graph attention message passing, with support for edge weights
and per-edge-type attention bias.
"""

import torch
import torch.nn as nn
from typing import Optional


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).

    Implements graph attention message passing with edge weights.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        concat: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads
        self.concat = concat

        # Linear projection
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        # Attention parameters
        self.a_src = nn.Parameter(torch.randn(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.randn(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.view(num_heads, -1))
        nn.init.xavier_uniform_(self.a_dst.view(num_heads, -1))

        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        if concat:
            self.out_proj = nn.Linear(out_dim, out_dim)
        else:
            self.out_proj = nn.Linear(self.head_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        edge_type_bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [N, in_dim] node features
            edge_index: [2, E] edge indices in COO format
            edge_weight: [E] or [E, H] edge weights (optional, multiplied after softmax)
            edge_type_bias: [E, H] per-edge-type attention bias (optional, added before softmax)

        Returns:
            out: [N, out_dim] updated node features
        """
        N = x.size(0)
        device = x.device

        # Linear projection: [N, out_dim]
        h = self.W(x)
        # Reshape to multi-head: [N, num_heads, head_dim]
        h = h.view(N, self.num_heads, self.head_dim)

        # Attention scores: source and destination contributions
        attn_src = (h * self.a_src).sum(dim=-1)  # [N, num_heads]
        attn_dst = (h * self.a_dst).sum(dim=-1)  # [N, num_heads]

        if edge_index.size(1) == 0:
            # No edges: return projected features
            out = h.view(N, -1) if self.concat else h.mean(dim=1)
            return self.out_proj(out)

        src, dst = edge_index[0], edge_index[1]

        # Per-edge attention: e_ij = LeakyReLU(a_src[i] + a_dst[j] + bias[edge_type])
        edge_attn = attn_src[src] + attn_dst[dst]  # [E, num_heads]
        if edge_type_bias is not None:
            edge_attn = edge_attn + edge_type_bias  # [E, num_heads]
        edge_attn = self.leaky_relu(edge_attn)

        # Validate edge weight shape: supports [E] or [E, num_heads]
        if edge_weight is not None:
            assert edge_weight.dim() in (1, 2), f"edge_weight must be [E] or [E, num_heads], got {edge_weight.shape}"

        # Softmax normalization (per destination node), implemented with scatter
        # edge_attn: [E, H]
        max_per_dst = torch.full((N, self.num_heads), -1e9, device=device)
        max_per_dst.scatter_reduce_(
            0,
            dst.unsqueeze(-1).expand(-1, self.num_heads),
            edge_attn,
            reduce="amax",
            include_self=True
        )

        edge_attn = edge_attn - max_per_dst[dst]
        edge_attn_exp = torch.exp(edge_attn)

        den = torch.zeros(N, self.num_heads, device=device)
        den.scatter_add_(0, dst.unsqueeze(-1).expand(-1, self.num_heads), edge_attn_exp)

        attn_norm = edge_attn_exp / (den[dst] + 1e-8)

        # Apply edge weights (edge_type_weight * gate): multiplied onto normalized attention
        if edge_weight is not None:
            if edge_weight.dim() == 1:
                attn_norm = attn_norm * edge_weight.unsqueeze(-1)  # [E, 1] broadcast
            else:
                attn_norm = attn_norm * edge_weight  # [E, num_heads]

        # Message passing: aggregate neighbor features
        msg = h[src] * attn_norm.unsqueeze(-1)  # [E, num_heads, head_dim]

        # Aggregate to destination nodes
        out = torch.zeros(N, self.num_heads, self.head_dim, device=device)
        dst_expanded = dst.view(-1, 1, 1).expand(-1, self.num_heads, self.head_dim)
        out.scatter_add_(0, dst_expanded, msg)

        # Merge heads
        if self.concat:
            out = out.view(N, -1)  # [N, out_dim]
        else:
            out = out.mean(dim=1)  # [N, head_dim]

        return self.out_proj(out)
