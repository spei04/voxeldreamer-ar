"""
3D sinusoidal positional encoding for VoxelDreamer-AR.

Additive embedding (not rotary). Returned tensor is added to the token
embedding before the first transformer block. Per-axis channel allocation
mirrors `rope_3d.RoPE3DConfig`.

Whether additive sinusoidal beats rotary on 3D-structured token streams is
an open empirical question. We provide it as an ablation arm.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class Sinusoidal3DEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        axis_dims: dict[str, int] | None = None,
        axis_bases: dict[str, float] | None = None,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        if axis_dims is None:
            per = embed_dim // 8
            extra = embed_dim // 2 - per * 4
            axis_dims = {"t": per + extra, "x": per, "y": per, "z": per}
        assert sum(axis_dims.values()) == embed_dim // 2
        self.axis_dims = axis_dims
        self.axis_bases = axis_bases or {
            "t": 100000.0, "x": 10000.0, "y": 10000.0, "z": 10000.0,
        }
        self.embed_dim = embed_dim

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [T, 4] tensor of (t, x, y, z) coordinates.

        Returns:
            embeddings: [T, embed_dim], one row per token.
        """
        assert positions.ndim == 2 and positions.shape[1] == 4
        device = positions.device
        pos = positions.to(dtype=torch.float32)
        T = pos.shape[0]
        parts: list[torch.Tensor] = []
        for axis_idx, axis in enumerate(("t", "x", "y", "z")):
            d_axis = self.axis_dims[axis]
            if d_axis == 0:
                continue
            base = self.axis_bases.get(axis, 10000.0)
            channel_range = torch.arange(0, d_axis, dtype=torch.float32, device=device)
            inv_freq = 1.0 / (base ** (channel_range / d_axis))
            pos_axis = pos[:, axis_idx]
            freqs = torch.outer(pos_axis, inv_freq)  # [T, d_axis]
            parts.append(freqs.sin())
            parts.append(freqs.cos())
        return torch.cat(parts, dim=-1)  # [T, embed_dim]
