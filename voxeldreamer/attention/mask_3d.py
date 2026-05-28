"""
3D-aware attention masks for VoxelDreamer-AR.

Two structural priors to ablate against full causal attention:

1. **3D-local window**: a token attends only to past tokens whose voxel
   coordinates are within an L_inf or L_2 radius. Forces locality in 3D
   rather than in serialization order.

2. **Axial / strided**: a token attends only to past tokens that share at
   least one of (x, y, z, t) coordinates with it (Qwen2-VL / VideoRoPE
   "axial attention" trick at much lower compute).

Both masks are causal (no future) by construction.

The masks produced are additive bias tensors of shape [T, T] suitable for
adding to attention scores before softmax. -inf positions are masked out.

Note: train.py uses Flash-Attention 3 with a window_size parameter, not
explicit masks. We provide explicit masks here for ablations + interpretable
debugging. Integrating into the FA3 path would require a custom kernel or
falling back to standard scaled-dot-product attention for the masked layers.
"""

from __future__ import annotations

import torch


def causal_3d_local_mask(
    positions: torch.Tensor,
    radius: float = 8.0,
    metric: str = "linf",
) -> torch.Tensor:
    """
    Args:
        positions: [T, 4] tensor of (t, x, y, z) coordinates per token.
                   Non-voxel tokens (action / camera) should use a sentinel
                   like (-1, -1, -1) for (x,y,z) so they attend broadly.
        radius: spatial radius for "local" inclusion. t-axis is not masked
                here — temporal attention is governed by the causal mask
                + window_size in train.py.
        metric: "linf" or "l2".

    Returns:
        mask: [T, T] tensor with -inf where attention is disallowed and
              0 elsewhere. To be *added* to attention scores before softmax.
    """
    assert positions.ndim == 2 and positions.shape[1] == 4
    T = positions.shape[0]
    device = positions.device
    pos_xyz = positions[:, 1:].to(dtype=torch.float32)  # [T, 3]

    # Pairwise distance over (x,y,z)
    diff = pos_xyz[:, None, :] - pos_xyz[None, :, :]  # [T, T, 3]
    if metric == "linf":
        dist = diff.abs().amax(dim=-1)
    elif metric == "l2":
        dist = diff.norm(dim=-1)
    else:
        raise ValueError(f"unknown metric: {metric}")

    in_window = dist <= radius  # [T, T] bool

    # Causal: row i attends to column j only if j <= i.
    idx = torch.arange(T, device=device)
    causal = idx[None, :] <= idx[:, None]  # [T, T]

    # Non-voxel tokens (sentinel -1) always allowed
    is_voxel_row = (positions[:, 1:] >= 0).all(dim=-1)
    is_voxel_col = (positions[:, 1:] >= 0).all(dim=-1)
    voxel_pair = is_voxel_row[:, None] & is_voxel_col[None, :]
    # If either side is a non-voxel token, allow regardless of window
    allowed = (~voxel_pair) | in_window
    allowed = allowed & causal

    mask = torch.zeros((T, T), dtype=torch.float32, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask


def causal_axial_mask(positions: torch.Tensor) -> torch.Tensor:
    """
    Axial mask: token i attends to token j (with j <= i) iff they share at
    least one of the four axes (t, x, y, z). Dramatically sparser than full
    attention while preserving 3D structural communication.

    Useful as a Tier-2 sweep arm — very cheap to integrate and a strong
    inductive bias when 3D structure is real.
    """
    assert positions.ndim == 2 and positions.shape[1] == 4
    T = positions.shape[0]
    device = positions.device
    pos = positions.to(dtype=torch.long)

    shared = torch.zeros((T, T), dtype=torch.bool, device=device)
    for ax in range(4):
        ax_pos = pos[:, ax]
        shared |= ax_pos[:, None] == ax_pos[None, :]

    idx = torch.arange(T, device=device)
    causal = idx[None, :] <= idx[:, None]
    allowed = shared & causal

    mask = torch.zeros((T, T), dtype=torch.float32, device=device)
    mask.masked_fill_(~allowed, float("-inf"))
    return mask
