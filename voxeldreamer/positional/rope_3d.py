"""
3D / 4D Rotary Positional Embeddings for VoxelDreamer-AR.

Drop-in replacement for autoresearch train.py's `_precompute_rotary_embeddings`.
Produces (cos, sin) tables of shape [1, T, 1, head_dim // 2] so the existing
`apply_rotary_emb(x, cos, sin)` function works unchanged.

The key insight: train.py's RoPE splits head_dim into two halves and rotates
between them, using a single cos/sin per position. We keep that machinery
intact — we just *internally* compose the cos/sin from multiple axes.

Axes: (t, x, y, z). t = frame index, (x,y,z) = voxel position. Channels are
partitioned across axes (defaulting to lower frequencies on t per Qwen2-VL /
VideoRoPE guidance — long temporal extents must not periodically alias).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch


@dataclass
class RoPE3DConfig:
    head_dim: int
    # Channels-per-axis (must sum to head_dim // 2). Default biases toward t/z.
    axis_dims: dict[str, int] = field(default_factory=dict)
    # Per-axis base period. Larger base = lower frequency = longer effective range.
    # Default: temporal gets a much larger base so periodic distractors don't fire.
    axis_bases: dict[str, float] = field(default_factory=lambda: {
        "t": 100000.0, "x": 10000.0, "y": 10000.0, "z": 10000.0,
    })

    def __post_init__(self):
        assert self.head_dim % 2 == 0, "head_dim must be even"
        half = self.head_dim // 2
        if not self.axis_dims:
            # Default split: prioritize t and z (height in Minecraft) slightly.
            per = half // 4
            extra = half - per * 4
            self.axis_dims = {"t": per + extra, "x": per, "y": per, "z": per}
        assert sum(self.axis_dims.values()) == half, (
            f"axis_dims must sum to head_dim // 2 ({half}), got {sum(self.axis_dims.values())}"
        )
        assert all(d % 2 == 0 for d in self.axis_dims.values()), (
            "each axis_dim must be even (rotary pairs are channel/channel+d_axis)"
        )


def precompute_rope_3d(
    positions: torch.Tensor,
    config: RoPE3DConfig,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        positions: int/float tensor of shape [T, 4] with columns (t, x, y, z).
        config: RoPE3DConfig specifying head_dim, axis_dims, axis_bases.
        device: target device; defaults to positions.device.
        dtype: output dtype (typically torch.bfloat16 to match train.py).

    Returns:
        (cos, sin), each of shape [1, T, 1, head_dim // 2]. Drop-in for the
        existing `apply_rotary_emb(x, cos, sin)` in autoresearch train.py.
    """
    assert positions.ndim == 2 and positions.shape[1] == 4, (
        f"positions must be [T, 4] (t,x,y,z), got {tuple(positions.shape)}"
    )
    device = device or positions.device
    positions = positions.to(device=device, dtype=torch.float32)
    T = positions.shape[0]

    cos_parts: list[torch.Tensor] = []
    sin_parts: list[torch.Tensor] = []
    for axis_idx, axis in enumerate(("t", "x", "y", "z")):
        d_axis = config.axis_dims[axis]
        if d_axis == 0:
            continue
        base = config.axis_bases.get(axis, 10000.0)
        # Channel range maps each of d_axis channels to a frequency.
        channel_range = torch.arange(0, d_axis, dtype=torch.float32, device=device)
        # Standard RoPE: inv_freq[i] = 1 / base^(2i / d). We use d = 2 * d_axis
        # so that channel positions match how head_dim is split into pairs.
        inv_freq = 1.0 / (base ** (channel_range / d_axis))
        pos_axis = positions[:, axis_idx]  # [T]
        freqs = torch.outer(pos_axis, inv_freq)  # [T, d_axis]
        cos_parts.append(freqs.cos())
        sin_parts.append(freqs.sin())

    cos = torch.cat(cos_parts, dim=-1)  # [T, head_dim // 2]
    sin = torch.cat(sin_parts, dim=-1)
    cos = cos.to(dtype)[None, :, None, :]  # [1, T, 1, head_dim // 2]
    sin = sin.to(dtype)[None, :, None, :]
    return cos, sin


def precompute_rope_1d_equivalent(
    seq_len: int,
    head_dim: int,
    base: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Plain 1D RoPE matching train.py's behaviour bit-for-bit. Used as the
    baseline arm for ablations and to sanity-check the 3D path collapses
    correctly when only the t-axis carries nonzero positions.
    """
    device = device or torch.device("cpu")
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)[None, :, None, :]
    sin = freqs.sin().to(dtype)[None, :, None, :]
    return cos, sin


def build_positions_from_layout(
    frame_indices: Sequence[int],
    voxel_indices: Sequence[tuple[int, int, int] | None],
    action_marker: tuple[int, int, int] = (-1, -1, -1),
) -> torch.Tensor:
    """
    Construct a [T, 4] (t, x, y, z) positions tensor for a mixed sequence of
    voxel tokens and action/camera tokens.

    voxel_indices[i] = None means "this token isn't a voxel" (action / camera /
    separator); we use a sentinel `action_marker` so the position is constant
    across all non-voxel tokens within a frame (they share that frame's t).

    Returns:
        positions: long tensor of shape [T, 4].
    """
    assert len(frame_indices) == len(voxel_indices)
    out = []
    for t, v in zip(frame_indices, voxel_indices):
        if v is None:
            out.append((t,) + action_marker)
        else:
            out.append((t,) + tuple(v))
    return torch.tensor(out, dtype=torch.long)
