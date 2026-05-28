"""CPU tests for voxeldreamer.positional.rope_3d."""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from voxeldreamer.positional.rope_3d import (
    RoPE3DConfig,
    build_positions_from_layout,
    precompute_rope_1d_equivalent,
    precompute_rope_3d,
)


def apply_rotary_emb(x, cos, sin):
    """Verbatim copy of train.py's apply_rotary_emb so tests are self-contained."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


def test_config_default_split():
    cfg = RoPE3DConfig(head_dim=64)
    assert sum(cfg.axis_dims.values()) == 32, cfg.axis_dims


def test_config_custom_split_sums_correctly():
    cfg = RoPE3DConfig(head_dim=64, axis_dims={"t": 14, "x": 6, "y": 6, "z": 6})
    assert sum(cfg.axis_dims.values()) == 32


def test_precompute_shape():
    cfg = RoPE3DConfig(head_dim=64)
    positions = torch.tensor([[t, 0, 0, 0] for t in range(8)], dtype=torch.long)
    cos, sin = precompute_rope_3d(positions, cfg, dtype=torch.float32)
    assert cos.shape == (1, 8, 1, 32), cos.shape
    assert sin.shape == (1, 8, 1, 32), sin.shape


def test_apply_rotary_emb_drop_in():
    """Our cos/sin tables must work with the verbatim train.py apply_rotary_emb."""
    cfg = RoPE3DConfig(head_dim=32)
    T = 16
    positions = torch.tensor([[t, t % 4, (t // 4) % 4, t // 16] for t in range(T)], dtype=torch.long)
    cos, sin = precompute_rope_3d(positions, cfg, dtype=torch.float32)
    x = torch.randn(2, T, 4, 32)  # [B, T, n_head, head_dim]
    y = apply_rotary_emb(x, cos, sin)
    assert y.shape == x.shape
    # Output should differ from input (rotation is not the identity)
    assert not torch.allclose(y, x)


def test_position_zero_is_identity():
    """When all positions are zero, RoPE rotation is the identity."""
    cfg = RoPE3DConfig(head_dim=32)
    T = 4
    positions = torch.zeros(T, 4, dtype=torch.long)
    cos, sin = precompute_rope_3d(positions, cfg, dtype=torch.float32)
    x = torch.randn(1, T, 2, 32)
    y = apply_rotary_emb(x, cos, sin)
    assert torch.allclose(y, x, atol=1e-5), (y - x).abs().max()


def test_1d_equivalent_matches_train_py():
    """Sanity-check the 1D fallback matches the exact recipe in train.py."""
    seq_len = 32
    head_dim = 64
    cos, sin = precompute_rope_1d_equivalent(seq_len, head_dim, dtype=torch.float32)
    # Recompute from scratch using train.py's recipe
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32)
    inv_freq = 1.0 / (10000.0 ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    expected_cos = freqs.cos()[None, :, None, :]
    expected_sin = freqs.sin()[None, :, None, :]
    assert torch.allclose(cos, expected_cos, atol=1e-5)
    assert torch.allclose(sin, expected_sin, atol=1e-5)


def test_build_positions_from_layout():
    frame_indices = [0, 0, 0, 1, 1, 1]
    voxel_indices = [None, (1, 2, 3), (1, 2, 4), None, (5, 6, 7), (5, 6, 8)]
    pos = build_positions_from_layout(frame_indices, voxel_indices)
    assert pos.shape == (6, 4)
    # Frame indices flow through
    assert pos[:, 0].tolist() == frame_indices
    # Non-voxel rows use sentinel
    assert pos[0].tolist() == [0, -1, -1, -1]
    assert pos[3].tolist() == [1, -1, -1, -1]
    # Voxel rows preserve (x,y,z)
    assert pos[1].tolist() == [0, 1, 2, 3]
    assert pos[5].tolist() == [1, 5, 6, 8]


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} rope_3d tests passed.")


if __name__ == "__main__":
    run_all()
