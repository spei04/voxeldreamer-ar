"""CPU tests for voxeldreamer.attention.mask_3d."""

import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from voxeldreamer.attention.mask_3d import causal_3d_local_mask, causal_axial_mask


def test_local_mask_is_causal():
    positions = torch.tensor([[t, 0, 0, 0] for t in range(8)], dtype=torch.long)
    mask = causal_3d_local_mask(positions, radius=100.0)  # huge radius -> only causality bites
    # Upper triangle (j > i) should be -inf
    T = positions.shape[0]
    for i in range(T):
        for j in range(i + 1, T):
            assert mask[i, j].item() == float("-inf"), (i, j, mask[i, j])
    # Lower-triangular + diagonal should be 0
    for i in range(T):
        for j in range(i + 1):
            assert mask[i, j].item() == 0.0, (i, j, mask[i, j])


def test_local_mask_blocks_distant_voxels():
    # Two voxel tokens at very different spatial positions.
    positions = torch.tensor([[0, 0, 0, 0], [0, 100, 100, 100]], dtype=torch.long)
    mask = causal_3d_local_mask(positions, radius=8.0, metric="linf")
    # row=1 (the distant one), col=0 (the origin): causal allows but distance > radius
    assert mask[1, 0].item() == float("-inf")


def test_local_mask_allows_nearby_voxels():
    positions = torch.tensor([[0, 0, 0, 0], [0, 1, 1, 1]], dtype=torch.long)
    mask = causal_3d_local_mask(positions, radius=8.0, metric="linf")
    assert mask[1, 0].item() == 0.0


def test_local_mask_lets_non_voxel_through():
    # Non-voxel tokens (sentinel -1) should not be filtered by spatial window.
    positions = torch.tensor([[0, -1, -1, -1], [0, 100, 100, 100]], dtype=torch.long)
    mask = causal_3d_local_mask(positions, radius=1.0)
    # The voxel can attend to the action token at row=1, col=0
    assert mask[1, 0].item() == 0.0


def test_axial_mask_self_allowed():
    positions = torch.tensor([[0, 1, 2, 3], [0, 4, 5, 6]], dtype=torch.long)
    mask = causal_axial_mask(positions)
    # Diagonal always allowed (every token shares all axes with itself)
    assert mask[0, 0].item() == 0.0
    assert mask[1, 1].item() == 0.0


def test_axial_mask_shared_axis_allowed():
    # Two tokens that share t=0 but have different (x,y,z): should be allowed.
    positions = torch.tensor([[0, 1, 2, 3], [0, 9, 9, 9]], dtype=torch.long)
    mask = causal_axial_mask(positions)
    assert mask[1, 0].item() == 0.0


def test_axial_mask_no_shared_blocked():
    positions = torch.tensor([[0, 1, 2, 3], [1, 9, 9, 9]], dtype=torch.long)
    mask = causal_axial_mask(positions)
    # row=1, col=0: nothing shared, causal allows i=1>j=0, but axial forbids
    assert mask[1, 0].item() == float("-inf")


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} mask tests passed.")


if __name__ == "__main__":
    run_all()
