"""CPU tests for voxeldreamer.tokenizer.orderings."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from voxeldreamer.tokenizer.orderings import (
    camera_frustum_order,
    hilbert_order,
    morton_order,
    raster_order,
    voxel_major_zyx,
)


def assert_is_permutation(perm: torch.Tensor, n: int):
    assert perm.shape == (n,), perm.shape
    sorted_perm, _ = perm.sort()
    assert torch.equal(sorted_perm, torch.arange(n, dtype=torch.long)), (
        f"not a permutation of [0, {n}): {perm.tolist()}"
    )


def test_raster_is_identity():
    perm = raster_order(3, 4, 5)
    assert_is_permutation(perm, 60)
    assert torch.equal(perm, torch.arange(60))


def test_voxel_major_zyx_is_permutation():
    perm = voxel_major_zyx(3, 4, 5)
    assert_is_permutation(perm, 60)


def test_voxel_major_first_column():
    """In voxel_major_zyx the first Z entries should be the column at (x=0, y=0)."""
    X, Y, Z = 2, 3, 4
    perm = voxel_major_zyx(X, Y, Z)
    # raster index for (x=0, y=0, z) is (0*Y + 0)*Z + z = z
    first_column = perm[:Z]
    assert first_column.tolist() == list(range(Z))


def test_morton_is_permutation_cube():
    perm = morton_order(4, 4, 4)
    assert_is_permutation(perm, 64)


def test_morton_is_permutation_non_cube():
    perm = morton_order(3, 5, 7)
    assert_is_permutation(perm, 3 * 5 * 7)


def test_hilbert_is_permutation_powers_of_two():
    for N in (2, 4, 8):
        perm = hilbert_order(N)
        assert_is_permutation(perm, N ** 3)


def test_camera_frustum_first_voxel_is_closest():
    perm = camera_frustum_order(4, 4, 4, camera_xyz=(0.0, 0.0, 0.0))
    # The voxel at (0,0,0) has raster idx 0; it should be first.
    assert perm[0].item() == 0


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} ordering tests passed.")


if __name__ == "__main__":
    run_all()
