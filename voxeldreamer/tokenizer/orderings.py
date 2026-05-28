"""
Locality-preserving orderings of 3D voxel grids into 1D token sequences.

For VoxelDreamer-AR, the order in which voxels appear in the token stream
sets what "previous tokens" the AR model conditions on. Different orderings
yield very different inductive biases.

Implemented orderings:
- raster: x fastest, then y, then z. Simple baseline. Bad locality across y/z.
- voxel_major_zyx: z fastest, then y, then x. Column-of-voxels per (x,y).
- morton: Z-order bit interleaving. Sub-cube clustering, fast to compute.
- hilbert: 3D Hilbert curve. Best locality preservation but more expensive.
- camera_frustum: nearest-to-camera first (requires a camera pose).

All functions return a permutation tensor `perm` of length X*Y*Z such that
`flat[perm[i]]` gives the i-th token's voxel value when the original grid
was flattened in raster order.
"""

from __future__ import annotations

import torch


def raster_order(X: int, Y: int, Z: int) -> torch.Tensor:
    """x fastest, then y, then z. Identity permutation of raster-flattened grid."""
    return torch.arange(X * Y * Z, dtype=torch.long)


def voxel_major_zyx(X: int, Y: int, Z: int) -> torch.Tensor:
    """z fastest, then y, then x. Walks vertical voxel columns."""
    perm = torch.empty(X * Y * Z, dtype=torch.long)
    idx = 0
    for x in range(X):
        for y in range(Y):
            for z in range(Z):
                # Convert (x,y,z) into the raster-order index used by the grid.
                perm[idx] = (x * Y + y) * Z + z
                idx += 1
    return perm


def _morton_encode_3d(x: int, y: int, z: int) -> int:
    """Bit-interleave (x, y, z) into a single Morton/Z-order key."""
    def part1by2(n: int) -> int:
        n &= 0x000003FF  # 10-bit mask is plenty for our grid sizes
        n = (n ^ (n << 16)) & 0xFF0000FF
        n = (n ^ (n << 8))  & 0x0300F00F
        n = (n ^ (n << 4))  & 0x030C30C3
        n = (n ^ (n << 2))  & 0x09249249
        return n
    return (part1by2(z) << 2) | (part1by2(y) << 1) | part1by2(x)


def morton_order(X: int, Y: int, Z: int) -> torch.Tensor:
    """
    Returns a permutation that traverses the grid in Morton (Z-order).
    Works for any X,Y,Z (not just powers of 2) — we sort by Morton key.
    """
    coords = [(x, y, z, (z * Y + y) * X + x)  # raster index = (z*Y+y)*X + x for z-major fastest? Actually pick a canonical raster.
              for z in range(Z) for y in range(Y) for x in range(X)]
    # Canonical raster: x fastest -> raster_idx = (z * Y + y) * X + x
    coords = [(_morton_encode_3d(x, y, z), (z * Y + y) * X + x)
              for z in range(Z) for y in range(Y) for x in range(X)]
    coords.sort(key=lambda t: t[0])
    perm = torch.tensor([raster_idx for (_, raster_idx) in coords], dtype=torch.long)
    return perm


def _hilbert_axes_to_index(coords: list[int], p: int) -> int:
    """
    Skilling's inverse Hilbert transform: maps n-D coords (each p bits wide)
    to its position along the n-D Hilbert curve. Reference: Skilling 2004,
    "Programming the Hilbert curve".

    Returns the Hilbert distance (0 .. 2^(n*p) - 1).
    """
    n = len(coords)
    coords = list(coords)

    # Inverse undo of the orientation transform.
    M = 1 << (p - 1)
    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(n):
            if coords[i] & Q:
                coords[0] ^= P
            else:
                t = (coords[0] ^ coords[i]) & P
                coords[0] ^= t
                coords[i] ^= t
        Q >>= 1

    # Gray-encode the coordinates.
    for i in range(1, n):
        coords[i] ^= coords[i - 1]
    t = 0
    Q = M
    while Q > 1:
        if coords[n - 1] & Q:
            t ^= Q - 1
        Q >>= 1
    for i in range(n):
        coords[i] ^= t

    # Bit-interleave the transposed Hilbert representation into a single int.
    h = 0
    for level in range(p):
        for dim in range(n):
            bit = (coords[dim] >> level) & 1
            h |= bit << (n * level + dim)
    return h


def hilbert_order(N: int) -> torch.Tensor:
    """
    3D Hilbert curve permutation for an N x N x N cube where N is a power of 2.
    For non-cube voxel grids we recommend padding to the next power-of-2 cube
    or falling back to morton_order which handles arbitrary shapes.

    The returned permutation `perm` satisfies: traversing voxels in the order
    `perm[0], perm[1], ..., perm[N**3 - 1]` (where each entry is a canonical
    raster index `(z * N + y) * N + x`) walks the 3D Hilbert curve.
    """
    assert N > 0 and (N & (N - 1)) == 0, f"N must be a power of 2 for hilbert_order, got {N}"
    p = N.bit_length() - 1  # N = 2**p

    pairs = []  # (hilbert_distance, raster_index)
    for z in range(N):
        for y in range(N):
            for x in range(N):
                h = _hilbert_axes_to_index([x, y, z], p)
                raster = (z * N + y) * N + x
                pairs.append((h, raster))
    pairs.sort(key=lambda t: t[0])
    return torch.tensor([raster for (_, raster) in pairs], dtype=torch.long)


def camera_frustum_order(
    X: int, Y: int, Z: int, camera_xyz: tuple[float, float, float]
) -> torch.Tensor:
    """
    Walks voxels nearest-to-camera first. Useful for streaming tokens by
    perceptual relevance — closer voxels are seen first by the agent.

    Tie-breaks by raster order for determinism.
    """
    cx, cy, cz = camera_xyz
    voxels = []
    for z in range(Z):
        for y in range(Y):
            for x in range(X):
                d2 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
                voxels.append((d2, (z * Y + y) * X + x))
    voxels.sort()
    return torch.tensor([raster_idx for (_, raster_idx) in voxels], dtype=torch.long)


ORDERINGS = {
    "raster": raster_order,
    "voxel_major_zyx": voxel_major_zyx,
    "morton": morton_order,
}
