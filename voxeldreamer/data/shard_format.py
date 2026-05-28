"""
Shard format spec + writer/reader for VoxelDreamer-AR trajectories.

Shards are stored under `~/.cache/autoresearch/voxeldreamer/` to mirror the
upstream autoresearch convention. Each shard is one Parquet file with one row
per clip.

Columns (see voxeldreamer/data/README.md for the spec):

  clip_id       int64
  tokens        list[int32]
  positions_t   list[int32]
  positions_x   list[int16]
  positions_y   list[int16]
  positions_z   list[int16]
  num_frames    int32

Position columns carry per-token (t, x, y, z); non-voxel tokens use -1 in
x/y/z (matching the action-marker sentinel used by rope_3d.py).

Note: pyarrow is a hard dependency for writing/reading these shards, but the
upstream autoresearch already pulls it in via pandas/pyarrow for the Parquet
loaders in prepare.py. We import it lazily here so the synthetic-data path
remains usable on a fresh checkout without optional deps installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Clip:
    clip_id: int
    tokens: np.ndarray         # int32 [T]
    positions_t: np.ndarray    # int32 [T]
    positions_x: np.ndarray    # int16 [T]
    positions_y: np.ndarray    # int16 [T]
    positions_z: np.ndarray    # int16 [T]
    num_frames: int

    def __post_init__(self):
        T = self.tokens.shape[0]
        for arr_name in ("positions_t", "positions_x", "positions_y", "positions_z"):
            arr = getattr(self, arr_name)
            assert arr.shape == (T,), f"{arr_name} shape {arr.shape} != ({T},)"

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])

    def positions_4d(self) -> np.ndarray:
        """Stack the four per-axis position arrays into a [T, 4] int array."""
        return np.stack(
            [self.positions_t, self.positions_x, self.positions_y, self.positions_z], axis=1
        ).astype(np.int32)


def write_shard(path: str | Path, clips: Iterable[Clip]) -> None:
    """Write a list of Clips to a single Parquet shard at `path`."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    clip_list = list(clips)
    table = pa.table({
        "clip_id":     pa.array([c.clip_id for c in clip_list], type=pa.int64()),
        "tokens":      pa.array([c.tokens.tolist() for c in clip_list], type=pa.list_(pa.int32())),
        "positions_t": pa.array([c.positions_t.tolist() for c in clip_list], type=pa.list_(pa.int32())),
        "positions_x": pa.array([c.positions_x.tolist() for c in clip_list], type=pa.list_(pa.int16())),
        "positions_y": pa.array([c.positions_y.tolist() for c in clip_list], type=pa.list_(pa.int16())),
        "positions_z": pa.array([c.positions_z.tolist() for c in clip_list], type=pa.list_(pa.int16())),
        "num_frames":  pa.array([c.num_frames for c in clip_list], type=pa.int32()),
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(path))


def read_shard(path: str | Path) -> list[Clip]:
    """Read a Parquet shard back into a list of Clips."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    n = table.num_rows
    clip_ids   = table["clip_id"].to_pylist()
    tokens_l   = table["tokens"].to_pylist()
    pt_l       = table["positions_t"].to_pylist()
    px_l       = table["positions_x"].to_pylist()
    py_l       = table["positions_y"].to_pylist()
    pz_l       = table["positions_z"].to_pylist()
    nf_l       = table["num_frames"].to_pylist()

    out: list[Clip] = []
    for i in range(n):
        out.append(Clip(
            clip_id=int(clip_ids[i]),
            tokens=np.asarray(tokens_l[i], dtype=np.int32),
            positions_t=np.asarray(pt_l[i], dtype=np.int32),
            positions_x=np.asarray(px_l[i], dtype=np.int16),
            positions_y=np.asarray(py_l[i], dtype=np.int16),
            positions_z=np.asarray(pz_l[i], dtype=np.int16),
            num_frames=int(nf_l[i]),
        ))
    return out
