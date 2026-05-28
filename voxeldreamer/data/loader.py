"""
Position-aware dataloader for VoxelDreamer-AR.

Extends autoresearch's `make_dataloader` interface from `(inputs, targets, epoch)`
to `(inputs, targets, positions_in, positions_tgt, epoch)`. The 3D-RoPE path in
train.py uses `positions_in` to compute (cos, sin) tables per batch.

CPU-only loader — when integrated into train.py, the GPU buffers and pinned
memory will mirror the existing autoresearch implementation. This module is
the design + skeleton; the GPU-integration line items are in Phase 2 W4.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from voxeldreamer.data.shard_format import Clip, read_shard


def iter_clips_from_shards(shard_paths: list[str | Path]):
    """Yield Clip objects from a list of shard files. Cycles forever."""
    epoch = 0
    while True:
        epoch += 1
        for p in shard_paths:
            for clip in read_shard(p):
                yield clip, epoch


def make_position_aware_dataloader(
    shard_paths: list[str | Path],
    batch_size: int,
    seq_len: int,
    *,
    bos_token: int,
    pad_token: int | None = None,
    device: torch.device | str = "cpu",
):
    """
    CPU-friendly position-aware dataloader.

    Packs clips into rows of length `seq_len + 1`. Each row begins with `bos_token`
    (carrying position (0, -1, -1, -1)). When a clip doesn't fit, we crop the
    *shortest* clip in the buffer to fill the row exactly — matching autoresearch's
    100%-utilization policy.

    Yields:
        inputs, targets:     [B, T] long tensors on `device`
        positions_in:        [B, T, 4] long tensor on `device`  -> use for RoPE in fwd
        positions_tgt:       [B, T, 4] long tensor on `device`  -> rarely needed but exposed
        epoch:               int (1-indexed cycle through shard_paths)
    """
    row_capacity = seq_len + 1
    clip_stream = iter_clips_from_shards(shard_paths)
    clip_buffer: list[Clip] = []
    BUFFER_SIZE = 64

    def refill():
        nonlocal epoch
        while len(clip_buffer) < BUFFER_SIZE:
            clip, ep = next(clip_stream)
            epoch = ep
            clip_buffer.append(clip)

    epoch = 1
    refill()

    # Pre-allocate row buffers on CPU.
    row_tokens = np.zeros((batch_size, row_capacity), dtype=np.int64)
    row_positions = np.zeros((batch_size, row_capacity, 4), dtype=np.int64)

    while True:
        for b in range(batch_size):
            pos = 0
            # Start each row with BOS at position (0, -1, -1, -1)
            row_tokens[b, 0] = bos_token
            row_positions[b, 0] = [0, -1, -1, -1]
            pos = 1
            while pos < row_capacity:
                refill()
                remaining = row_capacity - pos

                # Best-fit: largest clip that fully fits
                best_idx, best_len = -1, 0
                for i, c in enumerate(clip_buffer):
                    if c.length <= remaining and c.length > best_len:
                        best_idx, best_len = i, c.length
                if best_idx >= 0:
                    c = clip_buffer.pop(best_idx)
                    row_tokens[b, pos:pos + c.length] = c.tokens
                    row_positions[b, pos:pos + c.length] = c.positions_4d()
                    pos += c.length
                else:
                    # Crop the shortest clip to fill
                    j = min(range(len(clip_buffer)), key=lambda i: clip_buffer[i].length)
                    c = clip_buffer.pop(j)
                    row_tokens[b, pos:pos + remaining] = c.tokens[:remaining]
                    row_positions[b, pos:pos + remaining] = c.positions_4d()[:remaining]
                    pos += remaining

        tokens = torch.from_numpy(row_tokens).to(device=device)
        positions = torch.from_numpy(row_positions).to(device=device)
        inputs = tokens[:, :-1].contiguous()
        targets = tokens[:, 1:].contiguous()
        positions_in = positions[:, :-1].contiguous()
        positions_tgt = positions[:, 1:].contiguous()
        yield inputs, targets, positions_in, positions_tgt, epoch
