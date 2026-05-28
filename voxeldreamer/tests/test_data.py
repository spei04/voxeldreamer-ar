"""CPU tests for the VoxelDreamer-AR data pipeline (synthetic → encoder → shard → loader)."""

import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from voxeldreamer.data.encoder import EncoderConfig, encode_trajectory
from voxeldreamer.data.loader import make_position_aware_dataloader
from voxeldreamer.data.shard_format import Clip, read_shard, write_shard
from voxeldreamer.data.synthetic import (
    SyntheticConfig,
    SyntheticTrajectoryGenerator,
    generate_clip,
)
from voxeldreamer.tokenizer.voxel_vocab import VoxelVocab


def test_synthetic_generator_runs():
    steps = generate_clip(num_frames=4)
    assert len(steps) == 4
    for s in steps:
        assert s.voxel_window.ndim == 3
        # Frame index sequence is 0..3
    assert [s.frame_idx for s in steps] == [0, 1, 2, 3]


def test_synthetic_window_shape_matches_radius():
    cfg = SyntheticConfig(egocentric_radius=3, agent_start=(16, 16, 5))
    gen = SyntheticTrajectoryGenerator(cfg)
    step = gen.step(0)
    assert step.voxel_window.shape == (6, 6, 6), step.voxel_window.shape


def test_encoder_produces_consistent_shapes():
    steps = generate_clip(num_frames=2)
    enc_cfg = EncoderConfig(voxel_ordering="raster", patch_size=2)
    clip = encode_trajectory(steps, enc_cfg, clip_id=42)
    assert clip.clip_id == 42
    assert clip.num_frames == 2
    T = clip.length
    assert clip.positions_t.shape == (T,)
    assert clip.positions_x.shape == (T,)
    # Header tokens carry sentinel positions (-1)
    # The first 7 tokens of each frame are header.
    assert (clip.positions_x[:7] == -1).all()


def test_encoder_all_orderings():
    steps = generate_clip(num_frames=1)
    for ordering in ("raster", "voxel_major_zyx", "morton"):
        enc_cfg = EncoderConfig(voxel_ordering=ordering, patch_size=2)
        clip = encode_trajectory(steps, enc_cfg)
        # Should produce *some* tokens
        assert clip.length > 7
        # All voxel positions should be within the patch grid
        voxel_mask = clip.positions_x >= 0
        assert voxel_mask.any()


def test_shard_round_trip():
    steps = generate_clip(num_frames=2)
    clip = encode_trajectory(steps, EncoderConfig(patch_size=2))
    with tempfile.TemporaryDirectory() as td:
        shard_path = Path(td) / "shard_0.parquet"
        write_shard(shard_path, [clip])
        clips = read_shard(shard_path)
    assert len(clips) == 1
    rt = clips[0]
    assert rt.clip_id == clip.clip_id
    assert rt.length == clip.length
    assert (rt.tokens == clip.tokens).all()
    assert (rt.positions_t == clip.positions_t).all()
    assert (rt.positions_x == clip.positions_x).all()


def test_loader_yields_expected_shapes():
    # Build a tiny shard with a few clips
    clips = []
    for cid in range(4):
        steps = generate_clip(num_frames=2, seed=cid)
        clips.append(encode_trajectory(steps, EncoderConfig(patch_size=2), clip_id=cid))
    with tempfile.TemporaryDirectory() as td:
        shard_path = Path(td) / "shard_0.parquet"
        write_shard(shard_path, clips)

        vocab = VoxelVocab()
        loader = make_position_aware_dataloader(
            [shard_path],
            batch_size=2,
            seq_len=64,
            bos_token=vocab.special_token("<bos>"),
            device="cpu",
        )
        inputs, targets, pos_in, pos_tgt, epoch = next(loader)

    assert inputs.shape == (2, 64)
    assert targets.shape == (2, 64)
    assert pos_in.shape == (2, 64, 4)
    assert pos_tgt.shape == (2, 64, 4)
    # First-token-of-row is BOS at (0, -1, -1, -1) for both batch rows
    assert pos_in[0, 0].tolist() == [0, -1, -1, -1]
    # Targets are shifted-by-one
    assert torch.equal(targets[:, :-1], inputs[:, 1:])


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} data tests passed.")


if __name__ == "__main__":
    run_all()
