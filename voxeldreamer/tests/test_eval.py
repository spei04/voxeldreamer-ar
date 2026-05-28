"""CPU tests for voxeldreamer.eval.*."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from voxeldreamer.data.encoder import EncoderConfig, encode_trajectory
from voxeldreamer.data.synthetic import generate_clip
from voxeldreamer.data.synthetic_loop import SquareLoopConfig, square_loop
from voxeldreamer.eval.counterfactual import counterfactual_fidelity
from voxeldreamer.eval.loop_closure import (
    LoopClosureResult,
    loop_closure_drift,
    make_oracle_rollout,
    make_random_token_rollout,
    voxel_iou,
)
from voxeldreamer.eval.voxel_recon import token_level_voxel_accuracy
from voxeldreamer.tokenizer.voxel_vocab import VoxelVocab


def test_square_loop_returns_to_start_xyz():
    """The square loop should put the agent at the same (x, y) at end as start."""
    steps = square_loop(SquareLoopConfig(side_length=4))
    assert steps[0].agent_xyz == steps[-1].agent_xyz, (steps[0].agent_xyz, steps[-1].agent_xyz)


def test_voxel_iou_perfect_match():
    a = {(0, 0, 0): 5, (1, 0, 0): 5}
    assert voxel_iou(a, a) == 1.0


def test_voxel_iou_disjoint():
    a = {(0, 0, 0): 5}
    b = {(1, 1, 1): 5}
    assert voxel_iou(a, b) == 0.0


def test_voxel_iou_empty_both():
    assert voxel_iou({}, {}) == 1.0


def test_loop_closure_oracle_is_perfect():
    """Oracle rollout (replays ground truth) should give drift = 0."""
    steps = square_loop(SquareLoopConfig(side_length=3))
    enc_cfg = EncoderConfig(patch_size=2)
    clip = encode_trajectory(steps, enc_cfg)
    oracle = make_oracle_rollout(clip.tokens)
    result = loop_closure_drift(steps, enc_cfg, oracle, burn_in_frames=1)
    assert isinstance(result, LoopClosureResult)
    assert result.drift == 0.0, result
    assert result.iou == 1.0


def test_loop_closure_random_baseline_is_terrible():
    """Random-token rollout should achieve near-zero IoU."""
    steps = square_loop(SquareLoopConfig(side_length=3))
    enc_cfg = EncoderConfig(patch_size=2)
    vocab = enc_cfg.vocab
    random_rollout = make_random_token_rollout(vocab.vocab_size, seed=0)
    result = loop_closure_drift(steps, enc_cfg, random_rollout, burn_in_frames=1)
    # Random tokens almost never match block ids at exact positions.
    assert result.iou < 0.2, result.iou


def test_counterfactual_oracle_matches_B():
    """Oracle rollout of trajectory B (given B's prefix) should give fidelity_to_B = 1."""
    steps_A = generate_clip(num_frames=6, seed=0)
    steps_B = generate_clip(num_frames=6, seed=1)
    # Force a shared prefix manually for the first 3 frames (use A's prefix for B)
    for i in range(3):
        steps_B[i] = steps_A[i]
    enc_cfg = EncoderConfig(patch_size=2)
    # Oracle rollout from B's tokens
    clip_B = encode_trajectory(steps_B, enc_cfg)
    oracle = make_oracle_rollout(clip_B.tokens)
    result = counterfactual_fidelity(
        trajectory_A=steps_A,
        trajectory_B=steps_B,
        intervention_frame=3,
        encoder_cfg=enc_cfg,
        rollout_fn=oracle,
    )
    # Oracle replays B's tokens given B's prefix, so it MUST recover B perfectly.
    assert result.fidelity_to_B == 1.0
    # A's final state may or may not coincide with B's depending on the seeds;
    # the load-bearing claim is "predicted matches B", not "B differs from A".
    assert result.fidelity_to_A <= result.fidelity_to_B


def test_voxel_recon_perfect_match():
    tokens = np.array([100, 101, 102, 103], dtype=np.int32)
    positions = np.array([
        [0, -1, -1, -1],  # not a voxel
        [0, 0, 0, 0],     # voxel
        [0, 1, 0, 0],     # voxel
        [0, 0, 1, 0],     # voxel
    ], dtype=np.int32)
    result = token_level_voxel_accuracy(
        predicted_tokens=tokens,
        ground_truth_tokens=tokens,
        positions=positions,
        block_token_offset=0,
        block_token_end=200,
    )
    assert result.accuracy == 1.0
    assert result.total_positions == 3


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} eval tests passed.")


if __name__ == "__main__":
    run_all()
