"""
Loop-closure drift evaluation for VoxelDreamer-AR.

The crown-jewel eval. Given a closed-loop trajectory (start state = final state),
measure how much the model's *predicted* final voxel state diverges from the
ground-truth final voxel state when the model has only the start frame as context
and must autoregressively roll out the rest.

Drift = 1 - voxel-IoU between predicted and ground-truth final-frame voxel grid.
       = 0 means perfect consistency; 1 means total amnesia.

Implementation is model-agnostic: pass any callable that maps
(prefix_tokens, prefix_positions, num_tokens_to_generate) -> generated_tokens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

from voxeldreamer.data.encoder import EncoderConfig, encode_trajectory
from voxeldreamer.data.shard_format import Clip
from voxeldreamer.data.synthetic import TrajectoryStep


class RolloutFn(Protocol):
    def __call__(
        self,
        prefix_tokens: np.ndarray,    # [T_prefix] int
        prefix_positions: np.ndarray, # [T_prefix, 4] int
        num_to_generate: int,
    ) -> np.ndarray:                  # [num_to_generate] int — predicted tokens
        ...


@dataclass
class LoopClosureResult:
    drift: float                 # 1 - IoU(pred_final, gt_final). Lower is better.
    iou: float
    num_voxel_tokens_pred: int
    num_voxel_tokens_gt: int


def _decode_final_frame_voxels(
    tokens: np.ndarray,
    positions: np.ndarray,
    final_frame_idx: int,
    vocab,
) -> dict[tuple[int, int, int], int]:
    """
    From a flat (tokens, positions) sequence, extract the last-frame voxel
    state as a dict mapping (x, y, z) -> block_token_id.

    Only voxel tokens (those whose position has x >= 0) are considered.
    """
    out: dict[tuple[int, int, int], int] = {}
    for tok, pos in zip(tokens.tolist(), positions.tolist()):
        t, x, y, z = pos
        if t != final_frame_idx:
            continue
        if x < 0:
            continue  # skip header / action / camera tokens
        if vocab.kind_of(int(tok)) != "block":
            continue
        out[(int(x), int(y), int(z))] = int(tok)
    return out


def voxel_iou(
    pred: dict[tuple[int, int, int], int],
    gt: dict[tuple[int, int, int], int],
) -> float:
    """
    Intersection-over-union over voxel-token identity. Position must match AND
    the token (block type) must match for a voxel to count as "intersected".
    """
    if not pred and not gt:
        return 1.0
    pred_set = {(pos, tok) for pos, tok in pred.items()}
    gt_set = {(pos, tok) for pos, tok in gt.items()}
    inter = len(pred_set & gt_set)
    union = len(pred_set | gt_set)
    if union == 0:
        return 1.0
    return inter / union


def loop_closure_drift(
    steps: list[TrajectoryStep],
    encoder_cfg: EncoderConfig,
    rollout_fn: RolloutFn,
    *,
    burn_in_frames: int = 1,
) -> LoopClosureResult:
    """
    Args:
        steps: a closed-loop trajectory (last frame's state matches first frame).
        encoder_cfg: EncoderConfig used to tokenize the trajectory.
        rollout_fn: model-side predict-next-N-tokens function.
        burn_in_frames: how many leading frames the model gets as context.

    Returns:
        LoopClosureResult with drift = 1 - IoU(pred_final_voxels, gt_final_voxels).
    """
    assert burn_in_frames >= 1 and burn_in_frames < len(steps)

    full_clip = encode_trajectory(steps, encoder_cfg)
    full_tokens = full_clip.tokens
    full_positions = full_clip.positions_4d()

    # Determine the prefix length (in tokens) corresponding to `burn_in_frames`.
    # Frames are sequential in `positions_t`; find the first token where
    # positions_t > (burn_in_frames - 1).
    pos_t = full_positions[:, 0]
    after_burn_in_mask = pos_t >= burn_in_frames
    if not after_burn_in_mask.any():
        raise ValueError("burn_in_frames covers the entire trajectory")
    first_post_idx = int(after_burn_in_mask.argmax())

    prefix_tokens = full_tokens[:first_post_idx]
    prefix_positions = full_positions[:first_post_idx]
    num_to_generate = full_tokens.shape[0] - first_post_idx

    pred_post = rollout_fn(prefix_tokens, prefix_positions, num_to_generate)
    assert pred_post.shape[0] == num_to_generate, (
        f"rollout returned {pred_post.shape[0]} tokens, expected {num_to_generate}"
    )

    # Reconstruct the full predicted sequence by stitching prefix + prediction.
    # The positions of the post-burn-in tokens are *known* from the trajectory
    # layout — the model isn't predicting positions, just tokens.
    final_frame_idx = int(pos_t.max())
    gt_final = _decode_final_frame_voxels(full_tokens, full_positions, final_frame_idx, encoder_cfg.vocab)
    pred_full_tokens = np.concatenate([prefix_tokens, pred_post])
    pred_final = _decode_final_frame_voxels(pred_full_tokens, full_positions, final_frame_idx, encoder_cfg.vocab)

    iou = voxel_iou(pred_final, gt_final)
    return LoopClosureResult(
        drift=1.0 - iou,
        iou=iou,
        num_voxel_tokens_pred=len(pred_final),
        num_voxel_tokens_gt=len(gt_final),
    )


def make_oracle_rollout(full_tokens: np.ndarray) -> RolloutFn:
    """
    Returns a rollout function that 'predicts' by replaying the ground-truth
    suffix. Used for sanity checks: loop_closure_drift should return 0.0.
    """
    def fn(prefix_tokens, prefix_positions, num_to_generate):
        offset = prefix_tokens.shape[0]
        return full_tokens[offset:offset + num_to_generate].copy()
    return fn


def make_random_token_rollout(vocab_size: int, seed: int = 0) -> RolloutFn:
    """
    Random-token rollout. Establishes the floor: loop_closure_drift here
    should be near 1.0 (essentially zero IoU with truth).
    """
    rng = np.random.default_rng(seed)
    def fn(prefix_tokens, prefix_positions, num_to_generate):
        return rng.integers(0, vocab_size, size=num_to_generate, dtype=np.int32)
    return fn
