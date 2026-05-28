"""
Counterfactual fidelity evaluation.

Given two trajectories that share a common prefix and then diverge by a
single action intervention, measure whether the model's predicted divergence
matches the ground-truth divergence.

Concretely:
  - trajectory_A: actions [a1, a2, ..., a_t, a_{t+1}, ..., a_N]
  - trajectory_B: actions [a1, a2, ..., a'_t, a_{t+1}, ..., a_N]
                  (identical prefix, different action at step t)

  Both trajectories are tokenized.
  We feed the model the shared prefix tokens + the divergent action token,
  ask it to roll out to step N, and compare against ground truth B.

A good world model:
  - Diverges from A's continuation when the action diverges.
  - Converges to B's specific outcome (not just "any divergence").

Counterfactual fidelity = IoU(predicted_final_voxels, B_final_voxels).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxeldreamer.data.encoder import EncoderConfig, encode_trajectory
from voxeldreamer.data.synthetic import TrajectoryStep
from voxeldreamer.eval.loop_closure import (
    RolloutFn,
    _decode_final_frame_voxels,
    voxel_iou,
)


@dataclass
class CounterfactualResult:
    fidelity_to_B: float       # IoU between prediction and trajectory_B's truth
    fidelity_to_A: float       # IoU between prediction and trajectory_A's truth (should be lower)
    intervention_responsive: bool  # True iff fidelity_to_B > fidelity_to_A


def counterfactual_fidelity(
    trajectory_A: list[TrajectoryStep],
    trajectory_B: list[TrajectoryStep],
    intervention_frame: int,
    encoder_cfg: EncoderConfig,
    rollout_fn: RolloutFn,
) -> CounterfactualResult:
    assert len(trajectory_A) == len(trajectory_B)
    assert 0 < intervention_frame < len(trajectory_A)
    # Sanity: prefix really is shared.
    for i in range(intervention_frame):
        if trajectory_A[i].action_id != trajectory_B[i].action_id:
            raise ValueError(
                f"trajectories differ before intervention_frame at step {i}: "
                f"{trajectory_A[i].action_id} vs {trajectory_B[i].action_id}"
            )

    clip_A = encode_trajectory(trajectory_A, encoder_cfg, clip_id=0)
    clip_B = encode_trajectory(trajectory_B, encoder_cfg, clip_id=1)
    pos_A = clip_A.positions_4d()
    pos_B = clip_B.positions_4d()

    # Cut B at the first token belonging to the intervention frame: model gets
    # everything up to and including that frame's action_id token, then rolls out.
    intervention_mask = pos_B[:, 0] >= intervention_frame
    first_intervention_idx = int(intervention_mask.argmax())
    # Find the action token inside the intervention frame (it sits in the header).
    # For simplicity, include the whole header of the intervention frame in the prefix.
    next_frame_mask = pos_B[:, 0] >= (intervention_frame + 1)
    if next_frame_mask.any():
        cut_idx = int(next_frame_mask.argmax())
    else:
        cut_idx = clip_B.length

    prefix_tokens = clip_B.tokens[:cut_idx]
    prefix_positions = pos_B[:cut_idx]
    num_to_generate = clip_B.length - cut_idx
    if num_to_generate <= 0:
        raise ValueError("intervention_frame is the last frame — nothing to generate")

    pred_post = rollout_fn(prefix_tokens, prefix_positions, num_to_generate)

    final_frame_idx = int(pos_B[:, 0].max())
    gt_B_final = _decode_final_frame_voxels(clip_B.tokens, pos_B, final_frame_idx, encoder_cfg.vocab)
    gt_A_final = _decode_final_frame_voxels(clip_A.tokens, pos_A, final_frame_idx, encoder_cfg.vocab)
    pred_full_tokens = np.concatenate([prefix_tokens, pred_post])
    pred_final = _decode_final_frame_voxels(pred_full_tokens, pos_B, final_frame_idx, encoder_cfg.vocab)

    iou_B = voxel_iou(pred_final, gt_B_final)
    iou_A = voxel_iou(pred_final, gt_A_final)
    return CounterfactualResult(
        fidelity_to_B=iou_B,
        fidelity_to_A=iou_A,
        intervention_responsive=iou_B > iou_A,
    )
