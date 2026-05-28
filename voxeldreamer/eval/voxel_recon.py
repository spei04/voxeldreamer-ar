"""
Voxel state reconstruction evaluation.

Auxiliary metric: at a fixed frame, can the model's hidden state (or its
auxiliary voxel-prediction head, if present) recover the ground-truth voxel
window?

Used for Tier 3 ablations that add an auxiliary voxel-prediction loss
(see RESEARCH.md). For Tier 1/2 sweeps it is informational only.

Two flavors:
  - Token-level: just check whether the next-token argmax over voxel positions
                 matches the ground-truth block id. Reuses the per-position
                 cross-entropy already computed during training.
  - Direct readout: requires the model to expose an aux head that projects the
                    hidden state at each voxel position into a block-type logit.
                    Skeleton interface only — not used in Phase 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelReconResult:
    accuracy: float       # fraction of voxel positions whose predicted block matches GT
    total_positions: int


def token_level_voxel_accuracy(
    predicted_tokens: np.ndarray,
    ground_truth_tokens: np.ndarray,
    positions: np.ndarray,
    block_token_offset: int,
    block_token_end: int,
) -> VoxelReconResult:
    """
    Count only positions whose ground-truth token is a block token; over those,
    return the fraction where prediction == ground truth.

    Args:
        predicted_tokens: [T] int
        ground_truth_tokens: [T] int
        positions: [T, 4] int — used only to identify voxel positions (x >= 0)
        block_token_offset / block_token_end: vocab range for block tokens.
    """
    assert predicted_tokens.shape == ground_truth_tokens.shape
    voxel_mask = positions[:, 1] >= 0
    block_mask_gt = (
        (ground_truth_tokens >= block_token_offset)
        & (ground_truth_tokens < block_token_end)
    )
    consider = voxel_mask & block_mask_gt
    total = int(consider.sum())
    if total == 0:
        return VoxelReconResult(accuracy=1.0, total_positions=0)
    correct = int(((predicted_tokens == ground_truth_tokens) & consider).sum())
    return VoxelReconResult(accuracy=correct / total, total_positions=total)
