"""
Encode a synthetic TrajectoryStep stream into a VoxelDreamer-AR Clip.

This is the bridge from the simulator (or future MineRL collector) to the
on-disk shard format. The encoding is the strawman baseline described in
voxeldreamer/data/README.md: per-frame patch-token re-emission with no delta
encoding yet. Delta encoding is a Phase 2 follow-up.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxeldreamer.data.shard_format import Clip
from voxeldreamer.data.synthetic import TrajectoryStep
from voxeldreamer.tokenizer.orderings import ORDERINGS
from voxeldreamer.tokenizer.voxel_vocab import VoxelVocab


@dataclass
class EncoderConfig:
    voxel_ordering: str = "voxel_major_zyx"
    # Patch size in voxels. P=1 = one token per voxel (baseline);
    # P>1 hashes a patch into a single token by majority block id.
    patch_size: int = 1
    vocab: VoxelVocab = None

    def __post_init__(self):
        if self.vocab is None:
            self.vocab = VoxelVocab()
        assert self.voxel_ordering in ORDERINGS, (
            f"unknown ordering {self.voxel_ordering!r} (have {list(ORDERINGS)})"
        )
        assert self.patch_size >= 1


def _patch_tokenize(window: np.ndarray, P: int) -> np.ndarray:
    """
    Reduce a [W, H, D] voxel window to a [W/P, H/P, D/P] grid of single-token
    representations. Token = majority block id in each patch.
    """
    if P == 1:
        return window
    W, H, D = window.shape
    assert W % P == 0 and H % P == 0 and D % P == 0, (
        f"window shape {window.shape} not divisible by patch size {P}"
    )
    Wp, Hp, Dp = W // P, H // P, D // P
    out = np.zeros((Wp, Hp, Dp), dtype=window.dtype)
    for i in range(Wp):
        for j in range(Hp):
            for k in range(Dp):
                patch = window[i*P:(i+1)*P, j*P:(j+1)*P, k*P:(k+1)*P]
                vals, counts = np.unique(patch, return_counts=True)
                out[i, j, k] = vals[counts.argmax()]
    return out


def encode_trajectory(
    steps: list[TrajectoryStep],
    cfg: EncoderConfig | None = None,
    clip_id: int = 0,
) -> Clip:
    """Encode a list of TrajectoryStep into a single Clip ready for sharding."""
    cfg = cfg or EncoderConfig()
    vocab = cfg.vocab

    tokens: list[int] = []
    pt: list[int] = []
    px: list[int] = []
    py: list[int] = []
    pz: list[int] = []

    for step in steps:
        patches = _patch_tokenize(step.voxel_window, cfg.patch_size)
        Wp, Hp, Dp = patches.shape
        ordering_fn = ORDERINGS[cfg.voxel_ordering]
        perm = ordering_fn(Wp, Hp, Dp).numpy()

        # Header tokens: <frame_sep> <camera_sep> yaw pitch <action_sep> action <bos>
        header = [
            vocab.special_token("<frame_sep>"),
            vocab.special_token("<camera_sep>"),
            vocab.yaw_token(step.camera_yaw),
            vocab.pitch_token(step.camera_pitch),
            vocab.special_token("<action_sep>"),
            vocab.action_token(step.action_id),
            vocab.special_token("<bos>"),
        ]
        for tok in header:
            tokens.append(int(tok))
            pt.append(step.frame_idx)
            px.append(-1); py.append(-1); pz.append(-1)

        # Voxel tokens in chosen order
        flat = patches.reshape(-1)  # canonical raster order (x,y,z): flat[(z*Hp + y)*Wp + x]
        # But our orderings.py uses canonical (z*Y+y)*X+x; numpy default flatten is C-order
        # over (W, H, D). With shape (Wp, Hp, Dp), flat[((w*Hp)+h)*Dp+d] = patches[w,h,d].
        # Reconciling: orderings.py expects flat[(z*Y+y)*X+x] for shape (X, Y, Z) with
        # x fastest. We pass our patch grid as (X=Wp, Y=Hp, Z=Dp) and flatten in
        # x-fastest order to match.
        flat_xfast = np.transpose(patches, (2, 1, 0)).reshape(-1)  # now flat[(z*Hp + y)*Wp + x]

        for token_idx, raster in enumerate(perm):
            block_id = int(flat_xfast[raster])
            tokens.append(vocab.block_token(block_id))
            pt.append(step.frame_idx)
            # Recover (x, y, z) for this raster index, shape (Wp, Hp, Dp)
            x = raster % Wp
            y = (raster // Wp) % Hp
            z = raster // (Wp * Hp)
            px.append(x); py.append(y); pz.append(z)

    return Clip(
        clip_id=clip_id,
        tokens=np.asarray(tokens, dtype=np.int32),
        positions_t=np.asarray(pt, dtype=np.int32),
        positions_x=np.asarray(px, dtype=np.int16),
        positions_y=np.asarray(py, dtype=np.int16),
        positions_z=np.asarray(pz, dtype=np.int16),
        num_frames=len(steps),
    )
