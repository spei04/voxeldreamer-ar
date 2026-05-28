"""
Token vocabulary for VoxelDreamer-AR.

Single unified vocab covering:
- Voxel block types (one token per block type, e.g. AIR, STONE, DIRT, ...).
- Actions (discretized control: keys + mouse).
- Camera pose (quantized yaw/pitch buckets).
- Structural tokens (frame separator, BOS).

This is intentionally minimal: ~256 block types + ~70 action tokens + ~100
camera tokens + a handful of special tokens, comfortably under MineWorld's
8262-vocab. Smaller vocab => more parameter budget on the transformer itself
under autoresearch's fixed time budget.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import torch


# A trimmed default set; expand to match what the actual data pipeline emits.
DEFAULT_BLOCK_TYPES = (
    "air", "stone", "dirt", "grass", "wood", "leaves", "sand", "water",
    "lava", "cobblestone", "planks", "bedrock", "gravel", "iron_ore",
    "coal_ore", "diamond_ore", "log", "glass", "wool", "snow", "ice",
)

DEFAULT_ACTIONS = (
    "noop",
    "forward", "back", "left", "right", "jump", "sneak", "sprint",
    "attack", "use",
    "hotbar_1", "hotbar_2", "hotbar_3", "hotbar_4", "hotbar_5",
    "hotbar_6", "hotbar_7", "hotbar_8", "hotbar_9",
)


@dataclass
class VoxelVocab:
    block_types: tuple[str, ...] = DEFAULT_BLOCK_TYPES
    actions: tuple[str, ...] = DEFAULT_ACTIONS
    yaw_buckets: int = 36   # 10-degree buckets over 360
    pitch_buckets: int = 18  # 10-degree buckets over 180

    # Filled in __post_init__
    block_offset: int = field(init=False)
    action_offset: int = field(init=False)
    yaw_offset: int = field(init=False)
    pitch_offset: int = field(init=False)
    special_offset: int = field(init=False)
    vocab_size: int = field(init=False)

    def __post_init__(self):
        self.block_offset = 0
        self.action_offset = self.block_offset + len(self.block_types)
        self.yaw_offset = self.action_offset + len(self.actions)
        self.pitch_offset = self.yaw_offset + self.yaw_buckets
        self.special_offset = self.pitch_offset + self.pitch_buckets
        # Specials: BOS, FRAME_SEP, CAMERA_SEP, ACTION_SEP, EOS, PAD
        self.specials = ("<bos>", "<frame_sep>", "<camera_sep>", "<action_sep>", "<eos>", "<pad>")
        self.vocab_size = self.special_offset + len(self.specials)

    # ---- Encode helpers ----

    def block_token(self, block: str | int) -> int:
        if isinstance(block, str):
            idx = self.block_types.index(block)
        else:
            idx = int(block)
        return self.block_offset + idx

    def action_token(self, action: str | int) -> int:
        if isinstance(action, str):
            idx = self.actions.index(action)
        else:
            idx = int(action)
        return self.action_offset + idx

    def yaw_token(self, yaw_degrees: float) -> int:
        bucket = int((yaw_degrees % 360) / (360 / self.yaw_buckets))
        return self.yaw_offset + bucket

    def pitch_token(self, pitch_degrees: float) -> int:
        clamped = max(-90.0, min(90.0, pitch_degrees))
        bucket = int((clamped + 90) / (180 / self.pitch_buckets))
        bucket = min(bucket, self.pitch_buckets - 1)
        return self.pitch_offset + bucket

    def special_token(self, name: str) -> int:
        return self.special_offset + self.specials.index(name)

    # ---- Encode whole frames ----

    def encode_frame(
        self,
        voxel_grid_flat: Iterable[int],
        camera_yaw: float,
        camera_pitch: float,
        action: str | int,
    ) -> list[int]:
        """
        Returns the token stream for one (frame, camera, action) tuple in
        canonical layout:

            <frame_sep> <camera_sep> yaw pitch <action_sep> action <bos>
            v0 v1 v2 ... vN

        Voxel ordering is the caller's responsibility — pass the already-
        permuted flat voxel-id array (see tokenizer/orderings.py).
        """
        out = [
            self.special_token("<frame_sep>"),
            self.special_token("<camera_sep>"),
            self.yaw_token(camera_yaw),
            self.pitch_token(camera_pitch),
            self.special_token("<action_sep>"),
            self.action_token(action),
        ]
        out.extend(self.block_token(v) for v in voxel_grid_flat)
        return out

    # ---- Decode helpers (mostly for debugging) ----

    def kind_of(self, token: int) -> str:
        if token < self.action_offset:
            return "block"
        if token < self.yaw_offset:
            return "action"
        if token < self.pitch_offset:
            return "yaw"
        if token < self.special_offset:
            return "pitch"
        return "special"
