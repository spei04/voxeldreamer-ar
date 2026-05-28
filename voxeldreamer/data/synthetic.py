"""
Synthetic Minecraft-style trajectory generator for unit testing the
VoxelDreamer-AR data pipeline without needing a real Minecraft instance.

Generates a fake (frame, action, camera_pose, voxel_chunk) trajectory using
simple procedural rules:
- A WxHxD voxel world initialized with a flat "ground" layer.
- At each step, the agent picks a random action; if it's a "place" action,
  a single voxel changes; if it's a movement, the camera pose updates.
- The voxel chunk emitted per frame is the egocentric window around the agent.

This is *not* Minecraft physics — it's the minimum scaffolding needed to
exercise the loader, tokenizer, and 3D-RoPE plumbing on CPU.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SyntheticConfig:
    world_shape: tuple[int, int, int] = (32, 32, 32)
    egocentric_radius: int = 4  # produces 8x8x8 voxel windows by default
    num_block_types: int = 8
    agent_start: tuple[int, int, int] = (16, 16, 4)
    seed: int = 0


@dataclass
class TrajectoryStep:
    frame_idx: int
    voxel_window: np.ndarray  # shape (D, D, D) of block ids
    camera_yaw: float
    camera_pitch: float
    action_id: int  # index into a small action set
    agent_xyz: tuple[int, int, int]
    changed_voxels: list[tuple[int, int, int, int]]  # (x, y, z, new_block) relative to window


class SyntheticTrajectoryGenerator:
    """Single deterministic trajectory generator. Iterate to get TrajectoryStep records."""

    NUM_ACTIONS = 10  # placeholder set matching voxel_vocab default size

    def __init__(self, cfg: SyntheticConfig | None = None):
        self.cfg = cfg or SyntheticConfig()
        self.rng = random.Random(self.cfg.seed)
        self.world = self._init_world()
        self.agent_xyz = self.cfg.agent_start
        self.camera_yaw = 0.0
        self.camera_pitch = 0.0
        self._prev_window: np.ndarray | None = None

    def _init_world(self) -> np.ndarray:
        W, H, D = self.cfg.world_shape
        w = np.zeros((W, H, D), dtype=np.int32)
        # Ground layer (z=0,1) is block type 1 ("stone")
        w[:, :, :2] = 1
        return w

    def _ego_window(self) -> np.ndarray:
        r = self.cfg.egocentric_radius
        ax, ay, az = self.agent_xyz
        W, H, D = self.cfg.world_shape
        x0, x1 = max(0, ax - r), min(W, ax + r)
        y0, y1 = max(0, ay - r), min(H, ay + r)
        z0, z1 = max(0, az - r), min(D, az + r)
        return self.world[x0:x1, y0:y1, z0:z1].copy()

    def step(self, frame_idx: int) -> TrajectoryStep:
        action = self.rng.randrange(self.NUM_ACTIONS)
        changed: list[tuple[int, int, int, int]] = []

        # Action semantics (placeholders; not Minecraft-accurate)
        if action == 0:  # noop
            pass
        elif action in (1, 2, 3, 4):  # move forward/back/left/right
            dx, dy = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}[action]
            W, H, D = self.cfg.world_shape
            self.agent_xyz = (
                max(0, min(W - 1, self.agent_xyz[0] + dx)),
                max(0, min(H - 1, self.agent_xyz[1] + dy)),
                self.agent_xyz[2],
            )
        elif action == 5:  # turn yaw
            self.camera_yaw = (self.camera_yaw + 10.0) % 360.0
        elif action == 6:  # turn pitch
            self.camera_pitch = max(-90.0, min(90.0, self.camera_pitch + 5.0))
        elif action in (7, 8):  # place / break block at agent's feet+1
            ax, ay, az = self.agent_xyz
            tz = az + (1 if action == 7 else 0)
            if 0 <= tz < self.cfg.world_shape[2]:
                new_block = self.rng.randrange(1, self.cfg.num_block_types) if action == 7 else 0
                old = self.world[ax, ay, tz]
                if old != new_block:
                    self.world[ax, ay, tz] = new_block
                    # Express the change in window-local coordinates
                    r = self.cfg.egocentric_radius
                    cx, cy, cz = ax - (self.agent_xyz[0] - r), ay - (self.agent_xyz[1] - r), tz - (self.agent_xyz[2] - r)
                    if 0 <= cx < 2 * r and 0 <= cy < 2 * r and 0 <= cz < 2 * r:
                        changed.append((cx, cy, cz, int(new_block)))
        # action 9: unused / no-effect

        window = self._ego_window()
        step = TrajectoryStep(
            frame_idx=frame_idx,
            voxel_window=window,
            camera_yaw=self.camera_yaw,
            camera_pitch=self.camera_pitch,
            action_id=action,
            agent_xyz=self.agent_xyz,
            changed_voxels=changed,
        )
        self._prev_window = window
        return step

    def generate(self, n_frames: int) -> list[TrajectoryStep]:
        return [self.step(t) for t in range(n_frames)]


def generate_clip(num_frames: int = 16, seed: int = 0) -> list[TrajectoryStep]:
    """Convenience: one synthetic clip of `num_frames` steps."""
    return SyntheticTrajectoryGenerator(SyntheticConfig(seed=seed)).generate(num_frames)
