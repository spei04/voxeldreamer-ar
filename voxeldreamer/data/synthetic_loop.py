"""
Closed-loop synthetic trajectory generator.

Produces trajectories where the agent's (x, y, z, yaw, pitch) at the final
frame matches the initial state, so loop-closure drift becomes well-defined:
the predicted state at frame N-1 should match the ground-truth voxel window
at frame 0.

The default "square walk" pattern: take K steps forward, turn 90 left,
repeat 4 times. After 4*K steps the agent is back where it started.

Place / break actions are interleaved deterministically so the world changes
along the way and the eval is sensitive to whether the model tracked those
changes through the loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from voxeldreamer.data.synthetic import (
    SyntheticConfig,
    SyntheticTrajectoryGenerator,
    TrajectoryStep,
)


@dataclass
class SquareLoopConfig:
    side_length: int = 4
    yaw_turns_per_corner: int = 9  # 9 * 10deg = 90deg using the gen's yaw step
    seed: int = 0
    place_every: int = 0  # 0 = never place; >0 = place a block every N steps


def square_loop(cfg: SquareLoopConfig | None = None) -> list[TrajectoryStep]:
    """Walk a closed square loop and return the trajectory.

    Frame layout: forward x K, yaw-left x M, forward x K, yaw-left x M, ...
    After 4 sides + 4 corners the agent is at the starting (x, y, z, yaw).
    """
    cfg = cfg or SquareLoopConfig()
    sim = SyntheticTrajectoryGenerator(SyntheticConfig(seed=cfg.seed))

    # Forward in the generator's action set is action_id=1.
    # Yaw turn is action_id=5.
    forward_id = 1
    yaw_id = 5
    place_id = 7

    # We need the simulator to deterministically execute given action ids
    # but its `step` samples actions via its internal RNG. Provide a small
    # wrapper that steps with a chosen action.

    steps: list[TrajectoryStep] = []
    frame_idx = 0

    def do_action(action_id: int, frame_idx: int) -> TrajectoryStep:
        # Mimic SyntheticTrajectoryGenerator.step but with a fixed action.
        # We forward the same world-update semantics by temporarily replacing rng.
        # Cleaner: re-implement the action effect here so we don't fight the rng.
        if action_id in (1, 2, 3, 4):
            dx, dy = {1: (0, 1), 2: (0, -1), 3: (-1, 0), 4: (1, 0)}[action_id]
            W, H, _ = sim.cfg.world_shape
            sim.agent_xyz = (
                max(0, min(W - 1, sim.agent_xyz[0] + dx)),
                max(0, min(H - 1, sim.agent_xyz[1] + dy)),
                sim.agent_xyz[2],
            )
        elif action_id == 5:
            sim.camera_yaw = (sim.camera_yaw + 10.0) % 360.0
        elif action_id == 6:
            sim.camera_pitch = max(-90.0, min(90.0, sim.camera_pitch + 5.0))
        elif action_id == 7:
            ax, ay, az = sim.agent_xyz
            tz = az + 1
            if 0 <= tz < sim.cfg.world_shape[2]:
                sim.world[ax, ay, tz] = 1  # always "stone" for determinism
        # other action ids: no-op for this loop generator

        window = sim._ego_window()
        s = TrajectoryStep(
            frame_idx=frame_idx,
            voxel_window=window,
            camera_yaw=sim.camera_yaw,
            camera_pitch=sim.camera_pitch,
            action_id=action_id,
            agent_xyz=sim.agent_xyz,
            changed_voxels=[],
        )
        return s

    # Record the initial state as frame 0 (no action applied yet).
    steps.append(do_action(0, frame_idx))  # noop just to capture the initial window
    frame_idx += 1

    sides_walked = 0
    for side in range(4):
        # Walk forward `side_length` steps
        for _ in range(cfg.side_length):
            steps.append(do_action(forward_id, frame_idx))
            frame_idx += 1
            if cfg.place_every and frame_idx % cfg.place_every == 0:
                steps.append(do_action(place_id, frame_idx))
                frame_idx += 1
        # We "walk forward" in 4 different cardinal directions by changing the meaning
        # of `forward`: we cycle through directions instead of actually rotating the
        # yaw (the simple model doesn't tie yaw to movement direction).
        # Use action 1 (+y), 4 (+x), 2 (-y), 3 (-x) successively.
        forward_id = [1, 4, 2, 3][(side + 1) % 4]
        # Optional yaw rotation purely to make the eval sensitive to yaw consistency
        for _ in range(cfg.yaw_turns_per_corner):
            steps.append(do_action(yaw_id, frame_idx))
            frame_idx += 1

    return steps
