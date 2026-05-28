# voxeldreamer/

VoxelDreamer-AR–specific code lives here, kept separate from upstream autoresearch files so the fork relationship stays clean.

## Planned contents

- `data/` — Minecraft trajectory collection scripts (VPT loader or MineRL driver) and shard writers compatible with `prepare.py`'s dataloader.
- `tokenizer/` — voxel-token vocab, action-token vocab, camera-pose tokenizer. Interleaving strategies (raster, Hilbert, voxel-major).
- `positional/` — 3D-RoPE, 3D-sinusoidal, learned 3D embeddings as drop-in replacements for `train.py`'s positional encoding.
- `attention/` — 3D-local attention masks, sparse 3D-axial attention.
- `eval/` — `evaluate_loop_closure()`, `evaluate_voxel_reconstruction()`, counterfactual evals. Designed to plug into `prepare.py`'s eval harness alongside `evaluate_bpb`.
- `sweeps/` — `program-voxeldreamer.md` overrides + saved `results.tsv` per branch.

## Status

Empty by design. Phase 1 of [../RESEARCH.md](../RESEARCH.md) validates upstream autoresearch on text first. Phase 2 populates this directory.

## Note on `prepare.py`

Upstream autoresearch treats `prepare.py` as read-only. VoxelDreamer-AR *will* need to modify it (to load Minecraft shards instead of text). This is an intentional, documented divergence — see RESEARCH.md Phase 2.
