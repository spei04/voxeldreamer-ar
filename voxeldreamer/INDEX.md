# VoxelDreamer-AR — file index

Quick navigation for collaborators and future-you.

## Documents (start here)

- [`../RESEARCH.md`](../RESEARCH.md) — thesis, experimental program, evals, 3-phase roadmap.
- [`../RESEARCH-LITERATURE.md`](../RESEARCH-LITERATURE.md) — prior work survey positioning this contribution.
- [`PHASE2.md`](PHASE2.md) — week-by-week execution plan for Phase 2 (Minecraft data + train.py integration).
- [`INTEGRATION-train-py.md`](INTEGRATION-train-py.md) — exact patch needed to migrate upstream `train.py` to VoxelDreamer-AR (3D-RoPE, position-aware data, loop-closure eval).
- [`program-voxeldreamer.md`](program-voxeldreamer.md) — autoresearch-agent instruction file. **Use only after Phase 2 is complete.**
- [`data/README.md`](data/README.md) — token-budget math, data source trade-offs, shard format.

## Code modules (all CPU-runnable)

### `positional/`
- [`rope_3d.py`](positional/rope_3d.py) — 4-axis (t,x,y,z) RoPE producing drop-in (cos, sin) tables.
- [`sinusoidal_3d.py`](positional/sinusoidal_3d.py) — additive 3D sinusoidal PE for ablations.

### `tokenizer/`
- [`voxel_vocab.py`](tokenizer/voxel_vocab.py) — unified vocab: blocks + actions + yaw/pitch + specials.
- [`orderings.py`](tokenizer/orderings.py) — raster, voxel-major, Morton, Hilbert (Skilling), camera-frustum orderings.

### `attention/`
- [`mask_3d.py`](attention/mask_3d.py) — 3D-local (L_inf / L_2) and axial causal masks.

### `data/`
- [`synthetic.py`](data/synthetic.py) — fake Minecraft-style trajectory generator (no game required).
- [`synthetic_loop.py`](data/synthetic_loop.py) — closed-loop square-walk generator.
- [`encoder.py`](data/encoder.py) — TrajectoryStep → Clip with per-frame layout and positions.
- [`shard_format.py`](data/shard_format.py) — Parquet shard writer/reader.
- [`loader.py`](data/loader.py) — position-aware dataloader extending autoresearch's contract.

### `eval/`
- [`loop_closure.py`](eval/loop_closure.py) — drift = 1 − voxel_IoU(predicted_final, gt_final). Crown-jewel eval.
- [`counterfactual.py`](eval/counterfactual.py) — paired-trajectory intervention test.
- [`voxel_recon.py`](eval/voxel_recon.py) — token-level voxel accuracy for aux-loss ablations.

### `tests/`
- [`run_all.py`](tests/run_all.py) — runs every `test_*.py` as a subprocess. **`python3 voxeldreamer/tests/run_all.py`**
- 42 tests passing across 6 test files.

### End-to-end CPU smoke test
- [`smoke_train.py`](smoke_train.py) — tiny GPT + 3D-RoPE + synthetic data + 10 training steps + eval. Run with **`python3 voxeldreamer/smoke_train.py`**. Proves the full integration shape works before any H100 time is spent.

## Phase status (updated 2026-05-27)

- [x] Phase 0 — research direction narrowed, repo scaffolded, lit reviewed.
- [x] Phase 0.5 — code modules, data pipeline, eval harness all in place + tested on CPU.
- [ ] Phase 1 — validate upstream autoresearch on text (one H100, one `uv run train.py`).
- [ ] Phase 2 — see `PHASE2.md`.
- [ ] Phase 3 — overnight autoresearch loop on Minecraft.
