# program-voxeldreamer.md

**VoxelDreamer-AR experimental program. Point your autoresearch agent here.**

This file replaces the upstream `program.md` *only after Phase 2 of the
roadmap is complete* — i.e., after `prepare.py` has been swapped to emit
Minecraft voxel-token shards and `train.py` has been wired up to consume
3D positions for RoPE. Until then, use upstream `program.md` for Phase 1
baseline validation.

If you are the autoresearch agent reading this: **read `../RESEARCH.md` and
`../RESEARCH-LITERATURE.md` before doing anything**. They explain the thesis,
the experimental program, and the prior work this contribution is positioning
against. Do not propose experiments that drift away from the 3D-structure
thesis — every sweep should isolate at least one axis from Tier 1/2/3 below.

## Setup

1. **Agree on a run tag**. Propose `vd-<date>-<short-slug>` e.g. `vd-jun3-rope3d`.
   Branch `voxeldreamer/<tag>` must not exist yet — this is a fresh run.
2. **Create the branch**: `git checkout -b voxeldreamer/<tag>` from `master`.
3. **Read the in-scope files**, in this order:
   - `../RESEARCH.md` — thesis + experimental program.
   - `../RESEARCH-LITERATURE.md` — what's been done, where the open question lives.
   - `voxeldreamer/data/README.md` — token budget, shard format, data pipeline.
   - `prepare.py` — Minecraft-shard data prep (post-Phase-2 version).
   - `train.py` — model + RoPE + training loop. **This is the only file you edit.**
   - `voxeldreamer/positional/rope_3d.py`, `voxeldreamer/attention/mask_3d.py` —
     drop-in modules to wire into train.py.
4. **Verify data exists**: check `~/.cache/autoresearch/voxeldreamer/` for shards.
   If absent, tell the human to run `uv run prepare.py` (post-Phase-2).
5. **Initialize results.tsv** with the standard header.
6. **Confirm and go.**

## Experimentation

Same loop as upstream `program.md`:

LOOP FOREVER:
1. Look at current branch / commit.
2. Pick one Tier-1/2/3 experimental axis (see below). Modify `train.py`.
3. `git commit` the change.
4. Run: `uv run train.py > run.log 2>&1` (~5 min wall-clock).
5. Extract `val_bpb` and the new `loop_closure_drift` value from `run.log`.
6. Record in `results.tsv` (additional column: `loop_drift`).
7. If `val_bpb` improved OR `loop_drift` improved meaningfully without
   `val_bpb` regression beyond 0.002, **keep**. Else **reset**.
8. **Never stop.** Continue until the human interrupts.

### What CAN you do

- Modify `train.py` (only).
- Choose any positional encoding from `voxeldreamer/positional/`.
- Choose any attention mask from `voxeldreamer/attention/`.
- Adjust hyperparameters needed to make a Tier-1/2/3 experiment work.

### What CANNOT you do

- Modify `prepare.py` (post-Phase-2 dataloader contract is frozen).
- Modify the `evaluate_bpb` or `evaluate_loop_closure` functions.
- Install new packages.
- Change the time budget.

## Experimental priorities — sweep these, in order

### Tier 1: tokenization & positional encoding (highest priority)

These directly test the 3D-structure thesis. Spend most of your experiments here.

| Axis | What to vary | Hypothesis |
|---|---|---|
| Positional encoding | 1D RoPE → 3D RoPE → 3D RoPE with axis-dim reallocation | 3D wins meaningfully on `loop_drift` |
| Per-axis frequency base | (t=100k, x=y=z=10k) → sweep t-base ∈ {1e4, 1e5, 1e6} | Lower temporal frequency improves long-horizon |
| Voxel ordering | raster → voxel_major_zyx → morton → hilbert | Locality-preserving orderings improve `val_bpb` |
| Camera-pose token | discretized → continuous-embed → Plücker | Continuous embedding matches diffusion baselines |

### Tier 2: attention structure

Run only after at least 3 Tier-1 experiments have established a baseline-beating config.

| Axis | What to vary | Hypothesis |
|---|---|---|
| 3D-local mask | full → L_inf radius {4, 8, 16, 32} | Local mask roughly matches full at lower compute |
| Axial mask | full → axial | Axial mask is a strong inductive bias for 3D structure |
| KV-cache compression | uniform decay → keep-one-per-voxel | Spatial-aware cache extends effective horizon |
| Cross-frame attention | full → sliding window → landmark frames | Landmark frames win on loop closure |

### Tier 3: loss & training (lowest priority, run last)

| Axis | What to vary | Hypothesis |
|---|---|---|
| Aux voxel-prediction loss | none → 0.1 × CE on voxel positions | Forces latent toward GT voxel state |
| Loop-closure penalty | none → 0.05 × IoU loss on synthetic loops | Direct supervision of the target metric |

### Anti-patterns — do NOT sweep these

These are *orthogonal* to our contribution. Sweeping them dilutes the research story.

- Optimizer choice (Muon vs AdamW vs Lion). Whatever upstream picked, stay there.
- Activation function (GeLU vs SwiGLU vs ReLU). Stay with upstream default.
- Generic width/depth scaling beyond what's needed for ablation matching.
- Normalization variants (RMSNorm vs LayerNorm). Stay with upstream default.
- Tokenizer vocab size beyond what `voxeldreamer/tokenizer/voxel_vocab.py` exposes.

If you find yourself reaching for these, **stop** and re-read RESEARCH.md
section "Experimental program — out of scope".

## Output format

The training script prints both the upstream summary and an additional
`loop_drift` line:

```
---
val_bpb:          0.987300
loop_drift:       0.342000      # 1 - voxel_IoU on held-out closed-loop set
training_seconds: 300.1
total_seconds:    330.5
peak_vram_mb:     46100.2
mfu_percent:      37.10
total_tokens_M:   480.3
num_steps:        912
num_params_M:     53.7
depth:            8
```

Extract both with:

```bash
grep -E "^val_bpb:|^loop_drift:|^peak_vram_mb:" run.log
```

## results.tsv schema

Tab-separated, six columns:

```
commit  val_bpb  loop_drift  memory_gb  status  description
```

Status: `keep`, `discard`, or `crash`. Description: short text on what
this experiment changed (e.g., `3d-rope axis_dims=(16,6,6,4)`).

## Stopping criterion

You don't stop on your own. The human will interrupt. If you genuinely
exhaust ideas, **think harder**: re-read RESEARCH-LITERATURE.md for cited
prior work, look at the worst near-misses in `results.tsv`, try combining
two near-misses, try a *radical* Tier-1 reformulation (e.g., remove RoPE
entirely and learn 4D position embeddings from scratch).
