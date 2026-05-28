# Phase 2 task breakdown — Minecraft data + train.py integration

*Read after RESEARCH.md. This is the concrete week-by-week execution plan
for the "build the Minecraft data pipeline + wire 3D-RoPE into train.py"
phase. Phase 1 (validate upstream autoresearch on text) is assumed complete.*

## Status at the start of Phase 2

Already in place (committed to this repo):

- `voxeldreamer/positional/{rope_3d,sinusoidal_3d}.py` — drop-in PE modules.
- `voxeldreamer/tokenizer/{orderings,voxel_vocab}.py` — voxel tokenization.
- `voxeldreamer/attention/mask_3d.py` — 3D-local and axial causal masks.
- `voxeldreamer/data/{synthetic,synthetic_loop,encoder,shard_format,loader}.py`
  — CPU-testable data pipeline working on synthetic data.
- `voxeldreamer/eval/{loop_closure,counterfactual,voxel_recon}.py` — eval
  harness with oracle and random-baseline sanity tests.
- 42 CPU tests passing.

What's missing: **real Minecraft data** and **train.py integration**.

## Week 1 — End-to-end on synthetic data, in autoresearch's loop

Goal: prove the full pipeline (data → loader → 3D-RoPE'd model → loss → eval)
runs end-to-end on the GPU, using *synthetic* trajectories. No Minecraft yet.

Tasks:
1. **Generate a synthetic shard.** Write a small `voxeldreamer/data/generate_synthetic_shards.py`
   that emits ~10k clips of `square_loop()` trajectories with varying
   `side_length` and `seed`. Total tokens should land at ~few hundred million
   so val_bpb is comparable to upstream autoresearch's 5-min training budget.
2. **Modify `prepare.py`.** Replace the text-dataset loader with the
   position-aware loader from `voxeldreamer/data/loader.py`. The
   `evaluate_bpb` function continues to work — bytes-per-token still makes
   sense for a token-level model.
3. **Modify `train.py`.**
   - Import `precompute_rope_3d`. Replace `_precompute_rotary_embeddings`
     with a call that uses `positions_in` from the loader.
   - Pipe `positions_in` through to the rotary call site at every layer.
   - Add `evaluate_loop_closure()` invocation at the end of training,
     emitting a `loop_drift:` line in the summary.
4. **Run B0 baseline**: 1D RoPE with `positions_in` set to (frame_idx, -1, -1, -1)
   — should reproduce upstream val_bpb behavior. **Sanity check**.
5. **Run B1 baseline**: switch on full 3D RoPE with the default axis-dim split.
   This is the first real VoxelDreamer-AR experiment.

Acceptance criteria for end of W1:
- Both B0 and B1 complete a 5-minute training run on H100.
- `loop_drift` value emits in the summary.
- Synthetic-data `loop_drift` for B1 < random-baseline (~0.95) — even untrained
  models should beat random when the world is mostly static.

### W1 risks

- **Position-conditioning slows training**: `precompute_rope_3d` runs per-batch
  rather than once-at-init. Mitigation: cache by `positions_in` content hash;
  for a fixed dataloader the same position patterns repeat.
- **Flash-Attention-3 + custom masks**: train.py uses FA3 which doesn't accept
  arbitrary attention masks. Mitigation: for Tier 2 mask experiments, fall
  back to `F.scaled_dot_product_attention` and accept the speed penalty for
  ablation runs.

## Week 2 — Real Minecraft data collection

Goal: collect ~10 hours of varied Minecraft trajectories with ground-truth
voxel state and intentional loop closures.

Tasks:
1. **Set up MineRL** (or Project Malmo if MineRL doesn't expose voxel state
   easily). Two known-working approaches:
   - **MineRL 1.0+** with a custom Forge mod that logs voxel chunks alongside
     observations. Higher upfront cost; clean state.
   - **Malmo (Minecraft 1.11)** has direct world-state access via XML mission
     specs. Lower upfront cost but older Minecraft version.
2. **Write a rollout script** (`voxeldreamer/data/collect_minecraft.py`):
   - Loads MineRL/Malmo env.
   - Runs a scripted-or-random agent.
   - At each tick, logs `(frame_rgb, action, camera_pose, agent_xyz, voxel_window)`.
   - Encodes via `encode_trajectory` into `Clip` objects.
   - Writes Parquet shards.
3. **Diversify trajectories**:
   - Random walks across multiple biomes / seeds.
   - **Loop trajectories**: square walks, oct walks, "go and come back" patterns
     — these are the held-out eval set for `loop_closure_drift`.
   - **Building trajectories**: agent places blocks, then revisits.
   - **Mining trajectories**: agent breaks blocks, then revisits.
4. **Sanity-check the data**: visualize a few clips, confirm voxel windows
   correspond to what the agent sees.
5. **Split train/val/eval-loop**:
   - 80% train shards
   - 10% val shards (for val_bpb)
   - 10% **closed-loop eval set** — disjoint from train, used for loop_drift.

Acceptance criteria for end of W2:
- ~10 hours of trajectory data on disk, ~1B tokens.
- Eval split contains ≥1000 closed-loop trajectories.
- A few hand-inspected clips look correct.

### W2 risks

- **MineRL Forge-mod engineering**: writing a Minecraft mod is the single
  hardest part of the project. Mitigation: start with Malmo for v0, even if
  the Minecraft version is older. Voxel state >> graphics fidelity for our
  thesis.
- **Voxel window choice**: 16×16×16 may be wrong. Watch agent's behavior
  early — if it consistently interacts with voxels outside the window, grow
  the window (and the token budget).

## Week 3 — Patch tokenizer for voxel chunks

Goal: replace the placeholder "majority block id per patch" tokenizer in
`encoder.py` with a learned voxel-patch VQ-VAE.

Tasks:
1. **Define the patch tokenizer**: 3D conv VQ-VAE with codebook size ~4096,
   patch size 4×4×4, latent dim 16. Input: voxel chunk. Output: codebook id
   per patch.
2. **Train it on the W2 data**: standard VQ-VAE loss + commitment loss. Should
   take <1 hour on H100 for ~100M voxel patches.
3. **Replace `_patch_tokenize`** in `voxeldreamer/data/encoder.py` with a call
   to the learned tokenizer.
4. **Re-emit all shards** with the learned tokens.
5. **Verify reconstruction quality**: held-out voxel chunks should round-trip
   with ≥95% per-voxel accuracy.

Acceptance criteria for end of W3:
- Learned tokenizer with ≥95% voxel reconstruction accuracy on held-out chunks.
- All Phase 2 shards re-emitted.

### W3 risks

- **VQ-VAE codebook collapse**: classic problem. Mitigation: use EMA codebook
  updates and a small commitment cost.
- **Tokenizer overfits to training distribution**: validate reconstruction on
  unseen biomes.

## Week 4 — Launch the autoresearch loop

Goal: replace `program.md` with `voxeldreamer/program-voxeldreamer.md` and
unleash the agent for overnight Tier-1 sweeps.

Tasks:
1. **Copy `program-voxeldreamer.md` to `program.md`** (or symlink).
2. **Run B0 + B1 + B2** as warm-up baselines with the new data and new tokenizer:
   - B0: 1D RoPE, unchanged train.py shape.
   - B1: 3D RoPE, default axis-dim split.
   - B2: 3D RoPE + voxel_major_zyx ordering instead of raster.
3. **Hand off to the autoresearch agent**: prompt it to read program.md and
   start the loop.
4. **Morning review** (next day, every day):
   - Read `results.tsv`. Did anything beat B1?
   - Sanity-check the top 3 experiments: re-run them to confirm reproducibility.
   - Update `program-voxeldreamer.md`'s Tier priorities based on what's working.

Acceptance criteria for end of W4:
- Overnight loop runs ~50–100 experiments without crashing.
- At least one experiment beats B1 on `loop_drift` while not regressing on `val_bpb`.
- Daily morning review process is sustainable.

## Stretch: Week 5+

If the Tier-1 sweep yields a clear winner:
- Run Tier-2 (attention structure) ablations.
- Run Tier-3 (aux loss) ablations.
- Scale: increase model from 50M → 200M params if H100 budget allows.
- Write the paper.

If the Tier-1 sweep is null (no clear winner over B0):
- Re-examine: is the synthetic data too easy? Move to harder Minecraft seeds.
- Re-examine: is the voxel window too small? The model can't reason about
  what's outside it.
- Consider the **negative result paper**: "3D-structured tokenization does
  not improve long-horizon consistency at single-GPU scale". This is also
  publishable if the ablations are clean.

## Known unknowns

- Does Minecraft's voxel sparsity actually help compression? The delta-encoding
  argument assumes <5% patches change per frame; this is unmeasured.
- Does FA3's window_size parameter compose with 3D-RoPE? Untested. May need
  to drop to SDPA for some ablations.
- VQ-VAE quality vs codebook size trade-off. 4096 may be too small for varied biomes.
