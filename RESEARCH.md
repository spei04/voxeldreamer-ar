# VoxelDreamer-AR

**A tokenized autoregressive world model for Minecraft with 3D-structured tokenization.**

## Thesis

Video world models drift over long horizons because their internal representations are 2D and frame-statistical — they have no explicit notion of persistent world state. Frontier video models (Sora, Veo, Genie 3) lose coherence after 30–60 seconds: objects morph, scenes don't loop-close, physics breaks.

We propose **VoxelDreamer-AR**: an autoregressive token-prediction world model for Minecraft where the token stream is **3D-structured by construction**. Frames, actions, camera pose, and voxel state are unified into a single sequence; tokens carry native 3D-grid positional information; attention respects 3D adjacency; the KV-cache functions as persistent world state.

**Hypothesis**: 3D-aware tokenization yields both (1) lower `val_bpb` AND (2) order-of-magnitude better loop-closure consistency on 60–120s trajectories, vs naive 1D positional encodings, at matched single-GPU compute.

## Why this is a real research contribution

- **Real open problem**: even frontier world models drift; nobody has cleanly ablated whether the drift comes from 2D latents vs limited compute.
- **Ground truth available**: Minecraft has *literal* voxel ground truth — we can supervise the 3D structure directly and measure loop closure objectively.
- **Tractable at single-GPU scale**: the contribution is the *method* (3D tokenization + PE), demonstrated on a constrained domain. Not a frontier-quality artifact.
- **Lineage**: extends VPT (action-conditioned Minecraft tokens) and Genie 2 (AR video world model) by adding 3D structural priors that those approaches lack.

## Experimental program (axes for autoresearch sweeps)

The autoresearch agent should sweep over axes that **isolate the 3D-structure contribution**, not generic transformer tweaks. Priority order:

### Tier 1 — Tokenization & positional encoding (the core thesis)
1. **3D positional encodings**: 3D-RoPE (per-axis rotary), 3D-sinusoidal, learned 3D embeddings, vs 1D-RoPE baseline.
2. **Token interleaving order**: voxel-major, raster-major, Hilbert-curve, depth-first vs breadth-first frame ordering. Does locality-preserving interleaving help?
3. **Voxel-token vocabulary design**: per-block-type tokens, learned VQ over voxel patches, hierarchical (block-type × orientation).
4. **Camera-pose tokenization**: continuous embeddings vs discretized tokens vs Plücker-coordinate encoding.

### Tier 2 — Attention structure
5. **3D-local attention masks**: full attention vs 3D-neighborhood windowed attention vs sparse 3D-axial attention.
6. **KV-cache as world state**: cache compression schemes that preserve 3D-spatial coverage (keep one token per voxel cell vs uniform decay).
7. **Cross-frame attention patterns**: which frames does each frame attend to? Sliding window vs landmark frames vs learned retrieval.

### Tier 3 — Loss & training
8. **Auxiliary voxel-prediction loss**: supervise the latent toward ground-truth voxel state at intermediate layers.
9. **Loop-closure penalty during training**: synthetic loops in training data with consistency loss.
10. **Mix of next-frame vs next-voxel prediction objectives**.

### Out of scope (do not sweep)
- Generic optimizer tweaks (Muon/AdamW comparisons) — already well-studied, dilutes the contribution.
- Activation function choices (GeLU/SwiGLU/etc.) — orthogonal to the thesis.
- Width/depth scaling beyond what's needed for ablation matching.

## Primary evaluation

| Metric | What it measures | Source |
|---|---|---|
| `val_bpb` | Bits-per-byte on held-out Minecraft token trajectories | Autoresearch-native, computed by `prepare.py`'s eval harness |
| **Loop-closure drift** | Agent walks a 60s path and returns to start; pixel/voxel disagreement between rendered start vs returned-start | Custom — built in Phase 2 |
| Voxel-state reconstruction | Does the model's predicted voxel grid match ground truth? | Custom — direct supervision check |
| Counterfactual fidelity *(secondary)* | Intervene on action at step t; does future change correctly? | Custom — ablation eval |

## Baselines

- **B0**: autoresearch's stock GPT on text (Phase 1 — validates the loop works on this hardware).
- **B1**: Same model architecture, Minecraft tokens, naive 1D RoPE. The "no 3D structure" baseline.
- **B2**: 2D positional encoding (frames × spatial-1D). The "video-aware but not 3D" baseline.
- **DIAMOND / GameNGen** *(for context, not direct head-to-head)*: published video world models. We claim improvement on loop closure at matched FLOPs; we do not claim FVD parity.

## Roadmap

### Phase 1 — Validate the autoresearch loop *(this week)*
- Get `uv sync && uv run prepare.py && uv run train.py` working on the target GPU.
- Confirm baseline `val_bpb` matches reference, autoresearch loop executes one happy-path experiment.
- No code changes; just verify infra.

### Phase 2 — Minecraft data pipeline *(weeks 2–3)*
- Build `voxeldreamer/` data pipeline. Two options:
  - **Option A**: Use VPT contractor demonstrations (public dataset of human Minecraft play with action labels). Frames + actions, no voxel ground truth — derive voxels via a frame-to-voxel reconstruction.
  - **Option B**: Drive a Minecraft instance programmatically (MineRL / Project Malmo), log raw `(frame, action, camera_pose, voxel_chunk)` tuples. More work, but gives clean voxel ground truth.
- Build the tokenizer: voxel-token vocab, action-token vocab, camera-pose tokens. Serialize trajectories to autoresearch's data-shard format.
- Modify `prepare.py` to load Minecraft shards instead of text. (This *does* mean modifying `prepare.py` — diverging from upstream autoresearch's "read-only" rule. Documented; intentional.)
- Add `evaluate_loop_closure()` alongside `evaluate_bpb()`.

### Phase 3 — Unleash autoresearch *(weeks 4+)*
- Customize `program.md` with the Tier 1/2/3 sweep priorities above.
- Launch the overnight loop.
- Daily morning review: read `results.tsv`, summarize wins, refine `program.md` based on what the agent is/isn't finding.

## Open questions to resolve before Phase 2

- VPT vs MineRL for data — voxel-supervision quality vs setup cost.
- Token budget per second of video — drives whether 60s loops fit in context.
- Whether to start with a fixed small Minecraft world (one biome, one seed) or varied.

## References to read

- VPT (Baker et al., 2022) — action-labeled Minecraft data at scale
- Genie 1/2 (DeepMind, 2024/2025) — AR video world models
- DIAMOND (Alonso et al., 2024) — diffusion-based game world models
- GameNGen (Valevski et al., 2024) — diffusion Doom world model
- 3D-RoPE / RoPE extensions to multiple axes (recent literature)
- World Labs technical reports on 3D-consistent generation
