# Literature review — VoxelDreamer-AR

*Synthesis of prior work that positions VoxelDreamer-AR's contribution. Citations verified May 2026.*

## TL;DR for the contribution

There are several existing AR-token video world models for Minecraft (MineWorld) and other domains (Genie 1/2/3, GAIA, VPT). **All of them tokenize pixels** via VQ-VAE / latent patches, and use **1D or 2D positional encodings**. None of them exploit the fact that Minecraft is *literally a voxel world with a known 3D ground-truth state*. The drift / consistency problem is universally acknowledged and is currently being solved through **scale and emergent memory** (Genie 3 calls this emergent and unprogrammed).

**Our bet**: Explicit 3D-structured tokenization + 3D-RoPE makes long-horizon consistency *geometric* rather than emergent. We expect this to deliver an order-of-magnitude improvement in loop-closure drift at fixed compute, demonstrated at a single-GPU scale where MineWorld already shows competitive baselines.

---

## 1. AR-token video world models (direct competitors)

### MineWorld (Microsoft, April 2025) — *the closest prior work*

The reference baseline. **arXiv:2504.08388**.

- **Architecture**: LLaMA decoder + RoPE. Same architecture family as `autoresearch/train.py`.
- **Tokenization**: VQ-VAE compresses 224×384 frames into 14×24 = **336 image patches per frame**, plus 70 action tokens. Vocab = 8262.
- **Context**: 5376 tokens ≈ 16 frames.
- **Sizes**: 300M / 700M / 1.2B params. 3–6 FPS on single GPU.
- **Training data**: 10M VPT video clips, 160M frames.
- **Acknowledged limitation** (direct quote): *"temporal consistency within this range [is not guaranteed] when the distance between game states exceeds this range."* — i.e., 16 frames bounds everything.

**Why this matters for us**: MineWorld is autoresearch-shaped (same architecture, same scale, same single-GPU envelope). We can compare directly. Replacing 14×24 VQ-VAE patches with native voxel tokens + 3D-RoPE is a clean ablation.

### Genie 1 / 2 / 3 (DeepMind, 2024 / 2024 / Aug 2025)

Frontier-scale AR foundation world models.

- **Genie 3** (Aug 2025): Real-time 24 FPS, 720p, several minutes of consistency. *"Visual memory extending as far back as one minute ago."*
- AR over previously generated frames + user actions. Frame-by-frame generation.
- **DeepMind's own framing**: physical consistency *"emerges"* over time *"because the model can remember what it previously generated — a capability that DeepMind says its researchers didn't explicitly program into the model."*

**Why this matters for us**: Genie 3's consistency is explicitly described as *emergent and unprogrammed*. Our thesis is the dual: **engineered, not emergent**. A small model with explicit 3D structure should achieve consistency that a much larger model gets only through emergent behavior.

### Oasis (Decart / Etched, Oct 2024) — *single-GPU Minecraft, diffusion*

- ViT-VAE encoder/decoder + diffusion transformer with axial spatiotemporal attention.
- Trained on millions of hours of Minecraft.
- **Single H100, 20 FPS, 360p**. Proves Minecraft world models are tractable at single-GPU inference scale.
- Diffusion-based, so not directly comparable on val_bpb — but a useful "what's the platform actually capable of" reference.

### GameNGen (Google, Aug 2024) — *Doom, diffusion*

- arXiv:2408.14837.
- RL agent plays Doom, recordings train a diffusion next-frame model.
- 20 FPS on single TPU. *"Remains stable over extended multi-minute play sessions."*
- Human raters distinguish only slightly better than chance after 5 minutes of generation.
- **Key insight relevant to us**: GameNGen achieves multi-minute stability through *agent-aligned training distribution*, not architectural priors. Suggests data curriculum matters as much as architecture.

### DIAMOND (Alonso et al., 2024) — *Atari, diffusion*

- Diffusion next-observation model trained with RL agent in the loop.
- Pixel-space diffusion, not latent.
- Atari-scale evaluation.
- Useful as a "smaller scale diffusion world model" reference but not directly comparable for Minecraft.

### GAIA-1 / GAIA-2 (Wayve, 2023 / 2024)

- Driving world models.
- AR token-based, large-scale (~9B GAIA-1, larger for GAIA-2).
- Strong on action-conditioning fidelity.
- 2D-positional, not 3D.
- Useful for tokenization design lessons (separate vocabs for video / action / text).

### VPT (Baker et al., OpenAI, 2022)

- arXiv:2206.11795.
- **Inverse dynamics model trained on contractor data**, then used to *pseudo-label* 70k hours of web Minecraft footage.
- Not a world model — a *behavior policy*. But the dataset (270k hours raw → 70k clean → IDM-labeled) is the de facto standard Minecraft pretraining corpus.
- **Critical for us**: VPT data is *frames + inferred actions* — **no voxel ground truth**. To get voxels we either (a) reconstruct from frames (lossy), or (b) collect new data with MineRL/Malmo.

---

## 2. Positional encodings for spatiotemporal data

### RoPE foundations (Su et al., 2021)

- Rotary positional embedding. Standard in modern LLMs (LLaMA, Qwen, Mistral, GPT-4-class).
- Acts on Q and K in attention. `apply_rotary_emb(x, cos, sin)`.
- 1D — each token has one position index.

### RoPE-3D in Qwen2-VL

- Splits head_dim into 3 subsets, each rotated by independent (t, x, y) frequencies.
- **Direct precedent for what we want.** Qwen2-VL processes video as a 3D-positional sequence.
- Key design choice: **assign lower frequencies to the temporal axis** so long-video periodic patterns don't confuse attention.

### VideoRoPE (Zheng et al., 2024)

- ICLR 2025 (and earlier preprint).
- Identifies that naive 3D-RoPE causes spurious attention spikes due to periodic frequency alignment across axes.
- Proposes Diagonal Layout Allocation (DLA) and Adjustable Temporal Spacing.
- Lesson: **frequency budget across axes is the key hyperparameter** for 3D-RoPE.

### VRoPE (Feb 2025, arXiv:2502.11664)

- Adds continuity/symmetry transforms over (t, x, y, text) indices.
- Targets video-LLM cross-modal attention smoothness.
- Less relevant for pure world-model setting but worth tracking.

### GeoPE — quaternion-based isotropic 3D rotations

- Constructs rotations as quaternionic sandwich products with SO(3) phase averaging.
- More expressive but exotic — probably wrong fit for a single-GPU ablation paper.

**Implication for VoxelDreamer-AR**: We have 4 axes to encode: **(t, x, y, z)** where t = frame index and (x,y,z) = voxel position within the world. The literature consistently says: (a) assign low frequencies to temporal axis, (b) tune the frequency budget per axis, (c) start from Qwen2-VL–style axis-wise RoPE before reaching for GeoPE-style isotropic schemes.

---

## 3. Locality-preserving sequence orderings

A 3D voxel grid must be serialized into a 1D token stream for AR generation. **The order matters** — it determines what "previous tokens" the model attends to.

### Hilbert curves vs Morton (Z-order)

- **Hilbert curves** preserve locality better — adjacent voxels in 3D tend to be adjacent in the 1D sequence.
- **Morton (Z-order)** is faster to compute (bit interleaving) but can have unbounded dilation — two adjacent 3D voxels may end up far apart in the 1D order.
- Hilbert curves are used in **Point Transformer v3** to define local neighborhoods.

### Recent work on Hilbert ordering in transformers

- **HilbertA (arXiv:2509.26538, Sep 2025)**: Hilbert attention for diffusion image generation. Shows measurable gains over raster order.
- **Neighbor-Aware Token Reduction via Hilbert Curve (arXiv:2512.22760, Dec 2025)**: token pruning with Hilbert ordering.
- **arXiv:2309.15199**: Generalized 3D Morton and Hilbert orderings.

**Implication for VoxelDreamer-AR**: Voxel-token interleaving is a first-class experimental axis. Candidates to sweep: raster (x→y→z), Morton, Hilbert, voxel-major (z fastest), camera-frustum-ordered (closest-to-camera first).

---

## 4. The drift problem — current consensus

What every paper acknowledges and nobody has solved at small scale:

| Model | Stable for | Failure mode |
|---|---|---|
| Genie 3 | ~minutes | Emergent and unexplained beyond ~1 min |
| Oasis | ~minutes (with caveats) | Block hallucinations on re-visit |
| MineWorld | **~16 frames** (hard bound from context) | Outside context: amnesia |
| GameNGen | 5+ minutes | Doom is small + RL-aligned data distribution |
| DIAMOND | Atari-scale | OOD beyond training trajectories |

**The pattern**: stability comes from one of (a) brute scale + emergent memory (Genie 3), (b) RL-aligned narrow data distribution (GameNGen), or (c) explicit context window that physically bounds the problem (MineWorld). **Nobody has tried explicit 3D structural priors as the source of consistency.**

---

## 5. Data sources for Minecraft world models

### VPT (270k → 70k hours)

- Pros: huge, public, action-labeled (via IDM).
- Cons: no voxel ground truth, no camera-pose ground truth (must be inferred), frame quality varies (web scrape).

### MineRL (60M state-action tuples)

- Pros: clean state-action pairs from human demos, public.
- Cons: smaller (60M vs VPT's 160M frames after MineWorld processing), 64×64 frames are low-res, includes game-state features but voxel access depends on MineRL API extensions.

### Custom MineRL / Project Malmo data collection

- **Highest scientific value**: we drive the game programmatically and log `(frame, action, camera_pose, voxel_chunk)` quadruples with perfect ground truth.
- Pros: clean voxel state, controllable trajectory generation, can intentionally include loop-closure trajectories in training.
- Cons: more engineering, smaller dataset (probably <1M frames for a single-GPU project).

### MiDaS (Torpey 2024)

- A 2024 Minecraft dataset focused on non-natural scenes.
- Worth checking for our use case but probably wrong domain.

**Recommendation for Phase 2**: Start with custom MineRL data collection. The dataset is smaller but every training example has perfect voxel + camera ground truth, which we need for the core eval (loop closure) anyway. We can mix in VPT for scale later if needed.

---

## 6. Where the genuine open question lives

Combining everything above:

1. AR-token Minecraft world models exist (MineWorld) and are competitive at single-GPU scale.
2. They tokenize pixels (VQ-VAE patches) — none use native voxel tokens.
3. They use 1D positional encoding — none use 3D-RoPE structured around (t, x, y, z).
4. Their consistency limit is set by context window (MineWorld: 16 frames) or emergent scale (Genie 3: minutes via implicit memory).
5. Nobody has demonstrated that **explicit 3D priors** can substitute for **scale-driven emergent memory** in long-horizon consistency.

**VoxelDreamer-AR's experimental program tests exactly that substitution.**

The contribution is publishable if either:
- **Strong result**: 3D-tokenized + 3D-RoPE world model matches MineWorld's `val_bpb` *and* dramatically improves loop closure — argues structural priors substitute for scale.
- **Negative result**: 3D structure does *not* help — argues that consistency really does require scale / emergence, with clean ablations to back the claim.

Both outcomes are paper-worthy. The trap to avoid: a mushy result where 3D structure marginally helps in some metrics, doesn't in others. The eval design needs to make the claim crisp — which is why **loop-closure drift on 60–120s trajectories with ground-truth voxels** is the centerpiece eval.
