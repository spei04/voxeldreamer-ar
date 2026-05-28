# Data pipeline design — VoxelDreamer-AR

## 1. The fundamental constraint: token budget

Autoresearch's default `MAX_SEQ_LEN = 2048`. This is *very small* for a video world model. Token budget per frame determines how many frames fit in context, which directly determines what "long horizon" can possibly mean for our experiments.

### Naive voxel encoding (does not fit)

A Minecraft chunk is 16×16×384 ≈ 100k voxels. One token per voxel is hopeless. Even an 8×8×8 = 512-voxel egocentric window per frame leaves room for only ~4 frames in 2048 context.

### Patch tokenization — the MineWorld trick, adapted to 3D

MineWorld compresses each 224×384 frame into 14×24 = 336 image patches. We do the analogous thing in 3D: each **voxel patch** of size `P × P × P` is represented by **one token** drawn from a learned VQ-VAE codebook over voxel patches.

For an 8×8×8 egocentric voxel window with `P = 2`, we get `(8/2)^3 = 64` patch tokens per frame. With 6 header tokens, that's 70 tokens per frame — **~29 frames per 2048-token row**.

For `P = 4`: 8 patch tokens per frame + 6 header = 14 tokens per frame — **~145 frames per row**.

### Delta encoding — the Minecraft trick

Minecraft is voxel-sparse: most voxels don't change frame-to-frame. Instead of re-tokenizing the whole window each frame, **emit only changed-patch tokens** with their position. Empirically (we will measure this in Phase 2) <5% of patches change per frame for typical agent trajectories. A 60-second trajectory at 1 Hz might fit in ~1500 tokens.

This is also the strongest version of the thesis: 3D structure is most exploitable when the world is locally static and changes are sparse.

### Recommended Phase 2 default

| Parameter | Value | Rationale |
|---|---|---|
| Egocentric window | 16×16×16 voxels | Agent's view, ~3 voxel meters reach |
| Patch size `P` | 4 | One token per 64 voxels — coarse but tractable |
| Patches per frame | 64 | (16/4)^3 |
| Per-frame layout overhead | ~10 tokens | frame_sep, camera, action, etc. |
| Tokens/frame (full) | 74 | full re-emit |
| Tokens/frame (delta, est.) | 5–15 | only changed patches + position markers |
| Frames per 2048 ctx | ~25 full-frame, ~150 delta |
| Frames per 8192 ctx (if we grow context) | ~100 full-frame, ~600 delta |

**A 60s loop-closure trajectory at 1 Hz = 60 frames fits comfortably in 2048 context with delta encoding, or in 8192 context without.**

## 2. Choice of data source: VPT vs MineRL vs custom

| | VPT | MineRL | Custom (MineRL / Malmo programmatic) |
|---|---|---|---|
| Hours of data | 70k (after IDM filter) | ~few hundred | Whatever we generate (probably <100h) |
| Action labels | Inferred via IDM | Human-recorded | Programmatic — perfect |
| Camera pose | Inferred from frames | Game-state available | Game-state — perfect |
| **Voxel ground truth** | **None** | **Available via API extensions** | **Perfect** |
| Engineering cost | High (must run IDM) | Low (gym API) | Medium (write rollout scripts) |
| Suitability for our thesis | Weak (no voxel GT) | Medium | **Strongest** |

**Recommendation**: Build **custom** Phase 2 data using MineRL's gym + a custom plugin that logs `(frame, action, camera_pose, voxel_chunk)` quadruples. The dataset will be smaller than VPT but every example carries the ground-truth voxel state we need both for direct supervision and for the loop-closure eval. We can mix in VPT later if scale becomes the bottleneck — but VPT's lack of voxel GT means it cannot be used for our primary eval metric, only for the BPB metric.

## 3. Shard format

Shards live at `~/.cache/autoresearch/voxeldreamer/` to mirror autoresearch's convention. Format: **one Parquet file per shard**, columns:

| column | type | description |
|---|---|---|
| `clip_id` | int64 | Unique clip identifier |
| `tokens` | list[int32] | The full token stream for this clip |
| `positions_t` | list[int32] | t (frame index) per token |
| `positions_x` | list[int16] | x voxel coord per token (-1 = non-voxel) |
| `positions_y` | list[int16] | y voxel coord per token |
| `positions_z` | list[int16] | z voxel coord per token |
| `num_frames` | int32 | Number of frames in this clip (sanity check) |

One clip per row; one shard = many clips. Parquet because (a) `prepare.py` already uses Parquet, (b) it handles variable-length list columns well, (c) supports streaming reads.

### Why carry positions in the shard rather than recompute them

We could compute (t, x, y, z) from token offsets if we used a strictly canonical per-frame layout. But:
- Delta encoding makes layout variable per frame (different number of changed patches).
- Action / camera tokens vary per frame.
- Computing position from offset would require parsing the token stream every step.

Carrying positions explicitly costs ~3× shard size but is straightforward, debuggable, and lets the dataloader stream them as-is.

## 4. Dataloader interface

Autoresearch's `make_dataloader` returns `(inputs, targets, epoch)`. We extend to **`(inputs, targets, positions_in, positions_tgt, epoch)`** where:
- `inputs`, `targets`: `[B, T]` long tensors (unchanged from autoresearch).
- `positions_in`, `positions_tgt`: `[B, T, 4]` long tensors with (t, x, y, z) per token.

`train.py` then passes `positions_in` into the model's RoPE table generation:

```python
cos, sin = precompute_rope_3d(positions_in.view(-1, 4), config)
cos = cos.view(B, T, 1, head_dim // 2)
sin = sin.view(B, T, 1, head_dim // 2)
```

Note: this *does* mean modifying `prepare.py` (autoresearch's "read-only" file). Documented divergence — see `RESEARCH.md` Phase 2.

## 5. Synthetic data generator for unit testing

`voxeldreamer/data/synthetic.py` emits a fake trajectory that does not require Minecraft. Used purely to test the loader and the position-aware training plumbing without depending on game data. Generates `(frame, action, camera_pose, voxel_chunk)` quadruples using simple procedural rules — e.g., a flat world with one moving block per step. This lets us validate the full data → loader → model → loss path on CPU before any real data is collected.

## 6. Phase 2 execution order

1. **W1**: implement `synthetic.py` + writer + reader + test data pipeline end-to-end on CPU. No Minecraft yet.
2. **W2**: stand up a MineRL-based rollout script that records `(frame, action, camera_pose, voxel_chunk)`. Generate ~10 hours of varied agent trajectories with intentional loop-closures.
3. **W3**: implement the patch VQ-VAE (or learned patch dictionary) for voxel chunks, train on the W2 data. Output: a voxel-patch tokenizer that produces ~64-token-per-frame representations.
4. **W4**: connect to autoresearch — modify `prepare.py` for Minecraft shards, modify `train.py` for 3D-RoPE and positions. Run B1 (1D RoPE) baseline. Then sweep 3D-RoPE variants overnight.
