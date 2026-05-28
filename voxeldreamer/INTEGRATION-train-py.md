# Integration patch: autoresearch `train.py` → VoxelDreamer-AR

*The exact changes needed to wire 3D-RoPE and position-aware data into the
upstream autoresearch training script. Cross-reference this with the
working CPU integration in `voxeldreamer/smoke_train.py`.*

The diff is small — ~20 lines in `train.py` plus a contract change in
`prepare.py`. Apply this when starting Phase 2 W1.

## 1. `prepare.py` — extend the dataloader return tuple

Today: `make_dataloader` yields `(inputs, targets, epoch)`.

Required: `make_dataloader` yields `(inputs, targets, positions_in, positions_tgt, epoch)`.

**Implementation**: replace the text-shard reading + tokenization pipeline
with `voxeldreamer.data.loader.make_position_aware_dataloader`. The 100%-
utilization, BOS-aligned, best-fit packing is already mirrored in the new
loader. `evaluate_bpb` continues to work because it counts per-token cross-
entropy in nats (vocab-size independent).

Add an import:

```python
from voxeldreamer.data.loader import make_position_aware_dataloader as _make_loader
```

Add a small wrapper to maintain the `make_dataloader(tokenizer, B, T, split)`
signature; pass the relevant shard paths internally.

## 2. `train.py` — replace 1D RoPE precompute with 3D

Today (lines ~183–193):

```python
def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
    if device is None:
        device = self.transformer.wte.weight.device
    channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
    inv_freq = 1.0 / (base ** (channel_range / head_dim))
    t = torch.arange(seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    cos, sin = freqs.cos(), freqs.sin()
    cos, sin = cos.bfloat16(), sin.bfloat16()
    cos, sin = cos[None, :, None, :], sin[None, :, None, :]
    return cos, sin
```

Replace with a call out to our module:

```python
from voxeldreamer.positional.rope_3d import precompute_rope_3d, RoPE3DConfig

# Initialization (in GPT.__init__):
self.rope_cfg = RoPE3DConfig(head_dim=head_dim)
# Don't precompute here — positions are per-batch in 3D-RoPE.
```

In the forward pass, replace the cached `self.cos[:, :T]` lookup with a
per-batch precompute. Today (around line 271):

```python
cos_sin = self.cos[:, :T], self.sin[:, :T]
```

Becomes (positions arrive as a new argument to `forward`):

```python
B, T = idx.shape
cos, sin = precompute_rope_3d(positions.view(-1, 4), self.rope_cfg)
cos = cos.view(B, T, 1, -1)
sin = sin.view(B, T, 1, -1)
cos_sin = (cos, sin)
```

`apply_rotary_emb` itself doesn't change — see the verbatim copy in
`voxeldreamer/smoke_train.py:apply_rotary_emb`.

## 3. `train.py` — thread `positions` through

The model's forward signature changes from `forward(idx, targets=None)` to
`forward(idx, positions, targets=None)`. The training step that calls it
goes from:

```python
inputs, targets, epoch = next(train_loader)
loss, _ = model(inputs, targets=targets)
```

to:

```python
inputs, targets, positions_in, positions_tgt, epoch = next(train_loader)
loss, _ = model(inputs, positions_in, targets=targets)
```

Same change for `evaluate_bpb` — pass through `positions_in`.

## 4. `train.py` — add `evaluate_loop_closure` invocation

At the end of training, alongside the existing `val_bpb` emission, run the
loop-closure eval over a held-out closed-loop set. The held-out shard path
should be a config constant (default: `~/.cache/autoresearch/voxeldreamer/eval_loop/`).

```python
from voxeldreamer.eval.loop_closure import loop_closure_drift

def model_rollout_fn(prefix_tokens, prefix_positions, num_to_generate):
    # autoregressive sampling — temperature 0 (argmax) for determinism
    ...

drifts = []
for steps in held_out_loop_trajectories:
    result = loop_closure_drift(steps, encoder_cfg, model_rollout_fn, burn_in_frames=1)
    drifts.append(result.drift)
loop_drift = float(np.mean(drifts))
print(f"loop_drift:       {loop_drift:.6f}")
```

The summary block emits a new `loop_drift:` line alongside `val_bpb:`.

## 5. Attention masks (Tier-2, optional)

If the agent reaches Tier-2 sweeps and wants to ablate 3D-local or axial
masks, swap the FA3 attention call for `F.scaled_dot_product_attention`
in the affected layers:

```python
from voxeldreamer.attention.mask_3d import causal_3d_local_mask

mask_2d = causal_3d_local_mask(positions[0], radius=8.0, metric="linf")
# Reshape for SDPA: [B, H, T, T] additive bias
mask = mask_2d.unsqueeze(0).unsqueeze(0)
out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
```

FA3's `window_size` parameter handles 1D causal local attention efficiently
but cannot express arbitrary 3D masks; SDPA accepts the explicit mask at a
~2-3× speed cost. Acceptable for ablations.

## 6. What does NOT change

- `Tokenizer` class in `prepare.py` — VoxelVocab from `voxeldreamer/tokenizer/voxel_vocab.py`
  is structurally similar; we just replace the BPE-text tokenizer with the
  voxel/action/camera vocab. Same `get_bos_token_id()` / `encode()` /
  `decode()` interface.
- The training loop's outer structure (5-min budget, optimizer, LR schedule).
- The branch + commit + results.tsv workflow.
- `evaluate_bpb` — works as-is on any token vocabulary.

## 7. Smoke test on CPU first

Before pushing to H100, run:

```bash
python3 voxeldreamer/smoke_train.py
```

This exercises the full pipeline on CPU with a tiny GPT, synthetic
trajectories, 10 training steps, and a final eval. If smoke_train.py fails
on the migration, fix it there first — debugging is far easier than on a
remote GPU box.

## Estimated effort

- `prepare.py` swap: ~2 hours.
- `train.py` RoPE migration: ~1 hour (small diff, but careful with the
  `cos_sin` tuple plumbing through every block).
- `evaluate_loop_closure` integration: ~2 hours including held-out shard
  generation.
- End-to-end first run on H100: ~1 hour debugging the inevitable shape
  mismatches.

Total: ~one focused day on the GPU. Phase 2 W1 is realistic in a week with
buffer for the inevitable surprises.
