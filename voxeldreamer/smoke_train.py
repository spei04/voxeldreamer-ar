"""
CPU-only end-to-end smoke test: synthetic Minecraft data -> 3D-RoPE-conditioned
tiny GPT -> a few training steps -> loop-closure eval.

This is NOT a real training run — it's a proof that the data + RoPE + model +
eval plumbing all connect on CPU before any of it touches an H100. Run it
after `python3 voxeldreamer/tests/run_all.py` passes; if `smoke_train.py`
runs cleanly, the integration shape is sound.

Usage:
    python3 voxeldreamer/smoke_train.py

Output:
    Prints per-step training loss and a final loop-closure drift number.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voxeldreamer.data.encoder import EncoderConfig, encode_trajectory
from voxeldreamer.data.loader import make_position_aware_dataloader
from voxeldreamer.data.shard_format import write_shard
from voxeldreamer.data.synthetic import generate_clip
from voxeldreamer.data.synthetic_loop import SquareLoopConfig, square_loop
from voxeldreamer.eval.loop_closure import loop_closure_drift, make_oracle_rollout
from voxeldreamer.positional.rope_3d import RoPE3DConfig, precompute_rope_3d
from voxeldreamer.tokenizer.voxel_vocab import VoxelVocab


# --- Tiny GPT with 3D-RoPE wired in --------------------------------------


def apply_rotary_emb(x, cos, sin):
    """Verbatim from autoresearch train.py."""
    assert x.ndim == 4
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


class TinyAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.head_dim)
        k = k.view(B, T, self.n_head, self.head_dim)
        v = v.view(B, T, self.n_head, self.head_dim)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)
        # Standard SDPA, causal.
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class TinyBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = TinyAttention(n_embd, n_head)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln1(x), cos, sin)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, n_layer: int = 2, n_embd: int = 64, n_head: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.blocks = nn.ModuleList([TinyBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)
        self.head_dim = n_embd // n_head
        self.rope_cfg = RoPE3DConfig(head_dim=self.head_dim)

    def forward(self, tokens, positions):
        B, T = tokens.shape
        # precompute_rope_3d takes [N, 4] and returns [1, N, 1, head_dim/2].
        # We pass B*T positions then reshape the tables back to per-batch.
        cos, sin = precompute_rope_3d(positions.view(-1, 4), self.rope_cfg, dtype=torch.float32)
        cos = cos.view(B, T, 1, -1)  # [B, T, 1, head_dim/2]
        sin = sin.view(B, T, 1, -1)
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.ln_f(x)
        return self.head(x)


# --- Smoke training loop -------------------------------------------------


def main():
    vocab = VoxelVocab()
    print(f"vocab_size = {vocab.vocab_size}")

    # Build a tiny synthetic shard
    clips = []
    for cid in range(8):
        steps = generate_clip(num_frames=4, seed=cid)
        clips.append(encode_trajectory(steps, EncoderConfig(patch_size=4), clip_id=cid))

    with tempfile.TemporaryDirectory() as td:
        shard = Path(td) / "shard_0.parquet"
        write_shard(shard, clips)

        loader = make_position_aware_dataloader(
            [shard],
            batch_size=2,
            seq_len=64,
            bos_token=vocab.special_token("<bos>"),
            device="cpu",
        )

        model = TinyGPT(vocab_size=vocab.vocab_size, n_layer=2, n_embd=64, n_head=4)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        print("\n--- training (CPU, 10 steps) ---")
        for step in range(10):
            inputs, targets, pos_in, pos_tgt, epoch = next(loader)
            logits = model(inputs, pos_in)
            loss = F.cross_entropy(logits.reshape(-1, vocab.vocab_size), targets.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(f"  step {step:2d}  loss={loss.item():.4f}")

        # Final sanity: loop-closure on a small closed-loop trajectory, using
        # an oracle rollout (we are not evaluating model quality, just plumbing).
        steps_loop = square_loop(SquareLoopConfig(side_length=2))
        clip_loop = encode_trajectory(steps_loop, EncoderConfig(patch_size=4))
        oracle = make_oracle_rollout(clip_loop.tokens)
        result = loop_closure_drift(
            steps_loop,
            EncoderConfig(patch_size=4),
            oracle,
            burn_in_frames=1,
        )
        print(f"\n--- eval ---")
        print(f"loop_drift (oracle): {result.drift:.4f}  (expected 0.0)")
        print(f"voxel_iou (oracle):  {result.iou:.4f}  (expected 1.0)")
        assert result.drift == 0.0, "smoke eval failed: oracle rollout should give zero drift"

    print("\nsmoke_train: OK")


if __name__ == "__main__":
    main()
