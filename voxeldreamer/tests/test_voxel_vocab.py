"""CPU tests for voxeldreamer.tokenizer.voxel_vocab."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from voxeldreamer.tokenizer.voxel_vocab import (
    DEFAULT_ACTIONS,
    DEFAULT_BLOCK_TYPES,
    VoxelVocab,
)


def test_vocab_size_matches_components():
    v = VoxelVocab()
    expected = (
        len(DEFAULT_BLOCK_TYPES)
        + len(DEFAULT_ACTIONS)
        + v.yaw_buckets
        + v.pitch_buckets
        + len(v.specials)
    )
    assert v.vocab_size == expected, (v.vocab_size, expected)


def test_block_token_round_trip():
    v = VoxelVocab()
    for name in DEFAULT_BLOCK_TYPES:
        tok = v.block_token(name)
        assert v.kind_of(tok) == "block"


def test_action_token_kind():
    v = VoxelVocab()
    tok = v.action_token("jump")
    assert v.kind_of(tok) == "action"


def test_yaw_buckets_cover_circle():
    v = VoxelVocab()
    tokens = {v.yaw_token(d) for d in range(0, 360)}
    assert len(tokens) == v.yaw_buckets


def test_pitch_clamping():
    v = VoxelVocab()
    # Out-of-range values should still produce valid pitch tokens.
    t_low = v.pitch_token(-180.0)
    t_high = v.pitch_token(180.0)
    assert v.kind_of(t_low) == "pitch"
    assert v.kind_of(t_high) == "pitch"


def test_specials_in_special_region():
    v = VoxelVocab()
    for name in v.specials:
        tok = v.special_token(name)
        assert v.kind_of(tok) == "special"


def test_encode_frame_shape():
    v = VoxelVocab()
    voxels = [0, 1, 2, 3, 4]  # 5 voxel indices
    tokens = v.encode_frame(voxels, camera_yaw=45.0, camera_pitch=10.0, action="forward")
    # 6 header tokens + 5 voxel tokens
    assert len(tokens) == 11
    assert v.kind_of(tokens[-1]) == "block"
    assert v.kind_of(tokens[0]) == "special"
    assert v.kind_of(tokens[2]) == "yaw"
    assert v.kind_of(tokens[3]) == "pitch"


def run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  OK  {t.__name__}")
    print(f"\n{len(tests)} vocab tests passed.")


if __name__ == "__main__":
    run_all()
