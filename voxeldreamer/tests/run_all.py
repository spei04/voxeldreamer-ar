"""Run every voxeldreamer/tests/test_*.py file as a subprocess. CPU-only."""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    test_files = sorted(HERE.glob("test_*.py"))
    failures = []
    for tf in test_files:
        print(f"\n=== {tf.name} ===")
        r = subprocess.run(
            [sys.executable, str(tf)],
            capture_output=True,
            text=True,
        )
        print(r.stdout, end="")
        if r.returncode != 0:
            failures.append((tf.name, r.stderr.strip() or "non-zero exit"))
            print(r.stderr, end="")
    print()
    if failures:
        print(f"{len(failures)} test file(s) failed:")
        for name, msg in failures:
            print(f"  {name}: {msg[:300]}")
        sys.exit(1)
    print("All test files passed.")


if __name__ == "__main__":
    main()
