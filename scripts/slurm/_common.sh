#!/usr/bin/env bash
# Sourced by other slurm scripts. NOT submitted directly.
#
# Sets up conda + uv on a Beery vision cluster compute node, prepares the
# project's Python deps. Idempotent — safe to source twice.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Conda (matches the user's standard sbatch pattern) ------------------
source /data/vision/beery/scratch/serena/.bashrc
conda activate bpp

# --- uv (autoresearch's project manager) ---------------------------------
# uv handles autoresearch's pyproject.toml + uv.lock. It lives alongside the
# conda env without conflict — it just manages a `.venv` for project deps.
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv into \$HOME/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "[setup] uv version: $(uv --version)"

# --- Sync project deps (cached after first run) --------------------------
if [ ! -d ".venv" ]; then
    echo "[setup] running 'uv sync' (first-time setup, ~3 min) ..."
    uv sync
else
    echo "[setup] .venv already present — skipping uv sync"
fi

# --- Sanity report -------------------------------------------------------
echo "[setup] hostname: $(hostname)"
echo "[setup] python:   $(which python)  -> $(python --version)"
echo "[setup] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    # nvidia-smi exits non-zero on CPU-only jobs ("No devices were found").
    # The `|| true` keeps `set -e` from killing the script — we only want
    # the GPU report as informational output.
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
