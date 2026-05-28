#!/bin/bash
# Sourced by other slurm scripts. NOT submitted directly.
#
# Sets up `uv` and the project's Python env on a compute node.
# Idempotent — safe to source twice. Logs to stdout.

set -euo pipefail

# --- Working directory ---------------------------------------------------
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# --- Modules (CLUSTER-SPECIFIC — edit for your site) --------------------
# Most Lmod-based clusters need at least Python and CUDA loaded explicitly.
# Examples:
#   module load python/3.10 cuda/12.4
#   module load anaconda3/2024.06
# If your cluster doesn't use modules, leave this section empty.
if command -v module >/dev/null 2>&1; then
    : # module load python/3.10 cuda/12.4
fi

# --- uv (project manager used by upstream autoresearch) ------------------
if ! command -v uv >/dev/null 2>&1; then
    echo "[setup] installing uv into \$HOME/.local/bin ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
echo "[setup] uv version: $(uv --version)"

# --- Sync project deps (cached after first run) --------------------------
if [ ! -d ".venv" ]; then
    echo "[setup] running 'uv sync' ..."
    uv sync
else
    echo "[setup] .venv already present — skipping uv sync"
fi

# --- Sanity: report node + GPU ------------------------------------------
echo "[setup] hostname: $(hostname)"
echo "[setup] CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi
