#!/bin/bash
#SBATCH --job-name=vd-smoke
#SBATCH --output=scripts/slurm/logs/%x-%j.out
#SBATCH --error=scripts/slurm/logs/%x-%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
# ----- CLUSTER-SPECIFIC ---------------------------------------------------
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ACCOUNT
# --------------------------------------------------------------------------
#
# CPU-only smoke test: runs the 42 voxeldreamer tests + the end-to-end
# smoke_train.py. Useful as a first job to confirm the environment is OK
# before requesting a GPU. ~2 minutes.

source scripts/slurm/_common.sh

echo "[smoke] running unit tests ..."
uv run python voxeldreamer/tests/run_all.py

echo "[smoke] running end-to-end smoke_train.py ..."
uv run python voxeldreamer/smoke_train.py

echo "[smoke] done at $(date -Iseconds)"
