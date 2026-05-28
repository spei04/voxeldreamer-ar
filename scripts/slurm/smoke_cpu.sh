#!/usr/bin/env bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --job-name=vd-smoke
#SBATCH --mem=16GB
#SBATCH --time=00:30:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --cpus-per-task=4
#
# CPU-only smoke test: runs the 42 voxeldreamer unit tests + the end-to-end
# smoke_train.py. Useful as a first job to confirm the environment is OK
# before requesting a GPU. ~2 minutes of actual work.
#
#   sbatch scripts/slurm/smoke_cpu.sh

source scripts/slurm/_common.sh

echo "[smoke] running unit tests ..."
uv run python voxeldreamer/tests/run_all.py

echo "[smoke] running end-to-end smoke_train.py ..."
uv run python voxeldreamer/smoke_train.py

echo "[smoke] done at $(date -Iseconds)"
