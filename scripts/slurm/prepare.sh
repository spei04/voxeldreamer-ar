#!/usr/bin/env bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --job-name=vd-prepare
#SBATCH --mem=60GB
#SBATCH --time=02:00:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --cpus-per-task=16
#
# One-time data preparation for upstream autoresearch.
# Downloads ~10 training shards + the validation shard, trains the BPE
# tokenizer. Cached at ~/.cache/autoresearch/.
#
# No GPU needed for this step. Run once before any training.
#
#   sbatch scripts/slurm/prepare.sh

source scripts/slurm/_common.sh

echo "[prepare] starting at $(date -Iseconds)"
uv run prepare.py
echo "[prepare] done at $(date -Iseconds)"
