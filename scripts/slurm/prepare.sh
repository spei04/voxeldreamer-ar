#!/bin/bash
#SBATCH --job-name=vd-prepare
#SBATCH --output=scripts/slurm/logs/%x-%j.out
#SBATCH --error=scripts/slurm/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# ----- CLUSTER-SPECIFIC: uncomment + edit ---------------------------------
# #SBATCH --partition=cpu
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --qos=normal
# --------------------------------------------------------------------------
#
# One-time data preparation for upstream autoresearch.
# Downloads ~10 training shards + the validation shard, trains the BPE
# tokenizer. Cached at ~/.cache/autoresearch/.
#
# Run once before train.sh. Re-run only if you want different shard counts.

source scripts/slurm/_common.sh

echo "[prepare] starting at $(date -Iseconds)"
uv run prepare.py
echo "[prepare] done at $(date -Iseconds)"
