#!/bin/bash
#SBATCH --job-name=vd-train
#SBATCH --output=scripts/slurm/logs/%x-%j.out
#SBATCH --error=scripts/slurm/logs/%x-%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
# ----- CLUSTER-SPECIFIC ---------------------------------------------------
# #SBATCH --partition=gpu
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --constraint=h100        # or a100, l40s, etc.
# --------------------------------------------------------------------------
#
# Single training experiment. Runs train.py once with whatever is committed
# right now. ~5 minutes training + ~1 minute startup/compilation.
#
# Logs everything to scripts/slurm/logs/<jobname>-<jobid>.out — the
# autoresearch summary block (val_bpb, peak_vram_mb, etc.) is at the end.
#
# Use this for: Phase 1 baseline validation, single ad-hoc experiments,
# B0/B1 baseline runs at the start of Phase 2 W4.

source scripts/slurm/_common.sh

COMMIT=$(git rev-parse --short HEAD)
echo "[train] starting at $(date -Iseconds)"
echo "[train] commit: $COMMIT"
echo "[train] branch: $(git rev-parse --abbrev-ref HEAD)"

# run.log is what upstream's program.md expects to grep for val_bpb.
uv run train.py > run.log 2>&1
RET=$?

echo "[train] exit code: $RET"
echo "[train] === run.log tail ==="
tail -n 25 run.log
echo "[train] ====================="

if [ $RET -ne 0 ]; then
    echo "[train] FAILED. Full stderr in scripts/slurm/logs/%x-%j.err"
    exit $RET
fi

# Extract metrics for results.tsv (autoresearch convention)
VAL_BPB=$(grep "^val_bpb:" run.log | awk '{print $2}')
PEAK_VRAM=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
MEMORY_GB=$(python3 -c "print(f'{${PEAK_VRAM:-0}/1024:.1f}')" 2>/dev/null || echo "0.0")

echo "[train] val_bpb=$VAL_BPB  memory_gb=$MEMORY_GB"

# Append to results.tsv if it exists (autoresearch convention — not git-tracked)
if [ -f results.tsv ]; then
    DESC="${VD_DESC:-single sbatch run}"
    printf "%s\t%s\t%s\tkeep\t%s\n" "$COMMIT" "$VAL_BPB" "$MEMORY_GB" "$DESC" >> results.tsv
    echo "[train] appended to results.tsv"
fi

echo "[train] done at $(date -Iseconds)"
