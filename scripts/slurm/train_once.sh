#!/usr/bin/env bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --job-name=vd-train
#SBATCH --mem=60GB
#SBATCH --time=01:00:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#
# Single autoresearch training experiment. Runs train.py once with whatever
# is committed right now. ~5 minutes of training + ~1 minute startup.
#
# A100 NOTE: upstream autoresearch was tuned for H100. On 40GB A100 you may
# need to lower DEPTH in train.py (8 -> 6 or 4) to avoid OOM. On 80GB A100
# the defaults should fit.
#
#   sbatch scripts/slurm/train_once.sh
#   # with a description appended to results.tsv:
#   sbatch --export=ALL,VD_DESC="baseline" scripts/slurm/train_once.sh

source scripts/slurm/_common.sh

COMMIT=$(git rev-parse --short HEAD)
echo "[train] starting at $(date -Iseconds)"
echo "[train] commit: $COMMIT"
echo "[train] branch: $(git rev-parse --abbrev-ref HEAD)"

uv run train.py > run.log 2>&1
RET=$?

echo "[train] exit code: $RET"
echo "[train] === run.log tail ==="
tail -n 25 run.log
echo "[train] ====================="

if [ $RET -ne 0 ]; then
    echo "[train] FAILED — see run.log for the Python traceback."
    exit $RET
fi

VAL_BPB=$(grep "^val_bpb:" run.log | awk '{print $2}')
PEAK_VRAM=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
MEMORY_GB=$(python -c "print(f'{${PEAK_VRAM:-0}/1024:.1f}')" 2>/dev/null || echo "0.0")

echo "[train] val_bpb=$VAL_BPB  memory_gb=$MEMORY_GB"

if [ -f results.tsv ]; then
    DESC="${VD_DESC:-single sbatch run}"
    printf "%s\t%s\t%s\tkeep\t%s\n" "$COMMIT" "$VAL_BPB" "$MEMORY_GB" "$DESC" >> results.tsv
    echo "[train] appended to results.tsv"
fi

echo "[train] done at $(date -Iseconds)"
