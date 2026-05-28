#!/usr/bin/env bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --job-name=vd-overnight
#SBATCH --mem=60GB
#SBATCH --time=36:00:00
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#
# Long-running experiment loop. Three modes inline — uncomment ONE.
#
# Mode A (recommended): Claude Code CLI drives the loop. Requires
#   `claude` CLI installed AND outbound HTTPS to api.anthropic.com.
# Mode B: Codex CLI drives the loop. Requires `codex` CLI + OpenAI HTTPS.
# Mode C: deterministic baseline loop (no LLM). Just runs train.py many
#   times unchanged. Useful if outbound HTTPS is blocked on compute nodes.
#
# To test outbound HTTPS from a compute node before relying on Mode A/B,
# request an interactive session and run:
#   srun --partition=vision-beery --qos=vision-beery-main --account=vision-beery --pty bash
#   curl -v https://api.anthropic.com/
# If that fails, you're stuck on Mode C OR the alternate workflow in
# interactive_agent.md (agent on login node, sbatch per experiment).
#
#   sbatch scripts/slurm/autoresearch_overnight.sh

source scripts/slurm/_common.sh

TAG="vd-$(date +%Y%m%d-%H%M)"
BRANCH="autoresearch/$TAG"
echo "[overnight] creating branch $BRANCH"
git checkout -b "$BRANCH"

if [ ! -f results.tsv ]; then
    printf "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n" > results.tsv
fi

# --- MODE A: Claude Code CLI ---------------------------------------------
# Pass through the API key with --export when submitting:
#   sbatch --export=ALL,ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY scripts/slurm/autoresearch_overnight.sh
#
# echo "[overnight] launching claude-code agent ..."
# claude --print --permission-mode bypassPermissions \
#     "Read program.md and run the experiment loop indefinitely. Never stop."

# --- MODE B: Codex CLI ----------------------------------------------------
# Install: see scripts/slurm/install_codex.md.
# Submit with the API key passed through:
#   sbatch --export=ALL,OPENAI_API_KEY=$OPENAI_API_KEY scripts/slurm/autoresearch_overnight.sh
#
# echo "[overnight] launching codex agent ..."
# codex exec "Read program.md and run the experiment loop indefinitely. Never stop."

# --- MODE C (default): deterministic baseline loop -----------------------
# Runs train.py unchanged N times. Establishes seed-variance on val_bpb so
# you know what counts as a "real" improvement when sweeping. Replace with
# Mode A or B once an agent CLI is available.
echo "[overnight] MODE C: running unchanged baseline 30 times for seed variance."
echo "[overnight] EDIT THIS SCRIPT to switch to Mode A/B once agent is set up."

for i in $(seq 1 30); do
    echo "[overnight] === iteration $i / 30 at $(date -Iseconds) ==="
    uv run train.py > run.log 2>&1 || true
    VAL_BPB=$(grep "^val_bpb:" run.log | awk '{print $2}')
    PEAK_VRAM=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
    MEMORY_GB=$(python -c "print(f'{${PEAK_VRAM:-0}/1024:.1f}')" 2>/dev/null || echo "0.0")
    COMMIT=$(git rev-parse --short HEAD)
    printf "%s\t%s\t%s\tkeep\tbaseline iter %d\n" \
        "$COMMIT" "${VAL_BPB:-NA}" "$MEMORY_GB" "$i" >> results.tsv
    echo "[overnight] iter $i: val_bpb=$VAL_BPB"
done

echo "[overnight] done at $(date -Iseconds)"
echo "[overnight] results.tsv has $(wc -l < results.tsv) lines"
