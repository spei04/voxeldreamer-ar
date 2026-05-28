#!/bin/bash
#SBATCH --job-name=vd-overnight
#SBATCH --output=scripts/slurm/logs/%x-%j.out
#SBATCH --error=scripts/slurm/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
# ----- CLUSTER-SPECIFIC ---------------------------------------------------
# #SBATCH --partition=gpu
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --constraint=h100
# --------------------------------------------------------------------------
#
# Overnight autoresearch loop driven by an AI agent CLI (Claude Code or Codex).
#
# REQUIREMENTS for this script to actually drive a research loop:
#   1. The agent CLI must be installed on the cluster. Options:
#        - Claude Code:  https://docs.claude.com/en/docs/claude-code
#                        npm install -g @anthropic-ai/claude-code
#        - Codex CLI:    https://github.com/openai/codex
#   2. Outbound HTTPS to the model provider must be allowed from compute
#      nodes. Many HPC clusters BLOCK this — check first by running
#      `curl -v https://api.anthropic.com/` from an interactive session.
#   3. An API key in the environment (ANTHROPIC_API_KEY or OPENAI_API_KEY).
#      Pass through sbatch with `--export=ALL,ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY`.
#
# If your cluster blocks outbound HTTPS, see `interactive_agent.md` for the
# alternate workflow: run the agent on a login node and submit per-experiment
# `train_once.sh` jobs from there.

source scripts/slurm/_common.sh

# Branch convention from program.md: each overnight session gets its own branch.
TAG="vd-$(date +%Y%m%d-%H%M)"
BRANCH="autoresearch/$TAG"
echo "[overnight] creating branch $BRANCH"
git checkout -b "$BRANCH"

# Initialize results.tsv if absent (autoresearch convention — not git-tracked).
if [ ! -f results.tsv ]; then
    printf "commit\tval_bpb\tmemory_gb\tstatus\tdescription\n" > results.tsv
fi

# --- Pick which agent CLI to drive the loop ------------------------------
# Uncomment the block that matches what you installed.

# OPTION A: Claude Code CLI
# echo "[overnight] launching claude-code agent ..."
# claude --print --permission-mode bypassPermissions \
#     "Read program.md and run the experiment loop indefinitely." \
#     2>&1 | tee scripts/slurm/logs/agent-$SLURM_JOB_ID.log

# OPTION B: Codex CLI
# echo "[overnight] launching codex agent ..."
# codex exec "Read program.md and run the experiment loop indefinitely." \
#     2>&1 | tee scripts/slurm/logs/agent-$SLURM_JOB_ID.log

# OPTION C: fallback — no agent installed, just run the baseline N times so
# the job doesn't waste GPU. Replace with one of the above as soon as the
# agent CLI is available on the cluster.
echo "[overnight] no agent configured — running baseline 100 times as a sanity check."
echo "[overnight] EDIT THIS SCRIPT to enable the agent CLI."
for i in $(seq 1 100); do
    echo "[overnight] iteration $i / 100 at $(date -Iseconds)"
    uv run train.py > run.log 2>&1 || true
    VAL_BPB=$(grep "^val_bpb:" run.log | awk '{print $2}')
    PEAK_VRAM=$(grep "^peak_vram_mb:" run.log | awk '{print $2}')
    MEMORY_GB=$(python3 -c "print(f'{${PEAK_VRAM:-0}/1024:.1f}')" 2>/dev/null || echo "0.0")
    COMMIT=$(git rev-parse --short HEAD)
    printf "%s\t%s\t%s\tkeep\tbaseline iter %d\n" "$COMMIT" "$VAL_BPB" "$MEMORY_GB" "$i" >> results.tsv
done

echo "[overnight] done at $(date -Iseconds)"
echo "[overnight] results.tsv has $(wc -l < results.tsv) lines"
