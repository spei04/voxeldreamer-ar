# Interactive agent-driven workflow

If your cluster blocks outbound HTTPS from compute nodes (very common), you
can't run the Claude Code / Codex agent inside an `sbatch` job. Use this
two-tier workflow instead:

## Tier 1: agent on the login node

The login node usually *does* have outbound HTTPS (so you can `git clone`,
`pip install`, etc.). Run the agent there interactively:

```bash
ssh you@cluster
cd voxeldreamer-ar
claude   # or: codex
```

Inside the agent session, prompt:

> Read program.md. Each iteration, modify train.py with one experimental
> change, then submit train_once.sh via sbatch and wait for it to finish
> with `scancel`-aware polling. Read val_bpb from the resulting
> results.tsv line and decide keep/discard.

The agent stays on the login node and submits per-experiment `sbatch` jobs.
Each job is ~5–10 minutes of compute. Throughput: ~6 experiments/hour
limited by Slurm queue time + start-up.

## Tier 2: per-experiment job runner

`train_once.sh` is the per-experiment runner. The agent submits it like:

```bash
sbatch --wait scripts/slurm/train_once.sh
# --wait blocks until the job finishes, returning its exit code.
```

After `sbatch --wait` returns, the agent reads the latest line from
`results.tsv` and the tail of `scripts/slurm/logs/vd-train-*.out` to decide
whether to keep or discard the experiment.

## Why this works when overnight.sh doesn't

- Outbound HTTPS lives on the login node, not the GPU node.
- GPU node only needs to run `train.py` — no model-provider API calls.
- `sbatch --wait` is the clean handoff: the agent thinks of each
  experiment as a synchronous "run train.py" function call.

## Comparison with the all-in-one `autoresearch_overnight.sh`

|  | overnight.sh (agent inside job) | interactive (agent on login) |
|---|---|---|
| Requires outbound HTTPS from GPU | Yes | No |
| GPU utilization | High (loops in-process) | Lower (Slurm queue overhead) |
| Crash isolation | Whole job dies | Per-experiment isolation |
| Best for | Cloud H100 boxes (Lambda, RunPod, Modal) | Traditional HPC clusters |

Default recommendation: try `autoresearch_overnight.sh` first. If outbound
HTTPS is blocked, fall back to this interactive workflow.
