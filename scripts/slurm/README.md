# Slurm scripts for VoxelDreamer-AR

`sbatch`-ready scripts for submitting jobs to a Slurm-managed GPU cluster.

## Before first use — cluster-specific edits

Open each `*.sh` script and uncomment / edit the lines under
`# ----- CLUSTER-SPECIFIC --------` near the top. The common ones:

```bash
#SBATCH --partition=gpu          # name your cluster uses (gpu, h100, a100, dgx, ...)
#SBATCH --account=YOUR_ACCOUNT   # mandatory on many clusters
#SBATCH --qos=normal             # some clusters require this
#SBATCH --constraint=h100        # GPU type if you want to pin one
```

Also edit `_common.sh` to enable the right `module load` lines for your
site (Python and CUDA versions).

## Scripts (in the order you'd run them)

| Script | Purpose | Resources | Run when |
|---|---|---|---|
| `smoke_cpu.sh` | Run our 42 unit tests + smoke_train.py on CPU. | 1 CPU node, 15 min | First — verifies env. |
| `prepare.sh` | One-time: download data shards + train BPE tokenizer. | 1 CPU node, 30 min, 32G RAM | Once before any training. |
| `train_once.sh` | Run `train.py` exactly once. ~5 min training. | 1 GPU, 30 min wall | Phase 1 baseline, ad-hoc runs. |
| `autoresearch_overnight.sh` | Long-running agent-driven experiment loop. | 1 GPU, 12 h wall | Phase 3 (after agent CLI installed). |

For HPC clusters that block outbound HTTPS on compute nodes, see
[`interactive_agent.md`](interactive_agent.md) for the alternate
login-node-driven workflow.

## Typical first session

```bash
git clone https://github.com/spei04/voxeldreamer-ar.git
cd voxeldreamer-ar

# 1. Edit the SBATCH headers in each script for your cluster
# 2. Smoke test (CPU, ~2 min)
sbatch scripts/slurm/smoke_cpu.sh
squeue -u $USER

# 3. One-time data prep (CPU, ~2 min)
sbatch scripts/slurm/prepare.sh

# 4. Phase 1 baseline: one training run
sbatch scripts/slurm/train_once.sh
# Wait, then look at scripts/slurm/logs/vd-train-<jobid>.out for val_bpb
```

## Logs

All scripts write stdout/stderr to `scripts/slurm/logs/<jobname>-<jobid>.{out,err}`.
The directory is gitignored — those logs are local to your cluster checkout.

## Monitoring

```bash
squeue -u $USER                                 # your queue
sacct -X --format=JobID,JobName,State,Elapsed   # recent jobs
tail -f scripts/slurm/logs/vd-overnight-*.out   # live tail an overnight job
```

## Cancel + restart

```bash
scancel <jobid>                                  # cancel one job
scancel -u $USER -n vd-overnight                 # cancel all overnight jobs
```

## Customizing for your experiment

- Pass a description through env var for `results.tsv`:
  ```bash
  VD_DESC="3d-rope axis_dims=(16,6,6,4)" sbatch --export=ALL,VD_DESC scripts/slurm/train_once.sh
  ```
- Request specific GPU type:
  ```bash
  sbatch --constraint=h100 scripts/slurm/train_once.sh
  ```
- Increase time budget for longer runs (default 30min wall covers 5-min training + startup):
  ```bash
  sbatch --time=01:00:00 scripts/slurm/train_once.sh
  ```
