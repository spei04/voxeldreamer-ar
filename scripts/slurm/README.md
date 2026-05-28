# Slurm scripts for VoxelDreamer-AR

`sbatch`-ready scripts for the Beery vision cluster (vision-beery partition,
A100 GPUs, conda env `bpp`).

## Already configured for your cluster

All SBATCH directives are filled in:

```bash
#SBATCH -o /data/vision/beery/scratch/serena/slurm_job/logs/%j.log
#SBATCH --partition=vision-beery
#SBATCH --qos=vision-beery-main
#SBATCH --account=vision-beery
#SBATCH --gres=gpu:a100:1                  # GPU jobs only
#SBATCH --cpus-per-task=16
```

`_common.sh` sources `/data/vision/beery/scratch/serena/.bashrc`, activates
the `bpp` conda env, then ensures `uv` is installed (autoresearch's project
manager). `uv` and conda coexist cleanly — conda gives us a Python interpreter
and base scientific stack, `uv` manages the autoresearch-specific deps in `.venv`.

## A100 note

Upstream autoresearch was tuned for H100. On 40GB A100 you may need to lower
`DEPTH` in `train.py` (default 8 → 6 or 4) to avoid OOM. On 80GB A100 the
defaults should fit. Watch the first `train_once.sh` run for `peak_vram_mb:`
in the log — if it's >38000 on a 40GB card, drop DEPTH and re-run.

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
# On the cluster (login node)
cd /data/vision/beery/scratch/serena
git clone https://github.com/spei04/voxeldreamer-ar.git
cd voxeldreamer-ar

# 1. CPU sanity check (~2 min)
sbatch scripts/slurm/smoke_cpu.sh
squeue -u $USER

# 2. One-time data prep (~2 min wallclock + queue time)
sbatch scripts/slurm/prepare.sh

# 3. Phase 1 baseline: one training run
sbatch scripts/slurm/train_once.sh
# Look at /data/vision/beery/scratch/serena/slurm_job/logs/<jobid>.log for val_bpb
```

## Logs

All scripts write to `/data/vision/beery/scratch/serena/slurm_job/logs/%j.log`
(your existing convention). Quick tail:

```bash
ls -t /data/vision/beery/scratch/serena/slurm_job/logs/ | head -5
tail -f /data/vision/beery/scratch/serena/slurm_job/logs/<jobid>.log
```

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
