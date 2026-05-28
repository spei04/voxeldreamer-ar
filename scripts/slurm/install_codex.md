# Installing the OpenAI Codex CLI on the cluster

Step-by-step for getting `codex` running on the Beery vision cluster so
`autoresearch_overnight.sh` Mode B can drive the experiment loop. Plain shell,
no sudo needed.

## Step 1: install on the login node

Two options. The binary install is cleanest on HPC — no Node toolchain,
self-contained, easy to update.

### Option A: prebuilt Linux binary (recommended on HPC)

```bash
ssh you@cluster
mkdir -p ~/bin
cd ~/bin

# Pick the right arch — `uname -m` on the login node:
#   x86_64  -> codex-x86_64-unknown-linux-musl.tar.gz
#   aarch64 -> codex-aarch64-unknown-linux-musl.tar.gz
ARCH=$(uname -m)
TARBALL="codex-${ARCH}-unknown-linux-musl.tar.gz"

# Find the latest release URL from the GitHub releases page,
# or grab it programmatically:
LATEST_URL=$(curl -sL https://api.github.com/repos/openai/codex/releases/latest \
    | grep "browser_download_url.*${TARBALL}" \
    | head -1 \
    | cut -d '"' -f 4)
curl -L -o codex.tar.gz "$LATEST_URL"
tar xzf codex.tar.gz
# The tarball usually drops the binary as `codex-<arch>-unknown-linux-musl`.
# Symlink to a plain `codex`:
ln -sf codex-${ARCH}-unknown-linux-musl codex
chmod +x codex
rm codex.tar.gz

# Make it findable
echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/bin:$PATH"
codex --version
```

### Option B: conda + npm in your `bpp` env

```bash
conda activate bpp
conda install -c conda-forge nodejs    # only first time
npm install -g @openai/codex            # installs into the conda env prefix
codex --version
```

Pros: Slurm jobs that already `conda activate bpp` get `codex` automatically.
Cons: heavier (Node toolchain in your env), slower install.

## Step 2: set up authentication

For unattended Slurm jobs you must use an **API key** — the interactive
"Sign in with ChatGPT" flow needs a browser. Get one from
https://platform.openai.com/api-keys and store it in your shell profile:

```bash
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

Test it works:

```bash
codex exec "What is 2+2? Answer in a single number."
```

Should return `4` with no further prompting.

## Step 3: test outbound HTTPS from a compute node

HPC clusters often restrict outbound network on compute nodes. Verify before
relying on Mode B in `autoresearch_overnight.sh`:

```bash
# Request a tiny interactive GPU job
srun --partition=vision-beery \
     --qos=vision-beery-main \
     --account=vision-beery \
     --gres=gpu:a100:1 \
     --cpus-per-task=4 \
     --mem=8G \
     --time=00:10:00 \
     --pty bash

# Inside the interactive session
source /data/vision/beery/scratch/serena/.bashrc
conda activate bpp
codex exec "What is 2+2?"
```

If that returns `4` — outbound is allowed, Mode B will work.
If it hangs or errors with a connection failure — outbound is blocked.
Fall back to `interactive_agent.md` (run codex on the login node, sbatch
per-experiment `train_once.sh` jobs from there).

## Step 4: switch overnight script to Mode B

Once codex + API key + outbound HTTPS all work, edit
`scripts/slurm/autoresearch_overnight.sh` and uncomment the Mode B block:

```bash
echo "[overnight] launching codex agent ..."
codex exec "Read program.md and run the experiment loop indefinitely. Never stop."
```

Comment out (or delete) the Mode C deterministic baseline block below it.

Submit with the API key passed through:

```bash
sbatch --export=ALL,OPENAI_API_KEY=$OPENAI_API_KEY scripts/slurm/autoresearch_overnight.sh
```

The `--export=ALL,...` is important — without it, your env vars (including
`OPENAI_API_KEY`) don't make it into the job.

## Updating codex later

```bash
# Option A users
cd ~/bin && rm codex codex-*-unknown-linux-musl
# (re-run the install snippet from Step 1)

# Option B users
conda activate bpp
npm update -g @openai/codex
```

## Troubleshooting

- `codex: command not found` after install → check `echo $PATH`; ensure
  `~/bin` (Option A) or your conda env's bin (Option B) is in it.
- `Unauthorized` errors → verify `OPENAI_API_KEY` is set and not stale.
  Test: `echo $OPENAI_API_KEY | head -c 12; echo`.
- Long hangs from compute node → outbound HTTPS is probably blocked; use
  the login-node workflow in `interactive_agent.md` instead.
