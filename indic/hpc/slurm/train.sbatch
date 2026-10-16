#!/bin/bash
# Single-node fine-tuning, on one GPU or on all of the node's GPUs.
#
# torchrun rather than `srun python`: the training step wants one process per GPU
# with NCCL between them, and torchrun sets the rendezvous up from one task.
# Launching N srun tasks that each also try to be rank 0 is the classic way to get
# a job that hangs in init_process_group until its wall clock runs out.
#
# One GPU is not a special case here -- torchrun with --nproc_per_node=1 is the
# same code path with a one-member process group -- so the same file covers the
# pilot (one card, Hindi only) and the main run.
#
#   sbatch -p hgx2q --gres=gpu:4 indic/hpc/slurm/train.sbatch
#   sbatch -p hgx2q --gres=gpu:1 indic/hpc/slurm/train.sbatch     # pilot
#
# Multi-NODE is deliberately out of scope. It needs a rendezvous endpoint, the
# right interconnect interface in NCCL_SOCKET_IFNAME, and a per-site answer about
# whether the fabric is InfiniBand or Ethernet. The compute estimate in
# docs/indic/training.md (roughly 50-100 A100-hours for the main run) fits inside
# one node, so the complexity buys nothing yet.

# ─── edit these ───────────────────────────────────────────────────────────────
#SBATCH --job-name=kindic-train
##SBATCH --partition=REPLACE_ME
##SBATCH --account=REPLACE_ME
#
# GPU generation. Training needs bf16, and bf16 tensor cores start at Ampere
# (sm_80). A pre-Ampere card does not merely run slower: fp16 overflows this
# checkpoint's activations, so there is no half-precision fallback. Find your
# site's feature names with
#     sinfo -o "%20N %10c %10m %25f %15G"
##SBATCH --constraint=REPLACE_ME_WITH_AMPERE_OR_NEWER
#
# One task that owns every GPU on the node; torchrun forks the ranks itself.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#
# Data loading is CPU-bound (audio decode, S3 token lookup), so give the node's
# cores to the one task and let the per-rank dataloader workers divide them.
#SBATCH --cpus-per-task=16
#
# Host RAM. A full fine-tune with AdamW is about 16 bytes/param, roughly 27 GB,
# but that lives in VRAM; this is for the dataset shards, the dataloader workers
# and the CPU-side unpack of the ternary checkpoint.
# No --mem: eX3 nodes report RealMemory=1, so any --mem request is rejected.
# On a cluster that tracks memory, pass e.g. --mem=64G on the sbatch line.
#SBATCH --time=24:00:00
# In the submit dir: Slurm will not create a missing logs/ and then writes nothing.
#SBATCH --output=train-%j.out
#SBATCH --error=train-%j.err
#
# Preemption: the loop writes $RUN_DIR/latest.pt every SAVE_EVERY steps and a
# requeued job resumes from it, so at most SAVE_EVERY steps are lost.
#SBATCH --requeue

set -euo pipefail

: "${STAGED_HF_HOME:?set STAGED_HF_HOME to the directory staged by indic.hpc.stage_cache}"
: "${RUN_DIR:=runs/${SLURM_JOB_ID:-local}}"   # stable across requeues
: "${SAVE_EVERY:=500}"
: "${REPO_DIR:=$(pwd)}"
: "${PYTHON:=python}"
: "${ACTIVATE:=}"

[ -n "$ACTIVATE" ] && source "$ACTIVATE"
cd "$REPO_DIR"
mkdir -p "$RUN_DIR"

eval "$("$PYTHON" -m indic.hpc.stage_cache env --hf-home "$STAGED_HF_HOME")"

export TMPDIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}/kindic-train-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
export TRITON_CACHE_DIR="$TMPDIR/triton"
export TORCHINDUCTOR_CACHE_DIR="$TMPDIR/inductor"

# How many ranks. SLURM_GPUS_ON_NODE is what the scheduler actually gave this job,
# which is the only trustworthy count: nvidia-smi reports every card on the node,
# including the ones belonging to someone else's job.
NPROC="${SLURM_GPUS_ON_NODE:-$(awk -F, '{print NF}' <<<"${CUDA_VISIBLE_DEVICES:-0}")}"

# Threads per rank, so N ranks do not each spawn one thread per core on the node.
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / NPROC ))
[ "$OMP_NUM_THREADS" -lt 1 ] && export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# NCCL diagnostics. WARN is quiet when things work and says which transport failed
# when they do not; the alternative is a job that hangs in the first all-reduce
# with no output at all. If a node has no NVLink between its cards and P2P over
# PCIe is broken (some virtualised or MIG-partitioned hosts), NCCL_P2P_DISABLE=1
# is the workaround -- uncomment it only after seeing that failure, since it costs
# real bandwidth.
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
# export NCCL_P2P_DISABLE=1

echo "== preflight on $(hostname), $NPROC rank(s) =="
# --require-gpu makes "the scheduler gave us no GPU" an immediate failure instead
# of a training run that silently falls back to the CPU and never finishes. The
# report also refuses pre-Ampere cards, which is the check that matters here.
"$PYTHON" -m indic.hpc.capability --require-gpu --out "$RUN_DIR/capability.json"
"$PYTHON" -m indic.hpc.stage_cache verify --hf-home "$STAGED_HF_HOME" --no-load-check
"$PYTHON" -m indic.hpc.env_lock capture --out "$RUN_DIR/env.json" \
  --label "${SLURM_JOB_NAME:-train}"

: "${TRAIN_MODULE:=indic.train.loop}"
RESUME=()
[ -f "$RUN_DIR/latest.pt" ] && RESUME=(--resume "$RUN_DIR/latest.pt")

# --standalone sets up a single-node rendezvous on localhost, which is what makes
# this work unchanged from one GPU to eight.
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NPROC" \
  -m "$TRAIN_MODULE" \
  --device cuda --out "$RUN_DIR" --save-every "$SAVE_EVERY" \
  "${RESUME[@]}" "$@"
echo "== done: $RUN_DIR =="
