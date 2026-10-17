#!/bin/bash
# Slurm wrapper for one overnight stage; run.sh passes partition, GPU, time and
# dependencies on the sbatch line. No --mem: eX3 nodes advertise RealMemory=1.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --requeue
[ -n "${ACTIVATE:-}" ] && source "$ACTIVATE"
export TMPDIR="/work/$USER/kindic-$SLURM_JOB_ID"; mkdir -p "$TMPDIR" 2>/dev/null || export TMPDIR=/tmp
ulimit -v unlimited 2>/dev/null   # a login-node limit propagated by sbatch would break big loads
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
echo "== $1 on $(hostname) job $SLURM_JOB_ID $(date '+%F %T')"
bash "$REPO_DIR/indic/overnight/stage.sh" "$1"
rc=$?
rm -rf "/work/$USER/kindic-$SLURM_JOB_ID"
exit $rc
