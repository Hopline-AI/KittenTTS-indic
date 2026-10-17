#!/bin/bash
# Overnight GPU test suite on one node, one job at a time (singleton).
#
#   CKPT=.../snapshots/<rev>/lm STAGED_HF_HOME=... ACTIVATE=... \
#     bash indic/hpc/slurm/night.sh [partition]      # default h200q
#   bash indic/hpc/slurm/night.sh report             # in the morning
#
# Jobs (synthetic data, embed mode -- this measures the GPU, not quality):
#   b16 b32 b64   1 GPU batch sweep: throughput and peak memory per batch size
#   ddp2          2 GPUs, batch 16 each: NCCL works, speedup vs b16
#   resume-a/-b   b reuses a's RUN_DIR and must continue from latest.pt (step 40)
#   soak          2 GPUs until the 60 min limit (ends TIMEOUT, expected): drift over time
# ponytail: singleton queue, so one OOM does not stop the rest.
set -euo pipefail
cd "$(dirname "$0")/../../.."
NIGHT=runs/night

if [ "${1:-}" = report ]; then
  sacct -n -X --name=kindic-night -S now-1days -o JobID%10,Comment%12,State%12,Elapsed,NodeList 2>/dev/null || true
  for d in "$NIGHT"/*/; do
    python - "$d" <<'PY'
import json, sys, statistics as st
d = sys.argv[1]
try:
    rows = [r for r in map(json.loads, open(d + "train.jsonl")) if "step" in r]
except FileNotFoundError:
    print(f"{d:28} no train.jsonl (job failed before training; see its .err)"); sys.exit()
warm = [r for r in rows if r["step"] >= 5] or rows
mem = max((r.get("max_mem_gb") or 0) for r in rows)
print(f"{d:28} steps {rows[0]['step']}-{rows[-1]['step']:<5} "
      f"tok/s {st.median(r['tok_per_s'] for r in warm):9.0f}  "
      f"step {st.median(r['step_s'] for r in warm):6.3f}s  mem {mem:5.1f} GB  "
      f"pad {st.median(r['pad_frac'] for r in warm):.2f}  "
      f"loss {rows[0]['loss']:.3f}->{rows[-1]['loss']:.3f}")
PY
  done
  echo "resume: runs/night/resume-a/ should show steps 0-59 and resume-b.out a resume from step 40"
  exit 0
fi

: "${CKPT:?set CKPT}" "${STAGED_HF_HOME:?}" "${ACTIVATE:?}"
PART="${1:-h200q}"
export STAGED_HF_HOME ACTIVATE
mkdir -p "$NIGHT"
COMMON=(--checkpoint "$CKPT" --mode embed --save-every 0)

job() {  # name gpus minutes [train args...]
  local name=$1 gpus=$2 mins=$3; shift 3
  RUN_DIR="$NIGHT/${RESUME_DIR:-$name}" sbatch --parsable -p "$PART" --gres=gpu:$gpus \
    --time=$mins --job-name=kindic-night --dependency=singleton --comment="$name" \
    --output="$NIGHT/$name.out" --error="$NIGHT/$name.err" \
    indic/hpc/slurm/train.sbatch "${COMMON[@]}" "$@"
  echo "  queued $name"
}

job b16      1 20 --steps 60 --batch-size 16
job b32      1 20 --steps 60 --batch-size 32
job b64      1 20 --steps 60 --batch-size 64
job ddp2     2 20 --steps 60 --batch-size 16
job resume-a 1 20 --steps 40 --batch-size 16 --save-every 20
RESUME_DIR=resume-a job resume-b 1 20 --steps 60 --batch-size 16 --save-every 20
job soak     2 60 --steps 100000 --batch-size 16 --log-every 20
echo "all queued; check with: squeue -u \$USER   morning: bash $0 report"
