#!/bin/bash
# Overnight KittenTTS-indic, one command on the eX3 login node:
#
#   IndicVoices-R (22 languages, <=HOURS each) -> shards -> Sarvam vocab swap
#   -> LoRA fine-tune on 1 GPU -> export -> baseline vs fine-tuned -> REPORT.md
#
#   cd ~/D1/kindic && source <venv>/bin/activate
#   export HF_TOKEN=... STAGED_HF_HOME=/global/D1/homes/$USER/hf
#   bash indic/overnight/run.sh                 # real run (checks first, then submits)
#   DRY=1 DATA=/tmp/k bash indic/overnight/run.sh   # laptop rehearsal, no GPU, no network
#
#   RESUME=1 bash indic/overnight/run.sh       # data already prepared: vocab swap, train, eval only
#
# Knobs (env): ARM=sarvam|qwen  HOURS=10  TRAIN_MIN=300  BATCH=16  PART=h200q
#              LANGS="hi:Hindi ta:Tamil ..."  DATA=/global/D1/homes/$USER/kindic-data
# Morning:     cat $DATA/REPORT.md
set -euo pipefail
REPO_DIR=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO_DIR"
: "${DRY:=0}" "${RESUME:=0}" "${ARM:=sarvam}" "${HOURS:=10}" "${TRAIN_MIN:=300}" "${BATCH:=16}" \
  "${PART:=h200q}" "${PY:=python}" "${DATA:=/global/D1/homes/$USER/kindic-data}"
# Sarvam-friendly scripts first, so they are prepared first if the night runs short.
: "${LANGS:=hi:Hindi bn:Bengali ta:Tamil te:Telugu mr:Marathi kn:Kannada ml:Malayalam
or:Odia pa:Punjabi gu:Gujarati ne:Nepali kok:Konkani mai:Maithili brx:Bodo doi:Dogri
sa:Sanskrit as:Assamese ur:Urdu sd:Sindhi ks:Kashmiri mni:Manipuri sat:Santali}"
export DATA ARM DRY PY TRAIN_MIN BATCH REPO_DIR
mkdir -p "$DATA/logs" "$DATA/status"
for pair in $LANGS; do echo "${pair%%:*}"; done > "$DATA/langs.txt"
say() { echo "== $*"; }

if [ "$DRY" = 1 ]; then
  say "dry run in $DATA"
  for pair in $LANGS; do
    "$PY" -m indic.overnight.fetch --lang "${pair#*:}" --code "${pair%%:*}" --out "$DATA" --dry > /dev/null
  done
  date '+%F %T' > "$DATA/status/fetch.ok"
  for stage in prepare train eval; do
    say "$stage"; bash indic/overnight/stage.sh "$stage" || true
  done
  say "report"; cat "$DATA/REPORT.md"
  exit 0
fi

# ── checks: everything that can fail in minutes fails here, not at 3 am ────
: "${HF_TOKEN:?export HF_TOKEN (accept ai4bharat/indicvoices_r terms first)}"
: "${STAGED_HF_HOME:?export STAGED_HF_HOME=/global/D1/homes/\$USER/hf}"
export ACTIVATE="${ACTIVATE:-${VIRTUAL_ENV:?activate the venv first}/bin/activate}" STAGED_HF_HOME
command -v sbatch > /dev/null || { echo "run this on the login node"; exit 1; }
"$PY" -c "import torch, transformers, peft, pyarrow, soundfile, librosa, huggingface_hub" \
  || { echo "venv is missing packages"; exit 1; }
say "partition $PART time limit: $(sinfo -h -p "$PART" -o %l 2>/dev/null | head -1 || true) (need ~$(( (TRAIN_MIN + 45) / 60 + 6 )) h total across 3 jobs)"
KITTEN=$(ls -d "$STAGED_HF_HOME"/hub/models--KittenML--kitten-tts-2/snapshots/*/ | head -1)
[ -n "$KITTEN" ] || { echo "kitten-tts-2 not staged; run indic.hpc.stage_cache download"; exit 1; }
SARVAM=$(ls -d "$STAGED_HF_HOME"/hub/models--sarvamai--sarvam-1/snapshots/*/ 2>/dev/null | head -1)
[ "$ARM" = qwen ] || [ -n "$SARVAM" ] || { echo "sarvam-1 tokenizer not staged"; exit 1; }

if [ "$ARM" = sarvam ] && [ ! -f "$KITTEN/lm/model.safetensors" ]; then
  say "fetching bf16 weights the vocab swap reads (3.5 GB, once)"
  "$PY" - "$STAGED_HF_HOME" "$(basename "$KITTEN")" <<'PY'
import sys
from huggingface_hub import hf_hub_download
hf_hub_download("KittenML/kitten-tts-2", "lm/model.safetensors", revision=sys.argv[2],
                cache_dir=f"{sys.argv[1]}/hub")
PY
fi

if [ "$RESUME" != 1 ]; then
say "probe: 3 Hindi clips from IndicVoices-R (gate + token)"
"$PY" -m indic.overnight.fetch --lang Hindi --code hi --out "$DATA/smoke" --probe
say "smoke: real encoders on CPU, offline, 2 clips"
( eval "$("$PY" -m indic.hpc.stage_cache env --hf-home "$STAGED_HF_HOME")"
  TOK=sarvamai/sarvam-1; [ "$ARM" = qwen ] && TOK="${KITTEN}lm"
  "$PY" -m indic.overnight.prepare --manifest "$DATA/smoke/manifests/hi.train.jsonl" \
    --out "$DATA/smoke/shards" --tokenizer "$TOK" --asr none --device cpu --limit 2 ) \
  || { echo "smoke failed: fix this before spending the night"; exit 1; }
say "smoke OK"

# ── fetch on this login node, in the background, 3 languages at a time ────
say "starting downloads (log: $DATA/logs/fetch.log)"
rm -f "$DATA/status/fetch".*
export HOURS
printf '%s\n' $LANGS > "$DATA/langs.pairs"
nohup setsid bash -c 'xargs -P 3 -I{} bash -c '\''p={}; "$PY" -m indic.overnight.fetch --lang "${p#*:}" --code "${p%%:*}" --out "$DATA" --hours "$HOURS"'\'' < "$DATA/langs.pairs"; date '+%F %T' > "$DATA/status/fetch.ok"' \
  > "$DATA/logs/fetch.log" 2>&1 < /dev/null &
fi

# ── Sarvam vocab swap in the foreground, offline like the jobs: a failure shows
#    up now instead of in a 3 am log. On a compute node via srun: the login node
#    cannot mmap the 3.5 GB weights. A few minutes once the node is granted.
if [ "$ARM" = sarvam ] && [ ! -f "$DATA/ckpt-sarvam/lm/model.safetensors" ]; then
  say "Sarvam vocab swap on a $PART node, waits for the queue (log: $DATA/logs/vocab.log)"
  ( eval "$("$PY" -m indic.hpc.stage_cache env --hf-home "$STAGED_HF_HOME")"
    srun -p "$PART" --gres=gpu:1 --cpus-per-task=8 --time=30 --job-name=kindic-vocab \
      bash -c 'echo "address-space limit: soft $(ulimit -v) hard $(ulimit -Hv)"; ulimit -v unlimited 2>/dev/null; exec "$@"' _ \
      "$PY" -m indic.vocab.build_sarvam_vocab --source "$KITTEN" --tokenizer "$SARVAM" \
      --out "$DATA/ckpt-sarvam" --init mean_rescaled ) > "$DATA/logs/vocab.log" 2>&1 \
    || { tail -25 "$DATA/logs/vocab.log"; echo "vocab swap failed: paste the lines above"; exit 1; }
  say "vocab swap OK"
fi

# ── GPU jobs, chained; each runs even if the one before failed (afterany) ──
submit() {  # stage minutes [dependency]
  sbatch --parsable -p "$PART" --gres=gpu:1 --time="$2" --job-name="kindic-$1" \
    ${3:+--dependency=afterany:$3} --export=ALL \
    -o "$DATA/logs/$1-%j.out" -e "$DATA/logs/$1-%j.err" \
    indic/overnight/stage.sbatch "$1"
}
J1=""; [ "$RESUME" = 1 ] || J1=$(submit prepare 300)
J2=$(submit train $(( TRAIN_MIN + 45 )) "$J1")
J3=$(submit eval 90 "$J2")
say "submitted prepare=${J1:-skipped} train=$J2 eval=$J3"
cat <<EOF

Done for tonight. Nothing else to run.
  progress:   squeue -u \$USER ; ls $DATA/status ; tail $DATA/logs/fetch.log
  morning:    cat $DATA/REPORT.md
EOF
