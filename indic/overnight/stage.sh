#!/bin/bash
# One stage of the overnight pipeline: prepare | train | eval.
# Run by stage.sbatch on a GPU node, or directly by run.sh when DRY=1.
# Every stage leaves DATA/status/<stage>.{started,ok,failed}; eval always
# rewrites DATA/REPORT.md.
set -uo pipefail
STAGE=$1
: "${DATA:?}" "${ARM:=sarvam}" "${DRY:=0}" "${PY:=python}" "${TRAIN_MIN:=300}" \
  "${PREP_PROCS:=4}" "${BATCH:=16}" "${FETCH_WAIT_MIN:=240}"
cd "$(dirname "$0")/../.."
mkdir -p "$DATA/status" "$DATA/logs"
rm -f "$DATA/status/$STAGE".{ok,failed}
date '+%F %T' > "$DATA/status/$STAGE.started"
fail() { echo "$*" > "$DATA/status/$STAGE.failed"; echo "FAILED: $*" >&2; exit 1; }

if [ "$DRY" = 1 ]; then
  DEVICE=cpu; export KINDIC_DRY=1
else
  DEVICE=cuda
  eval "$("$PY" -m indic.hpc.stage_cache env --hf-home "$STAGED_HF_HOME")" || fail "stage_cache env"
  KITTEN=$(ls -d "$STAGED_HF_HOME"/hub/models--KittenML--kitten-tts-2/snapshots/*/ | head -1)
fi
if [ "$ARM" = sarvam ]; then
  TOK=sarvamai/sarvam-1; CKPT="$DATA/ckpt-sarvam"
else
  TOK="${KITTEN:-dry}lm"; CKPT="${KITTEN:-dry}"
fi
[ "$DRY" = 1 ] && TOK=dry
LANGS=$(cat "$DATA/langs.txt")

prepare() {
  local deadline=$(( $(date +%s) + FETCH_WAIT_MIN * 60 ))
  for code in $LANGS; do
    while [ ! -f "$DATA/manifests/$code.DONE" ] && [ ! -f "$DATA/manifests/$code.FAILED" ]; do
      [ "$(date +%s)" -gt "$deadline" ] && { echo "[$code] fetch never finished; skipping"; continue 2; }
      sleep 30
    done
    [ -f "$DATA/manifests/$code.FAILED" ] && { echo "[$code] fetch failed; skipping"; continue; }
    for split in heldout train; do
      local procs=$PREP_PROCS; [ "$split" = heldout ] && procs=1
      local pids=()
      for ((i = 0; i < procs; i++)); do
        "$PY" -m indic.overnight.prepare --manifest "$DATA/manifests/$code.$split.jsonl" \
          --out "$DATA/shards/$split/$code" --tokenizer "$TOK" --asr none --device "$DEVICE" \
          --shard "$i" --num-shards "$procs" --rows-per-shard 500 \
          >> "$DATA/logs/prepare-$code.log" 2>&1 &
        pids+=($!)
      done
      for pid in "${pids[@]}"; do wait "$pid" || echo "[$code/$split] a prepare task failed (see logs/prepare-$code.log)"; done
    done
    echo "[$code] prepared: $(ls "$DATA/shards/train/$code"/*.parquet 2>/dev/null | wc -l) train shard files"
  done
  ls "$DATA"/shards/train/*/*.parquet > /dev/null 2>&1 || fail "no training shards were written"
}

train() {
  local args=(--mode lora --shards "$DATA/shards/train/*/*.parquet" --out "$DATA/run"
              --steps 200000 --time-budget-min "$TRAIN_MIN" --save-every 500)
  [ -f "$DATA/run/latest.pt" ] && args+=(--resume "$DATA/run/latest.pt")
  if [ "$DRY" = 1 ]; then
    args+=(--toy --device cpu --batch-size 2 --workers 0 --log-every 1 --save-every 5)
  else
    [ -f "$CKPT/lm/model.safetensors" ] || [ -f "$CKPT/lm/model-ternary.safetensors" ] \
      || fail "no checkpoint at $CKPT/lm (run.sh builds it before submitting)"
    args+=(--checkpoint "$CKPT/lm" --shard-tokenizer "$TOK" --device cuda
           --batch-size "$BATCH" --grad-checkpointing --lr 1e-4 --lr-embed 3e-4
           --warmup 200 --log-every 10 --workers 4 --max-tokens 1200)
  fi
  "$PY" -m indic.train.loop "${args[@]}" || fail "training loop (see the job .err)"
}

evaluate() {
  local args=(--data "$DATA" --device "$DEVICE")
  if [ "$DRY" = 1 ]; then
    args+=(--dry --baseline baseline --finetuned "$DATA/run/final.pt")
  else
    args+=(--baseline "$KITTEN" --finetuned "$DATA/run/export")
  fi
  "$PY" -m indic.overnight.evaluate "${args[@]}" || fail "evaluation (REPORT.md has the error)"
}

case "$STAGE" in
  prepare) prepare ;;
  train) train ;;
  eval) evaluate ;;
  *) fail "unknown stage $STAGE" ;;
esac
date '+%F %T' > "$DATA/status/$STAGE.ok"
# eval wrote REPORT.md before eval.ok existed; refresh the stage table.
[ "$STAGE" = eval ] && "$PY" -m indic.overnight.evaluate --data "$DATA" --finetuned - --report-only > /dev/null
exit 0
