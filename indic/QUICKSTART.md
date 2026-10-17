# Quickstart: from nothing to a trained model

> **Status:** this experiment is concluded (17 October 2026) and kept for reference only; the code is not maintained. See [the final report](../docs/indic/REPORT.md) for what was run and what we found.

Every command below is runnable today. Where something is blocked, it says so
and says what unblocks it.

## 0. Environment

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install pandas pyarrow peft bitsandbytes
```

On a cluster, pick the torch build that matches its CUDA — the `cu128` wheel
this was developed against is for a Blackwell consumer card and is wrong for an
A100. `python -m indic.hpc.capability` prints what it found and refuses hardware
the checkpoint cannot run on.

## 1. Credentials — this blocks four separate things

```bash
export HF_TOKEN=hf_...
```

Accept the terms once, with the same account, at each of:

| Repo | Blocks |
|---|---|
| `ai4bharat/indic-conformer-600m-multilingual` | every Indic error rate, and the data quality filter |
| `ai4bharat/IndicF5` | the base-model bake-off |
| `ai4bharat/indicvoices_r` | speaker variety, which is the binding constraint for cloning |
| `ai4bharat/Rasa` | expressive and emotion data |

Ungated, needing nothing: `google/fleurs`, `openai/whisper-large-v3`,
`microsoft/wavlm-base-plus-sv`, `sarvamai/sarvam-1`.

## 2. Check the harness still passes

```bash
.venv/bin/python -m indic.preflight          # 9 of 10 pass today
.venv/bin/python -m indic.train.selfcheck    # 9 of 9, ~8 seconds, CPU
```

A FAIL is a finding, not a crash. The one standing FAIL is Gate 2's reference
layout, which blocks *claiming* the layout is right and does not block training.

## 3. Decide the base model

```bash
.venv/bin/python -m indic.eval.run_eval --model ai4bharat/IndicF5 \
    --testset A --lang ta --reference-from fleurs --out runs/indicf5-A-ta
.venv/bin/python -m indic.eval.score --run runs/indicf5-A-ta --speaker-sim
```

Run it for hi/bn/ta/te/mr. **Speaker similarity is the primary endpoint**,
because similarity is the scarce resource and intelligibility is nearly
data-independent. Then confirm the winner distils to a single forward pass —
a 16-step flow model is 16x our inference budget and is not a CPU product.

## 4. Build the corpus

**~25 hours per language, ~125 hours total.** Not 900: see PLAN.md. Select for
**at least 50 distinct speakers per language** first and let the hours follow —
speakers are what buy cloning, and extra hours from the same speakers buy very
little.

Write a manifest, one JSON object per line:

```json
{"audio": "/data/syspin/hi/0001.wav", "text": "...", "speaker_id": "hi_spk07", "language": "hi"}
```

Then:

```bash
# one array task per shard; --asr auto needs HF_TOKEN
.venv/bin/python -m indic.train.prepare_shards \
    --manifest corpus.jsonl --out shards/ \
    --tokenizer sarvamai/sarvam-1 --asr auto \
    --shard $SLURM_ARRAY_TASK_ID --num-shards 16
```

About **3.5 GPU-hours** for 125 h, fully parallel. Shards are ~1 GB because the
codec tokens are tiny; a mel-based trainer reads `audio_path` and computes mel in
the dataloader rather than storing 17 GB of it.

The job writes `cer` per row and does not apply it. Filtering happens at read
time, **relative to each language's own ASR floor** — a flat threshold is
different strictness per language, since real human Tamil scores 4.87% CER
through this recognizer and real human Marathi scores 2.08%.

## 5. Train

```bash
.venv/bin/python -m indic.train.loop \
    --shards shards/ --shard-tokenizer sarvamai/sarvam-1 \
    --checkpoint /path/to/base/lm \
    --mode lora --lora-rank 32 --grad-checkpointing \
    --steps 50000 --batch-size 8 --workers 4 \
    --out checkpoints/s1
```

Swap `--checkpoint` for `--toy` to run the same path against a 2-layer model in
seconds, which is how to check the wiring before spending a GPU. Verified
end to end: real parquet shards, uniform language balancing, LoRA steps, loss
falling.

**About 2 A100-hours, or ~9 hours on a laptop 5070 Ti, in 0.76 GB.** No cluster
is needed at any point. LoRA r=32 is a quality decision rather than a memory
compromise: it beat full fine-tuning at 12.65 hours and halved unseen-speaker
degradation.

Defaults already set from measurements: near-uniform language sampling with a
repeat cap, references re-paired every epoch, loss on target audio only.

## 6. Score it

```bash
.venv/bin/python -m indic.eval.run_eval --model checkpoints/s1 --testset A --lang hi --out runs/s1-A-hi
.venv/bin/python -m indic.eval.score --run runs/s1-A-hi --speaker-sim
```

Always read error rates **against the floor** — the harness reports
`cer_above_floor` on every row because a rate without its floor is not
comparable across languages.

## 7. Ship to the edge

```bash
.venv/bin/python -m indic.bench.cpu_speed --models mini --threads 2
.venv/bin/python -m indic.bench.efficiency --lang hi --chunks 0
```

Settings and their justifications are in PLAN.md. The short version: **int8, 1-2
ONNX threads, 4-5 second chunks, 15 second reference clips SNR-selected to the
best 10-12 seconds, and no in-context prefix.**

## What is not written yet

- **`indic/train/loop.py` targets the autoregressive sequence format.** If the
  bake-off picks a mel-based base, the loss and the collator change; the shard
  reader, preprocessing, sampling, evaluation and HPC layers do not.
- **Mel feature extraction** in the dataloader — the shards carry `audio_path`
  for it, the extractor is not written.
- **A code-mix transliteration front end.** Measured as the cheapest quality win
  available: raw Hindi-English code-mix scores LLM-WER 0.855 and transliterating
  first takes it to 0.198. Days of work, not GPU time.
