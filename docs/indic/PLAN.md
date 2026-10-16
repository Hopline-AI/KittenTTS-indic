# The plan: a TTS that runs on essentially any machine

Supersedes the architecture assumptions in
[2026-10-16-findings-and-decisions.md](2026-10-16-findings-and-decisions.md),
which were written before CPU-everywhere became a requirement. The measurements
in that document all stand; what changes is which model they point at.

## The target, in one line

**~320M parameters, single forward pass, speaker-embedding conditioned,
int8-quantised, running faster than real time on one CPU core.**

## Why that size, measured rather than chosen

Cost on CPU is **linear in seconds of audio, not in parameters**, because the
vocoder dominates. Measured on one core of an Intel Ultra 7 (143 GFLOPS), the
80M model costs 26 ms fixed plus **59 ms per second of audio** — so it
approaches 17x real time on a single core, and 15M to 80M costs only 1.15x
because the extra parameters live in the text encoder, which is ~8% of the work.

That means parameters are cheap and the vocoder is the floor:

| Machine (one core) | GFLOPS | RTF at 320M | vs real time |
|---|---|---|---|
| Modern laptop | 90-143 | 0.07 | 13.7x |
| 10-year-old laptop | 40 | 0.26 | 3.8x |
| Low-end N100 / netbook | 25 | 0.42 | 2.4x |
| Raspberry Pi 5 | 12 | 0.87 | 1.2x |
| Raspberry Pi 4 | 5 | 2.09 | 0.5x |

**320M is the knee.** It clears real time on everything down to a Pi 5 on a
single core; 640M loses the Pi and buys little, since the acoustic model is not
where the time goes.

## What this rules out, and why

**The 1.7B autoregressive model.** Measured at **RTF 2.439 on CPU** — slower
than real time, 13.5 s to first audio even at a 95-character chunk. The cost
structure is the problem, not the size: an autoregressive decoder re-reads every
weight for each of ~250 tokens, so cost is weight-bytes x tokens. A single-pass
model reads them once.

**Multi-step flow matching.** Cost is `2 x params x frames x NFE`. At 16
function evaluations over ~940 frames that is a multi-TFLOP job on a processor
measured in hundreds of GFLOPS — roughly RTF 2.4, the same place the 1.7B lands.
Distilling to 4 steps helps but never reaches a single pass, and distilling to 1
step costs about 0.06 speaker similarity, which is the one thing we cannot
spend.

**In-context reference prompting.** The 2x2 in `indic/data/teacher_forced.json`
shows the speaker head alone recovers **93.1%** of the identity signal and the
reference prefix alone **86.6%** — they are ~90% redundant. The prefix costs 750
prompt tokens; on CPU every prompt token is another full weight read. Keep the
embedding, drop the prefix.

## The honest limit

Zero-shot cloning quality scales with **pretraining** data and has not saturated
at 60,000 hours (0.151 at 60 h, 0.645 at 60k h). A 320M model trained on ~900 h
will not clone like a 60,000-hour base, and no amount of engineering changes
that. What this design delivers is good Indic speech with *recognisable*
speaker cloning, everywhere, on one core. Bulbul-grade cloning is not available
at this size and should not be promised.

Intelligibility is the opposite: nearly data-independent (WER 2.30 at 60 h,
2.05 at 60,000 h). That part is reachable.

## What remains

### Blocking, and cheap
1. **`HF_TOKEN` with gated terms accepted** — blocks the IndicF5 bake-off, every
   Indic CER number, and the codec gate's ASR arm. Blocked three separate things.
2. **Licence review of IndicF5** — listed MIT, derives from CC-BY-NC F5-TTS
   weights. Spark-TTS was relicensed post-release for exactly this.

### The base-model decision
3. **Bake-off**: score IndicF5 and the current model on the same test set,
   speaker similarity primary. Already written (`indic/eval/run_eval.py` ships an
   `IndicF5Synthesizer`), costs no engineering, needs only the token.
4. **Pick the base** by that result, then adapt rather than train from scratch —
   from-scratch on 900 h measured MUSHRA 43.2 against 73.4 fine-tuned.

### Data, the long pole
5. Download ~900 h (SYSPIN ungated ~420 h, Rasa, IndicTTS, IndicVoices-R, Vaani).
6. **At least 50 distinct speakers per language** — this matters more than hours
   for cloning. Many-speaker data moved unseen-speaker similarity 78.93 to 88.18
   on these languages; 15 speakers x 10 min beat 5 x 30 min by 34% relative CER.
7. **Greedy aksara-balanced text selection** when subsampling — Tamil CER 0.812
   random against 0.074 greedy at one hour.
8. ASR-agreement filter on every row — skipping it took WER 11.9% to 26.2% at
   identical hours.
9. Run `indic/train/prepare_shards.py` as a Slurm array (30-80 GPU-hours).

### Training
10. Phase A: embedding warm-up, body frozen. Phase B: LoRA r=32 — measured as a
    *quality* decision, not a memory compromise (it beat full fine-tuning at
    12.65 h and halved unseen-speaker degradation).
11. Near-uniform language sampling with the epoch cap (already implemented).
12. 20-25% English replay. The "drop English" result everyone cites was a 50/50
    mix that never measured English at all.

### Then ship
13. int8 quantisation (this CPU has AVX-VNNI), single-pass export, streaming.
14. Measure on the real target machines, not this laptop.

## What is already done

17 commits. 9/10 preflight checks pass. The codec gate passed, the training loop
passes 9/9 self-checks including overfit-to-1e-4, three vocabulary arms verified
12/12, the eval harness has ASR floors and non-circular speaker scoring, and the
chunking fix took Hindi time-to-first-audio from 8.70 s to 5.03 s.

Seven bugs were fixed, five of which would have produced confident wrong answers
rather than failures.

## Tamil is the arm most likely to fail

Worst on every axis at once: perplexity 287 against English 61, 10.13 tokens per
word under Qwen3, no SYSPIN corpus, 24 h in Vaani against Hindi's 1,004. Plan for
it explicitly instead of discovering it.
