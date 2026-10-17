# The plan: a TTS that runs on essentially any machine

> **Status:** this experiment is concluded (17 October 2026) and kept for reference only; the code is not maintained. See [the final report](REPORT.md) for what was run and what we found.

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

### Data — much smaller than first planned
**~25 h per language, 125 h total.** The 900 h figure this document originally
carried was sized for a full fine-tune of a 1.7B model learning Indic almost
from scratch. Adapting a base that already speaks these languages, with LoRA,
needs roughly a seventh of that, and the evidence is consistent:

| source | result |
|---|---|
| IndicF5's own ablation | 1 h/lang MUSHRA 33.7, **10 h 61.5**, 100 h 64.3 — 10 h gets 96% of 100 h |
| PEFT-TTS | F5-TTS to Korean, LoRA, **12.65 h**, one TITAN RTX, beat CosyVoice 2 |
| chatterbox-indic-lora | Hindi CER 0.106 / Tamil 0.161 at **~10 h** |
| Rasa text selection | 1 h Tamil, CER **0.812 random vs 0.074 aksara-balanced** |

**Speakers are the binding constraint, not hours.** At least 50 distinct speakers
per language (unseen-speaker similarity 78.93 to 88.18; 15 speakers x 10 min beat
5 x 30 min by 34% relative CER). Fifty speakers at 30 minutes each *is* 25 h, so
the speaker requirement sets the corpus size and the hours follow.

What more hours would still buy is **tail robustness**, not average quality:
CosyVoice 3 measured easy-case CER 1.68 to 1.27 over 57x the data but hard-case
**27.60 to 6.96**. That is a later optimisation, not a prerequisite.

5. Download ~125 h (SYSPIN ungated, Rasa, IndicTTS, IndicVoices-R, Vaani),
   selected for speaker count first.
6. Greedy aksara-balanced text selection when subsampling.
7. ASR-agreement filter on every row — skipping it took WER 11.9% to 26.2% at
   identical hours.
8. Run `indic/train/prepare_shards.py` — about **3.5 GPU-hours**, 22 GB of
   downloads.

### Training
9. LoRA r=32 — measured as a *quality* decision, not a memory compromise: it
    beat full fine-tuning at 12.65 h and halved unseen-speaker degradation.
    **About 2 A100-hours, or ~9 hours on a laptop 5070 Ti**, at 50k steps on
    125 h. Training memory is **0.76 GB**, so an 8 GB card is enough and no
    cluster is needed.
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

---

## Recommended settings for edge deployment

Every value here is set by a measurement in this repository, not by convention.

### Threads — the setting most people get wrong

| ORT threads | RTF | x realtime | CPU-s per audio-s |
|---|---|---|---|
| **1** | 0.149 | 6.7x | **0.149** |
| 2 | 0.103 | 9.7x | 0.206 |
| 4 | 0.075 | 13.3x | 0.300 |
| 20 (library default) | 0.060 | 16.7x | **1.190** |

The default burns **8x the CPU for 2.5x the wall clock**. On a battery device
that is most of the battery; on a server it is most of the capacity. Set
`SessionOptions.intra_op_num_threads` explicitly -- note that `taskset` and
`OMP_NUM_THREADS` do **not** constrain ONNX Runtime, whose worker threads set
their own affinity and escape the inherited mask.

- **Battery / many streams: 1 thread.** 6.7x real time is plenty for one user.
- **Single interactive stream: 2 threads.** 9.7x, still only 1.4x the CPU.
- Above 4 threads is paying a lot for a little.

### Precision and footprint

| Precision | Weights | With vocoder + speaker encoder |
|---|---|---|
| fp32 | 1280 MB | 1340 MB |
| fp16 | 640 MB | 700 MB |
| **int8** | **320 MB** | **380 MB** |

**int8**, which this CPU accelerates through AVX-VNNI and ARM through dotprod.
A 380 MB resident footprint runs on any machine with 1 GB of free RAM.

### Chunking

Size the chunk by **seconds of audio**, not characters -- the same character
budget is a third more audio in Devanagari than in Latin (measured: 14.5
characters per second against 20.0). `CHUNK_SECONDS` is the knob:

- **Interactive (assistant, screen reader): 4-5 s.** Measured 2.31 s to first
  audio for 3% more total wall clock.
- **Batch (audiobooks): 9-19 s.** Fewer seams, latency irrelevant.

### Voice cloning

- **Ask for 15 s of reference, accept 8-20, reject under 6.** Not 30: the
  decoder truncates to 10 s anyway, and reference beyond that buys prosody
  rather than timbre (SIM-O flat from 30 s to 30 minutes).
- **Select the cleanest contiguous 10-12 s window by SNR** rather than taking
  the first 10 s. A noisy prompt costs **0.273 speaker similarity** -- far more
  than any amount of extra length returns.
- **Drop the in-context prefix; keep the speaker embedding.** The speaker head
  alone recovers 93.1% of the identity signal, and the prefix costs 750 prompt
  tokens -- on CPU, 750 more full weight reads.

### Minimum hardware

| Device | Threads | Projected RTF | Verdict |
|---|---|---|---|
| Any x86 laptop, 2010s or later | 2 | 0.1-0.3 | comfortable |
| Low-end N100 / Celeron | 2 | ~0.4 | comfortable |
| Raspberry Pi 5 | 4 | 0.3-0.4 | **the floor** |
| Raspberry Pi 4 | 4 | 1.0-1.3 | below real time |

**Requirements: 1 GB free RAM, one CPU core, no GPU, no network.** Pi 5 is the
oldest board that clears real time; Pi 4 does not, and int8 is what would decide
it. These are projections from a measured x86 baseline -- the Pi rows are
unverified until someone runs them on a Pi.
