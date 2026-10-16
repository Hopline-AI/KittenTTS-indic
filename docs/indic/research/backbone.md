# The backbone decision: keep the AR speech-LM, or move to flow matching?

**Session of 2026-10-16.** This document answers one question: what should the
central model be? Every claim carries a URL in §11. Numbers *we* measured this
week are marked **[ours]** and live in
`docs/indic/2026-10-16-findings-and-decisions.md`. Claims I could not verify are
marked **UNVERIFIED** and are not load-bearing.

---

## 1. Recommendation

**Move the backbone to F5-style non-autoregressive flow matching, with IndicF5 as
the research base starting now.**

But the reason is not the one the brief expects, and the difference matters for
everything downstream. **The data-efficiency axis is not AR vs NAR.** It is
*pretrained prior vs no pretrained prior*, and both families collapse without one:

| same model, same data, 12.65 h of a new language (Korean) | CER | WER | SIM-O | NMOS |
|---|---|---|---|---|
| F5-TTS **trained from scratch** | **17.345** | **45.883** | 0.641 | 2.64 |
| F5-TTS **pretrained**, full fine-tune (336M) | 5.671 | 12.032 | 0.659 | 3.58 |
| F5-TTS **pretrained**, LoRA on **1.72% of params** | **4.906** | 12.625 | 0.640 | **3.88** |
| CosyVoice 2 (0.5B AR hybrid, *Korean in its pretraining*) | 6.812 | 17.956 | 0.583 | 3.49 |
| Ground truth | 6.519 | 12.679 | 0.625 | 4.74 |

That is one table from one paper (PEFT-TTS, Interspeech 2025) and it does four
jobs at once. Read top to bottom: a pretrained prior is worth **3× the CER**;
LoRA on 5.81M parameters *beats* a full fine-tune; the whole thing ran on **a
single NVIDIA TITAN RTX**; and the 336M flow-matching model beats a 0.5B AR hybrid
that already had Korean in its training data, on CER, WER, SIM-O *and* NMOS.

So the recommendation rests on three legs, in order of weight:

1. **A 336M flow-matching model adapts to a new language on ~12 h and one
   consumer GPU.** Our 12 GB card fits LoRA and nothing else **[ours]** — which
   is exactly the regime this result lives in. IndicF5's own ablation says
   **10 h/language** retains quality (0.8% average drop vs 100 h); our budget of
   150–200 h/language is 15–20× that knee.
2. **It is ~5× faster and 5× smaller than what we have.** RTF 0.15 @16 NFE against
   our measured **0.78** **[ours]**; 336M against 1.7B ternary + 325M dense.
3. **IndicF5 is the only open base with a real Indic prior across all five of our
   languages.** Everything else with a clean licence has Hindi at best.

**Do not ship IndicF5's weights.** It is listed MIT on HuggingFace and is, by its
own paper, a fine-tune of the English F5-TTS checkpoint, which its authors
released **CC-BY-NC-4.0** because of Emilia. Our brief makes licences a hard
filter, so the shipping base must be a re-run of IndicF5's published recipe from
an **Apache-2.0** F5 checkpoint on **CC-BY-4.0** Indic data. See §6.

**Keep KittenTTS 2 as the fallback, not the baseline.** Its decoder stack *is*
CosyVoice's — same 6561-token FSQ codec at 25 Hz **[ours]** — so the streaming
upgrade is a decoder swap in the same token space, not a rewrite. That is the
right answer if sub-300 ms conversational streaming turns out to be a hard product
requirement, and it is the only thing NAR genuinely cannot do.

### The honest counter-argument, stated up front

**AR is not inherently data-hungry, and I found direct evidence against that
claim.** NVIDIA's Align2Speak adds a *new language* — including **Hindi** — to
Koel-TTS, a genuinely autoregressive codec LM, from **30 minutes** of data:
Portuguese CER falls from 33.00% to 3.94% after fine-tuning plus ASR-guided GRPO,
with the same trend reported for Hindi. **No hours-matched AR-vs-NAR head-to-head
study exists** — UNVERIFIED in both directions. What is demonstrated is that *the
base model's multilingual coverage and its alignment mechanism* decide data
efficiency, not the AR/NAR axis.

This weakens leg 1 of the recommendation considerably. It does not overturn it,
because legs 2 and 3 — inference cost and Indic coverage of *available open
bases* — are unaffected. But it changes the fallback plan (§8) and it is the
reason §7's experiment is about measuring IndicF5's prior rather than about
validating flow matching.

### What to do this week

The deciding experiment **is already written and costs no GPU-hours**.
`indic/eval/run_eval.py` already ships an `IndicF5Synthesizer`, and its module
docstring already names `--model ai4bharat/IndicF5` as a worked example. It needs
an `HF_TOKEN` and nothing else. See §7.

---

## 2. The recommended architecture, concretely

| | |
|---|---|
| Family | **NAR flow matching**, F5-TTS/DiT lineage |
| Research base **now** | `ai4bharat/IndicF5` — 337M DiT, 11 Indic languages, trained |
| Shipping base **later** | Our own EN→IN run from `mrfakename/OpenF5-TTS` (Apache-2.0) |
| Size | ~336M, against our 1.7B ternary + 325M dense **[ours]** |
| Adaptation | LoRA r=16 on DiT q/v + prompt adapter ≈ **5.81M trainable (1.72%)** |
| Audio | 100-mel @ 24 kHz → Vocos (13.6M); *replaces* S3 tokens + s3gen |
| Inference | 16–32 NFE Euler; RTF 0.15 @16 NFE, 0.0394 served (L20, TRT-LLM) |
| Cloning | In-context: reference audio + transcript prefixed to the mel sequence |

What changes: we stop predicting discrete codec tokens with a causal LM and start
predicting a mel spectrogram with a non-causal transformer that sees the whole
utterance at once. The reference voice stops being an x-vector injected at
position 0 and becomes *a prefix of the thing being generated*.

---

## 3. The comparison table

**RTF is not comparable across rows** — different GPUs, batch sizes and utterance
lengths — so the GPU is named wherever it was stated. Nor are the similarity
columns: SIM-o (WavLM, the standard), CosyVoice's SS, IndexTTS-2's FunASR
embeddings and IN-F5's S-SIM are four different metrics. Read for orders of
magnitude.

| Model | Family | Params | RTF | First-chunk latency | Training hours | SECS / SIM | CER / WER | Licence |
|---|---|---|---|---|---|---|---|---|
| **KittenTTS 2** *(incumbent)* | AR codec LM | 1.7B ternary + 325M | **0.78** (A100) | **17 s** (A100, 380-char chunk) | undisclosed | — | ppl en 60.9 · hi 68.3 · **ta 287.2** **[ours]** | Stellon Community — terminates >$1M |
| **IndicF5 / IN-F5** | NAR flow | **337M** | ~0.15 @16 NFE † | ≈ whole utterance | **1,417 h / 11 Indic langs** | S-SIM 97.3 (human 97.0) ‡ | WER 19.2 (human 18.4) ‡ | **MIT claimed** — NC upstream, §6 |
| F5-TTS Base v1 | NAR flow | 336M | 0.15 @16 · 0.31 @32 · **0.0394** served (L20) | 253 ms avg (L20, conc. 2) | 95–100K h (Emilia) | SIM-o 0.66–0.67 | WER 2.42 LS-PC / 1.83 Seed-en | **CC-BY-NC-4.0** |
| F5-TTS *small* | NAR flow | **155M** | — | — | **945 h** (WenetSpeech4TTS) | — | — | CC-BY-NC-4.0 |
| OpenF5-TTS | NAR flow | ~336M | (= F5-TTS) | (= F5-TTS) | Emilia-YODAS 114K h | *alpha; "inferior to official"* | — | **Apache-2.0** |
| ZipVoice | NAR flow | **123M** | ~30× faster than F5-TTS | — | 100K h (Emilia) | "comparable to F5-TTS" | — | UNVERIFIED |
| E2-TTS | NAR flow | 333M | 0.68 @32 NFE | — | 95–100K h | SIM-o 0.69 | WER 2.95 LS-PC | CC-BY-NC-4.0 |
| MegaTTS 3 | NAR latent diff | 0.45B | 0.124 (PeRFlow, 8 steps) | — | — | SIM-o 0.71 | WER 1.82 LS-clean | **encoder withheld → cloning impossible** |
| CosyVoice 2 | Hybrid AR→NAR | 0.5B | — | **formula only in paper** § | 166.8K h | SS 0.748 zh / 0.652 en | CER 1.45 zh / WER 2.57 en | Apache-2.0 |
| Fun-CosyVoice3 | Hybrid | 0.5B | — | "as low as **150 ms**" bi-streaming | 1M h | SIM 78.0 zh / 71.8 en | CER 1.16 zh / WER 2.02 en | Apache-2.0 |
| Spark-TTS | AR codec LM | 0.5B | 0.136 offline (L20, TRT) | **210 ms P50** (L20, named workload) | 102.5K h | SIM 0.672 zh / 0.584 en | CER 1.20 / WER 1.98 | **weights CC-BY-NC-SA-4.0** |
| Llasa | AR codec LM | 1B / 3B / 8B | — | — | 250K h | SIM-o 0.669 → 0.684 (1B→8B) | WER 3.22 → 2.97 en (1B→8B) | CC-BY-NC-4.0 |
| Orpheus | AR codec LM | 3B | — | ~200 ms (no GPU stated) | 100K+ h en | — | — | Apache-2.0, gated · **has a Hindi ckpt** |
| FireRedTTS-2 | AR dual-tf | unstated | — | 140 ms (L20) / "<100 ms" (paper) | 1.4M h | SIM 0.736 zh / 0.665 en | CER 1.14 / WER 1.95 | Apache-2.0 |
| IndexTTS-2 | Hybrid AR→NAR | unstated (2.5 = 0.8B) | 0.326 fp16 (4090) | — | 55K h | SS 0.865 (FunASR, *not* SIM-o) | CER 1.008 / WER 1.521 | bilibili — commercial <100M MAU |
| Higgs Audio v2 | AR + adapter | 5.77B | — | — | **10M+ h** | SIM 67.70 | Seed-TTS WER 2.44 | Boson Community — <100k AAU |
| **dots.tts** | **Continuous AR + flow head** | 2B | — | **54 ms** dual-streaming | ~1.5M h | **SIM 81.0 / 77.1 / 79.5** | **CER 0.94 / WER 1.30 / hard 6.60** | **Apache-2.0** · Hindi flagged high-WER |
| **VoxCPM2** | Hybrid AR + LocDiT | 2B | **0.13** (4090, Nano-vLLM) | streaming, no number | 2M+ h | SIM 79.5 zh / 75.3 en · **hi 85.6** | CER 0.97 / WER 1.84 · **hi WER 19.7** | **Apache-2.0** · Hindi only |
| GEPARD 1.0 | AR codec LM | **556M** | **0.067** single-stream (5090) | **~50 ms TTFA** (5090, vLLM) | unstated | SIM 0.585 | WER 0.036 (own protocol) | Apache-2.0 · no Indic |
| **Veena** | AR + SNAC | 3B (HF says 4B) | 0.05 self-reported | **<80 ms** H100 (self-reported) | **unpublished** | "92%" — no methodology | none published | **Apache-2.0** · **hi+en, ta/te/bn/mr on roadmap** |
| Indic-Speak | AR + SNAC | 3B (Llama-3.2) | — | ~200 ms | undisclosed | consistency 96.5 | **WER 26.5 / CER 17.9** | Indic Open Model v1.0 · **no cloning** |
| Hindi student (distilled IndicF5) | NAR flow, pruned | **190M** | 3.13 (Apple M4) | — | 17.6 h *synthetic* | SIM 0.750 (teacher 0.784) | CER 0.077 (teacher 0.030) | unstated |
| Sarvam Bulbul *(target)* | closed | — | — | — | — | — | LLM-WER 0.007 hi / 0.029 te | commercial |

† IndicF5 is F5-TTS Base at the same size, so F5-TTS's RTF should transfer;
**IndicF5's own RTF on a datacentre GPU is UNVERIFIED** — the only published figure
is 5.56 on an Apple M4.
‡ IN-F5's numbers are on its own IN11 test set with IndicConformer and a WavLM
cosine, on a scale where human recordings score 97.0 and 18.4. **Not comparable to
SIM-o or Seed-TTS-eval.**
§ **Correction worth carrying:** the widely-quoted "CosyVoice 2 = 150 ms" is not in
the CosyVoice 2 paper, which gives only a latency *formula*. The 150 ms claim is
attached to Fun-CosyVoice3's bi-streaming checkpoint.

---

## 4. Findings

### 4.1 Data efficiency comes from the prior, not from the architecture

Three independent results, all pointing the same way.

**(a) F5-TTS on Korean, 12.65 h, one TITAN RTX** — the table in §1. From scratch:
CER 17.345, WER 45.883, NMOS 2.64, and the authors report the model "often
resulting in misalignment between text and speech." Pretrained and LoRA-adapted on
1.72% of parameters: CER 4.906, NMOS 3.88 — *better than ground truth on CER*,
because the ground-truth reading is itself hard for Whisper.

**(b) IndicF5's own data-scaling ablation**, 150K steps, same architecture and
hyperparameters throughout:

| hours/language | MUSHRA | SSIM-seen | SSIM-unseen | WER monoglot | WER polyglot |
|---|---|---|---|---|---|
| 100 h | 64.3 | 93.1 | 93.9 | 32.6% | 30.5% |
| **10 h** | **61.5** | **93.1** | **94.0** | **31.3%** | 31.8% |
| 1 h | 33.7 | 92.7 | 93.5 | 59.4% | 60.4% |

10 h/language "retains key emergent behaviors … with an average reduction of 0.8%
in performance across metrics." Between 10 h and 1 h, MUSHRA falls 28 points and
WER doubles. Note that **SSIM stays above 92 even at 1 h** — the English prior
supplies speaker modelling essentially for free, across languages.

**(c) The same paper's strategy comparison**, which is the cleanest statement of
the principle:

| strategy | overall MUSHRA |
|---|---|
| Human recordings | 77.4 |
| **EN→IN** (fine-tune English ckpt on Indic only) | **73.4** |
| EN→EN+IN (fine-tune on Indic *and* English) | 66.2 |
| Φ→IN (**from scratch** on all 1,417 h) | **43.2** |

Training the identical 337M model from scratch on 1,417 hours gives MUSHRA 43.2.
The paper's own framing: fine-tuning on "1.4% of EN data is sufficient to reach
human-level synthesis."

**And the counter-evidence, which is real.** NVIDIA's Align2Speak does the same
trick on an *autoregressive* codec LM. Koel-TTS (AR decoder over NanoCodec at
21.5 FPS), pretrained on ~21K h, adds Polish, Portuguese and **Hindi** from
**30 min / 1 h / 5 h** subsets: "with only 30 minutes of Portuguese data, CER drops
from 33.00% in the baseline to 3.94% after fine-tuning and GRPO," and "models
trained with just 30 minutes of supervision already outperform the baseline by a
wide margin." The per-language Hindi numbers are in a figure, not a table, so the
**Hindi-specific figures are UNVERIFIED**.

**Conclusion, stated carefully.** Both families adapt to a new language on tens of
hours *given a strong multilingual base*. Both collapse without one. The widely
repeated claim "NAR flow matching is more data-efficient than AR speech-LMs" is
**not directly demonstrated anywhere I could find** — there is no hours-matched
head-to-head. What *is* circumstantially true is that competitive flow-matching
models train on ~100K h while the AR models in §3 sit at 100K–10M h, and that
P-Flow claimed "two orders of magnitude less training data" for speaker
similarity — but that is 2023, English-only, and confounded by the fact that AR
models are buying expressivity, dialogue and streaming, not only intelligibility.

**What this means for us:** stop treating this as an architecture question and
treat it as a *base-model* question. Which available open base has the best Indic
prior, at the smallest size, with the best inference profile, under a clean
licence? That question has a much clearer answer (§5).

### 4.2 Adding English data *hurt* Indic quality

EN→EN+IN scored 66.2 against EN→IN's 73.4 — mixing English back in during Indic
fine-tuning cost 7 MUSHRA points. The paper recommends keeping the original
English checkpoint for English and the Indic fine-tune for Indic.

**This cuts against D1's reasoning** **[ours]**, which keeps English recordings in
every training phase to protect English. D1's concern was *tokeniser fertility* —
Sarvam makes English ~14% more expensive — and IndicF5 has no BPE tokeniser at
all, just 685 raw characters. So D1 is not wrong; it is scoped to the AR arm. But
the English-retention strategy has to be re-decided per family, and in this family
the published answer is "route to two checkpoints."

### 4.3 IndicF5 is the best open Indic base — and it is not Sarvam Bulbul

The brief treats "IndicF5 / ElevenLabs / Sarvam Bulbul" as one quality tier. They
are not.

| system | Hindi LLM-WER | Telugu LLM-WER |
|---|---|---|
| ElevenLabs v3 | 0.006 | 0.041 |
| Sarvam Bulbul | 0.007 | 0.029 |
| Cartesia Sonic-3 | 0.025 | 0.029 |
| **IndicF5 (zero-shot)** | **0.125** | **0.079** |
| Indic Parler-TTS | — | 0.144 |

Roughly **18× worse than Bulbul on Hindi** on this protocol. Caveats that matter:
the source is the Praxy Voice paper, whose authors ship a competing system; n=10–20
per system; and LLM-WER is a Qwen-2.5-72B semantic judge, not literal WER. A
different protocol (50 FLEURS Hindi sentences, Vakyansh wav2vec2) puts IndicF5 at
WER 0.098 / CER 0.030 / SIM 0.784 / UTMOS 3.79 — much better, and not comparable.

For scale, the only frontier model publishing a Hindi number is VoxCPM2: **WER
19.699 on Minimax-MLS against ~1.84 on English.** A 10× Indic gap in a 2B model
trained on 2M hours. Indic is genuinely hard, for everyone, and nobody open is
near Bulbul.

**Read this as: IndicF5 is the best open starting point, not a finished product.**
It buys a strong Indic prior and a 5× speed-up. It does not buy Bulbul-class
intelligibility. That gap is what our 150–200 h/language is for.

### 4.4 IndicF5's weaknesses, which we would inherit

- **English inside Indic text is effectively dropped.** Its character tokeniser
  accepts Latin, but its pretraining corpus has no Latin-script *audio*. Measured
  code-mix LLM-WER **0.80–0.85** without preprocessing. For a product that must do
  "Hindi + English," this is a first-class problem, not polish.
- **Tamil is its thinnest language**, ~80 h — cited as the reason for weak Tamil
  code-mixing. Tamil is also *our* worst language **[ours]**. Both systems are weak
  in the same place; switching base does not fix Tamil, it only changes what we
  fine-tune from.
- **Reference audio is truncated to ~12 s.** F5-TTS clips long references (README
  "<12s"; community reports a 15 s hard cut). Our spec asks for **20–30 s**, and
  KittenTTS 2's card explicitly accepts "5–30 seconds". This is a real capability
  the incumbent has and the recommendation does not.
- **The reference transcript is required** at inference. We already ship a
  `ReferenceTranscriber`, and Cross-Lingual F5-TTS 2 removes the requirement
  entirely — so it is solvable, but it is work. (That paper's data requirements and
  language list are **UNVERIFIED**; read at abstract level only.)
- **AI4Bharat now point elsewhere.** The IndicF5 card was updated ~10 days ago to
  recommend Indic-Speak — a 3B Llama-3.2 AR model over SNAC codes that **does not
  support voice cloning at all**, and whose intelligibility (WER 26.5 / CER 17.9)
  is not obviously better. It is disqualified for us on requirement 3. The redirect
  is a signal about AI4Bharat's roadmap, not evidence that IndicF5 is a bad base.

### 4.5 Our 17 s first-chunk latency is mostly a chunking policy

Worth being honest about, because it is the metric we most want to beat and the
headline comparison overstates the win.

Our README records: on an A100, a 936-character passage "yielded its first 21 s of
audio after 17 s" **[ours]**. At RTF 0.78, 21 s of audio costs 16.4 s to generate.
**The 17 s is almost exactly the cost of the first chunk being 21 seconds long.**
The chunker splits at ~380 characters; shrink that to ~95 (~5 s of audio) and first
audio arrives in ~4 s, with no architecture change at all.

| | first audio for a 5 s sentence |
|---|---|
| KittenTTS 2 today, 380-char chunks | ~17 s |
| KittenTTS 2 with a 95-char chunker | **~4 s** |
| IndicF5 / F5-TTS, no chunking needed | **~0.8 s** |
| Spark-TTS, measured on a named GPU and workload | **0.21 s** |
| dots.tts dual-streaming | **0.054 s** |

NAR still wins by ~5× over a fixed AR chunker, and wins without any chunking
machinery. But a reader who believes the architecture alone is worth 17 s → 1 s has
been misled, and the cheapest latency fix available today needs no backbone change.

Note also where the floor is: **the genuinely low-latency numbers all belong to AR
and hybrid models** (dots.tts 54 ms, GEPARD 50 ms, Spark-TTS 210 ms). Non-causal
flow matching has no path below "one utterance."

### 4.6 Our decoder stack *is* CosyVoice's — which makes the fallback cheap

`kittenml/kittentts2/s3gen.py` is vendored from ResembleAI Chatterbox (Apache-2.0),
"which in turn builds on Alibaba's CosyVoice and Matcha-TTS", with
`SPEECH_VOCAB_SIZE = 6561` at 25 Hz **[ours]**. CosyVoice 2's speech tokeniser is
FSQ with a codebook of **6,561** at **25 Hz**, feeding 50 Hz mel at 24 kHz.

Same token space. **KittenTTS 2 is structurally CosyVoice 2 with a ternary
Qwen3-1.7B in place of Qwen2.5-0.5B.** Three consequences:

1. **CosyVoice 2's chunk-aware causal flow decoder is a drop-in**, not a rewrite:
   replace the decoder, interleave text/speech tokens at a fixed N:M ratio.
2. **Our LM is 3.4× larger than CosyVoice 2's for no demonstrated benefit.**
   CosyVoice 2 at 0.5B reaches WER 2.47 / SS 0.745 on LibriSpeech test-clean. And
   Llasa's scaling study shows 1B→8B is nearly flat (en WER 3.22→2.97), with
   inference-time search beating scale outright (1B + PRM/ORM = 1.46 beats 8B base
   = 2.97). **Nothing in the literature supports 1.7B for this task.**
3. The repo already ships a distilled single-step `student_flow` in that same token
   space — though its own docstring warns it is "**not faster** without a kernel
   that multiplies in low precision" and costs quality (mel L1 0.250 vs 0.168)
   **[ours]**.

### 4.7 Distillation is a second-order lever, not a decision

Few-step distillation reliably buys 2–4× at a measurable quality cost, in every
family:

- **Fast F5-TTS** (Empirically Pruned Step Sampling): RTF **0.123 → 0.030**, WER
  1.70 → 1.74 (test-en).
- **MegaTTS 3 + PeRFlow**: 25 steps → 8, RTF 0.124, SIM-o 0.70 vs 0.71.
- **dots.tts + MeanFlow**: an NFE=4 variant at WER 1.29 / SIM 76.2 en.
- **Depth-pruning IndicF5** 337M → 190M: 1.8× faster, keeps ~96% of UTMOS and
  speaker similarity, but CER degrades 0.030 → 0.077.
- **ZipVoice**: 123M flow matching vs F5-TTS's 336M on the same 100K h Emilia —
  3× smaller, up to 30× faster, "comparable quality." Worth a look as an
  architecture for the clean-licence rebuild.

None of these changes which family to pick; all are available *after* the choice.
The one finding to carry: the Hindi distillation paper reports its student trained
**from random initialisation on 17.6 h failed outright** (unintelligible), and that
warm-starting from the teacher is what made it work — §4.1 again.

---

## 5. The direct answer: keep KittenTTS 2, or switch?

**Switch the backbone. Keep the harness.**

### What switching costs

The instinct to protect the KittenTTS 2 investment is right but mis-locates where
the investment is:

| Asset | Survives a switch? |
|---|---|
| Preflight orchestrator + overnight driver | **Yes** — architecture-independent |
| Eval harness (Set A/B/C, floor CER, WavLM SECS, D8–D11) | **Yes** — already has an `IndicF5Synthesizer` |
| HPC portability layer, Slurm templates, shard reader, preprocessing job | **Yes** |
| Danda / NFC / Indic-digit text fixes (D4) | **Yes** |
| Data acquisition plan (~900 h, IndicVoices-R, Rasa) | **Yes** — the same corpora IndicF5 used |
| Licensing analysis | **Yes**, and it gets sharper (§6) |
| Codec round-trip gate (§3.6) | **No** — IndicF5 predicts mel directly, no S3 codec |
| Sarvam vocabulary surgery (D1, D2, §3.5) | **No** — 685 raw characters, no BPE |
| Ternary QAT work (D6, §3.2, §3.3) | **No** |
| 12 GB memory fit, `reserved` basis (D3, §3.4) | **Partly** — the method survives, the numbers do not |
| Teacher-forced loss gate (§3.7) | **No** — no teacher forcing in flow matching |
| `spk_proj` keyword-argument trap (§4d) | **No** — no speaker projection |
| fp16 activation analysis (D12, §4c) | **No** |

Roughly **60% of the week's work carries over untouched.** The 40% that does not
is, by the findings doc's own account, the part that was hardest *because the
backbone is awkward*: a quantiser whose training rule "cannot be identified from
the checkpoint alone", a reference layout that is **NOT CONFIRMED**, a
speaker-conditioning hook that "silently does nothing" on the obvious call, and a
card that fits only LoRA.

That last one flips from a constraint to an argument: PEFT-TTS adapted F5-TTS to a
new language with **5.81M trainable parameters on a single TITAN RTX**. Our 12 GB
card is not a limitation in the recommended path — it is the published setup.

### The case for keeping KittenTTS 2, stated as strongly as it deserves

- **The codec gate passed, and passed well.** Every Indic language reconstructs
  *better* than English through S3 **[ours]**. Real, measured, and it removed the
  project's headline risk.
- **Arbitrary-length reference conditioning.** x-vector → `spk_proj` takes 5–30 s,
  matching our spec. F5-TTS truncates at ~12 s.
- **Token-level streaming is natural** for AR; a non-causal DiT cannot stream at
  all. §4.5 shows every sub-250 ms number in this document belongs to AR or hybrid.
- **AR/hybrid holds the quality frontier.** dots.tts (2B, Apache-2.0) at CER 0.94 /
  WER 1.30 / SIM ~79, VoxCPM2 at 0.97 / 1.84 / 79.5, against F5-TTS's 1.53 / 2.00 /
  76.0.
- **Align2Speak shows AR adapting from 30 minutes** (§4.1), so the data-efficiency
  argument is not the slam-dunk it first appeared.

### Why they lose anyway

- The codec gate says *the decoder* is fine. It says nothing about whether a 1.7B
  ternary AR LM learns Tamil from 150 h. Our measurement on that is **Tamil
  perplexity 287.2, 4.7× English** **[ours]**, in the language where Qwen3 fertility
  is worst (10.13 tokens/word). IndicF5 has already *trained on* Tamil.
- The 12 s reference limit is a real loss, but 12 s is empirically enough: IN-F5
  holds SSIM > 92 on unseen speakers; the distilled Hindi student holds SIM 0.750.
- Token-level streaming is natural for AR but **we do not have it** — our streaming
  is explicitly "chunk-level, not token-level" **[ours]**. Every low-latency figure
  cited belongs to a chunk-aware causal flow decoder we would have to build either
  way.
- The frontier-quality models train on 1.5M–10M hours. That does not transfer to a
  <1000 h Indic budget, which is the regime §4.1 shows is dominated by the prior.
- Align2Speak's 30-minute result required a **21K h multilingual base plus
  ASR-guided GRPO**. We have neither: KittenTTS 2's training data is undisclosed,
  its Tamil perplexity is 287.2, and nobody has built the GRPO pipeline. The result
  argues that *an AR model with the right base* is data-efficient. It is not an
  argument for *this* AR model.

### And the one that settles it

**KittenTTS 2's licence has an expiry date built in.** The Stellon Labs Community
License terminates above $1M revenue or $1M cumulative funding, requires "Powered
by Stellon Labs" to be displayed, and §4b forbids using the model or its outputs to
train other models **[ours]**. For a project whose brief says "commercial use
intended; licences are a hard filter," this is a backbone we must leave anyway. The
only question is whether we leave now or after spending the allocation.

---

## 6. Licensing — the sharpest finding in this document

The licence chain for the recommended base does not close cleanly.

```
Emilia (101.7K h)             CC-BY-NC-4.0  "non-commercial purposes only"
  └─> F5-TTS Base v1 weights  CC-BY-NC-4.0  (code is MIT; the weights are not)
        └─> IndicF5 / IN-F5   listed MIT on HuggingFace    <-- conflict
```

The IN-F5 paper states that it fine-tunes "starting from the English checkpoint
pretrained on nearly 100K hours," and the F5-TTS repository states its pre-trained
models are CC-BY-NC "due to the training data Emilia." AI4Bharat's MIT label does
not visibly extinguish an upstream non-commercial term that the upstream authors
attached deliberately.

**This is a legal question and it is UNVERIFIED in both directions.** The F5-TTS
discussion thread records a community reading that the Emilia team may not consider
trained models to inherit NC — a user's opinion, not a published position.

**The escape hatch exists.** Every ingredient of the IN-F5 recipe has a permissive
substitute:

| Ingredient | NC version | Permissive substitute |
|---|---|---|
| Pretrained base | F5-TTS Base v1 (CC-BY-NC) | **`mrfakename/OpenF5-TTS` — Apache-2.0**, F5-TTS Base V1 config, 1M steps |
| Pretraining corpus | Emilia (CC-BY-NC) | **Emilia-YODAS, 113.9K h, CC-BY-4.0** |
| Indic corpus | — | IndicVoices-R (CC-BY-4.0), Rasa (CC-BY-4.0), IndicTTS, LIMMITS |
| Recipe | — | published in full: 150K steps, AdamW 5e-5, 30K frames/GPU, 48K warm-up, 32×H100 |

OpenF5-TTS is self-described **alpha**, English-only, "still inferior to the
official NC-licensed F5-TTS model," and publishes no benchmarks. Its quality is
**UNVERIFIED**, and it is the main risk in the shipping plan — the reason the
research base and the shipping base are deliberately different models.

Full licence survey, since this is a hard filter:

| Model | Licence | Verdict for a commercial product |
|---|---|---|
| OpenF5-TTS, CosyVoice 2/3, VoxCPM2, dots.tts, GEPARD, FireRedTTS-2, **Veena** | Apache-2.0 | **clean** |
| Orpheus | Apache-2.0, gated | clean, but access-controlled |
| IndicF5 | MIT claimed, NC upstream | **research only** until resolved |
| KittenTTS 2 | Stellon Community | terminates at $1M; **outputs cannot train other models** |
| Higgs Audio v2 | Boson Community | commercial until >100k annual active users |
| IndexTTS-2 | bilibili Model Use | commercial until >100M MAU / RMB 1B; Chinese law |
| Indic-Speak | Indic Open Model v1.0 | commercially usable, but **no voice cloning** |
| **Spark-TTS** | **weights CC-BY-NC-SA-4.0** | code is Apache-2.0; **the weights were relicensed away from it** |
| Llasa (all sizes), Fish S1-mini | CC-BY-NC(-SA)-4.0 | non-commercial |
| MegaTTS 3 | encoder withheld | **cloning impossible** — disqualified |
| Sarvam-1 tokeniser | non-commercial | already flagged **[ours]** |

Two traps worth naming: **Spark-TTS's weights were relicensed from Apache-2.0 to
CC-BY-NC-SA-4.0** after release, because of its training data — exactly the failure
mode IndicF5 may be in. And **HuggingFace parameter counts are inflated** by codec
embedding tables (Llasa-3B reports 4B, Veena-3B reports 4B); use paper figures for
architecture and HF figures for VRAM.

---

## 7. The cheapest experiment that decides this

**Run IndicF5 through the eval harness we already built. No training, no new
code.**

`indic/eval/run_eval.py` already contains an `IndicF5Synthesizer` that loads
`ai4bharat/IndicF5` via `AutoModel.from_pretrained(..., trust_remote_code=True)`,
handles the int16/float return-type trap, and refuses records without a reference
clip. The only blocker is the one the codec gate hit: AI4Bharat's repositories are
gated and this machine has no `HF_TOKEN`.

```
python -m indic.eval.run_eval --model ai4bharat/IndicF5 --testset A --lang ta \
    --reference-from fleurs --out runs/indicf5-A-ta
python -m indic.eval.score runs/indicf5-A-ta
```

Run it for `hi, bn, ta, te, mr, en`. Because of D9 and D10 **[ours]** this yields
`cer_above_floor` against the same recognizer and a WavLM SECS from references that
are *not* the target sentence — a number on **our** protocol, not a paper's. Record
wall-clock while you are there: it closes the UNVERIFIED IndicF5-RTF gap for free.

**The decision rule, written before the data:**

- **Switch, as recommended** — IndicF5 zero-shot already beats what a KittenTTS 2
  fine-tune could plausibly reach, in particular if Tamil `cer_above_floor` is low
  while our Tamil perplexity sits at 287.2 **[ours]**.
- **Go to the fallback** — SECS < 0.5 on WavLM, or Tamil CER no better than the ASR
  floor. The Indic prior is then not the asset §4.1 claims, and legs 1 and 3 of the
  recommendation both fail at once.
- English catastrophic (expected, §4.4) — does not change the verdict, but settles
  the routing decision: two checkpoints, not one.

Cost: hours of wall clock on one GPU. **This replaces the "highest-value remaining
hour" the findings doc currently nominates.**

The tie-breaker, only if the first run is ambiguous: fine-tune IndicF5 and
KittenTTS 2 on the *same* 10 h Tamil slice and score both. §4.1 says 10 h is enough
to see the difference, so even the tie-breaker is cheap — and it would be the
hours-matched AR-vs-NAR comparison that §4.1 says nobody has published.

---

## 8. Fallback plan

**If this is wrong, it will be wrong in one of two specific ways.**

### Failure mode A — streaming is a hard requirement

NAR flow matching cannot stream. A 5 s sentence arrives in ~0.8 s: fine for
narration, marginal for conversation. If the product needs <300 ms first-packet,
§4.5 shows no amount of NAR tuning gets there.

**Fall back to the CosyVoice 2 hybrid, on our existing LM.** §4.6 establishes that
our token space is already CosyVoice's — 6561 FSQ tokens at 25 Hz. Replace the flow
decoder with a chunk-aware causal one and interleave text/speech tokens at N:M.
This keeps the ternary checkpoint, the codec gate, the vocabulary surgery and the
QAT work — the entire 40% a switch discards. Apache-2.0, WER 2.47 / SS 0.745
LibriSpeech test-clean.

If we are rebuilding for streaming anyway, **dots.tts** is the better target: 2B,
Apache-2.0, **54 ms dual-streaming**, CER 0.94 / WER 1.30 / SIM ~79 — the best
open numbers in this document — using continuous AR latents with a flow-matching
head rather than discrete codec tokens. Hindi is flagged as higher-WER on its own
card, so Indic work would still be ours to do.

Cost of being wrong this way: the IndicF5 evaluation plus whatever fine-tuning had
started. Weeks, not months.

### Failure mode B — IndicF5's Indic prior is weaker than its paper suggests

§4.3 already shows an 18× Hindi gap to Bulbul on one protocol. If our own
evaluation confirms IndicF5 is not materially ahead of where a KittenTTS 2
fine-tune would land, the "strong Indic prior" premise fails.

**Fall back to a clean-licence base with real Indic coverage**, ranked:

1. **Veena** (Maya Research) — Apache-2.0, **Hindi + English including code-mixed**,
   with Tamil/Telugu/Bengali/Marathi on the published roadmap. It is already a LoRA
   fine-tune (attention rank 192, FFN rank 96, 4-bit NF4 base, 8×H100), which makes
   it evidence about adaptation cost rather than pretraining cost — and matches our
   hardware. **But it publishes no training hours, no WER and no Seed-TTS-eval
   numbers**; its "92% speaker similarity / 98% intelligibility / <80 ms" are
   self-reported with no stated methodology. Everything about Veena's quality is
   **UNVERIFIED** and would need the §7 harness run before it could be trusted.
2. **VoxCPM2** — Apache-2.0, RTF 0.13 on one RTX 4090, streaming, SIM 75.3 / WER
   1.84 en, 2M training hours, and it publishes a real Hindi number (WER 19.699 /
   SIM 85.6). **Hindi only** among our five, and 2B is larger than our efficiency
   goal wants. Its card claims fine-tuning from 5–10 minutes of audio.
3. **Fun-CosyVoice3 0.5B** — Apache-2.0, best open quality at the size we want
   (CER 1.16 zh / WER 2.02 en), 150 ms bi-streaming. No Indic coverage at all.
4. **Stay on KittenTTS 2 and spend the allocation**, accepting Tamil ppl 287.2 as
   the start point and the Stellon licence as a future migration.

### The tripwire

Both failure modes are detected by the same experiment in §7, which is why it runs
first. The thing that would make this document wrong and *not* be caught is a
licence ruling — so treat §6 as a parallel track with a different owner.

---

## 9. Decisions proposed

Numbering continues from the findings doc, which ends at D12.

**D13 — Move the backbone to NAR flow matching.** Not because NAR is inherently
data-efficient (§4.1 says that is unproven), but because a 336M flow-matching model
adapts to a new language on ~12 h and one consumer GPU, runs ~5× faster than what
we have, and IndicF5 is the only open base with a real prior across all five of our
languages.

**D14 — Research base is IndicF5; shipping base is a clean-licence rebuild.**
Deliberately different models. Use IndicF5 to learn and to measure; re-run the
published EN→IN recipe from OpenF5-TTS on CC-BY-4.0 Indic data for anything that
ships. Do not let the two converge by accident.

**D15 — Run the IndicF5 evaluation before any further training.** Already
implemented; needs only an `HF_TOKEN`. Any week spent fine-tuning either backbone
before this number exists is a week spent on a guess.

**D16 — Re-decide English handling per family.** D1 keeps English in every phase.
IN-F5 measured that strategy costing 7 MUSHRA points and recommends routing English
to a separate checkpoint. D1's reasoning was about tokeniser fertility, which does
not apply to a 685-character vocabulary — so D1 is not wrong, it is scoped to the
AR arm.

**D17 — Fix the chunker before blaming the architecture for latency.** §4.5 shows
17 s is ~21 s of audio at RTF 0.78. A smaller first chunk buys ~4× with no backbone
change, and it makes the AR arm an honest comparison rather than a strawman. Do
this even if D13 proceeds.

**D18 — Treat the Emilia→F5→IndicF5 licence chain as a blocking, parallel track.**
Unresolved in both directions, and the single thing most likely to invalidate the
shipping plan. Spark-TTS's weights being relicensed from Apache-2.0 to CC-BY-NC-SA
after release is the precedent to worry about.

**D19 — Do not scale the LM.** Llasa 1B→8B moves English WER 3.22→2.97 while
inference-time search on the 1B reaches 1.46. CosyVoice 2 at 0.5B matches models
3× its size. Our 1.7B is unsupported by any published scaling evidence for this
task, and our target is efficiency.

---

## 10. Honest gaps

- **No hours-matched AR-vs-NAR study exists.** The central premise of the brief —
  that NAR flow matching is more data-efficient — is **not directly demonstrated
  anywhere I found**. Align2Speak is active counter-evidence. §7's tie-breaker
  would be the first such comparison, which is itself a reason to run it.
- **IndicF5's RTF on a datacentre GPU is UNVERIFIED.** The only published figure is
  5.56 on an Apple M4. We infer F5-TTS's 0.15 @16 NFE from identical architecture
  and size, but have not measured it.
- **OpenF5-TTS's quality is UNVERIFIED.** Self-described alpha, no benchmarks, its
  own card says it is inferior to the NC checkpoint. The clean-licence shipping plan
  rests on it being good enough, or on us training a replacement.
- **Veena's everything is UNVERIFIED** — no hours, no WER, no methodology for its
  self-reported figures. It is ranked first in fallback B purely on licence plus
  Indic roadmap, which is a weak basis.
- **Align2Speak's Hindi numbers are in a figure, not a table**, so the
  Hindi-specific data points are UNVERIFIED; only the Portuguese 33.00% → 3.94% is
  quotable.
- **Per-language hours for IN-F5 are not published** — only 1,417 h across 11
  languages, plus a third-party ~80 h figure for Tamil. Our per-language budget
  cannot be benchmarked against theirs precisely.
- **The licence question (§6) is unresolved** and is the largest single risk.
- **No MOS or listening test in this document is ours.** Every quality number is
  self-reported by a system's authors or by a competitor. §7 produces the first
  number on our protocol.
- **Seed-TTS-eval has no Indic split**, and its similarity column hides at least
  four different metrics across rows. `test-hard` is omitted by Spark-TTS,
  FireRedTTS-2, IndexTTS-2 and Higgs — it is the discriminating split and its
  absence is rarely accidental.
- **Cross-Lingual F5-TTS 2 and ZipVoice** were read at abstract level only; their
  data requirements, language lists and licences are UNVERIFIED.
- **The web-search budget for this session was exhausted** partway through. A few
  targeted queries — Sarvam's and Krutrim's own 2026 Indic releases, Fish S1
  first-packet latency — could not be run, so the Indic commercial landscape in §4.3
  may be incomplete.

---

## 11. Sources

**IndicF5 / IN-F5**
- https://arxiv.org/abs/2505.20693 · https://arxiv.org/html/2505.20693v1
- https://huggingface.co/ai4bharat/IndicF5

**F5-TTS family and adaptation**
- Paper: https://arxiv.org/html/2410.06885v1 · https://aclanthology.org/2025.acl-long.313/
- Repo, runtime benchmarks, licences: https://github.com/SWivid/F5-TTS
- Weights (CC-BY-NC): https://huggingface.co/SWivid/F5-TTS
- Licence discussion: https://github.com/SWivid/F5-TTS/discussions/129
- **PEFT-TTS (Korean, 12.65 h, 1.72% params):** https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf
- Reference-length behaviour: https://github.com/niknah/ComfyUI-F5-TTS
- OpenF5-TTS (Apache-2.0): https://huggingface.co/mrfakename/OpenF5-TTS
- Fast F5-TTS / pruned step sampling: https://arxiv.org/pdf/2505.19931
- ZipVoice: https://arxiv.org/abs/2506.13053
- Cross-Lingual F5-TTS 2: https://arxiv.org/abs/2609.15184

**Data efficiency and new-language adaptation**
- **Align2Speak (AR, 30 min/language, incl. Hindi):** https://arxiv.org/html/2509.21718v1
- P-Flow: https://research.nvidia.com/labs/adlr/projects/pflow

**Data and licences**
- Emilia / Emilia-YODAS: https://huggingface.co/datasets/amphion/Emilia-Dataset
- IndicVoices-R: https://arxiv.org/html/2409.05356v2 · https://huggingface.co/datasets/ai4bharat/indicvoices_r
- Rasa: https://arxiv.org/html/2407.14056v2 · https://huggingface.co/datasets/ai4bharat/Rasa

**Hybrid / AR**
- CosyVoice 2: https://arxiv.org/html/2412.10117v3 · https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- CosyVoice 3: https://arxiv.org/html/2505.17589v2 · https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- Llasa (scaling): https://arxiv.org/html/2502.04128v2 · https://huggingface.co/HKUSTAudio/Llasa-3B
- Spark-TTS: https://arxiv.org/html/2503.01710v1 · https://huggingface.co/SparkAudio/Spark-TTS-0.5B · https://github.com/SparkAudio/Spark-TTS/tree/main/runtime/triton_trtllm
- Orpheus: https://github.com/canopyai/Orpheus-TTS · https://huggingface.co/collections/canopylabs/orpheus-multilingual-research-release-67f5894cd16794db163786ba
- FireRedTTS-2: https://arxiv.org/html/2509.02020v2 · https://huggingface.co/FireRedTeam/FireRedTTS2
- IndexTTS-2: https://arxiv.org/html/2506.21619v2 · https://github.com/index-tts/index-tts
- Higgs Audio v2: https://github.com/boson-ai/higgs-audio
- **dots.tts:** https://arxiv.org/abs/2606.07080 · https://huggingface.co/dots-studio/dots.tts-soar
- **VoxCPM2:** https://github.com/OpenBMB/VoxCPM · https://huggingface.co/openbmb/VoxCPM2
- GEPARD 1.0: https://arxiv.org/abs/2609.04222 · https://huggingface.co/nineninesix/gepard-1.0
- MegaTTS 3: https://arxiv.org/abs/2502.18924 · https://github.com/bytedance/MegaTTS3
- GLM-TTS survey table: https://arxiv.org/html/2512.14291

**Indic-specific**
- **Veena:** https://huggingface.co/maya-research/Veena · https://huggingface.co/maya-research/maya1
- Indic-Speak: https://huggingface.co/bodhan-ai/indic-speak
- Praxy Voice (Bulbul / ElevenLabs / IndicF5): https://arxiv.org/html/2604.25441v1
- PSP accent benchmark: https://arxiv.org/html/2604.25476
- Hindi depth-pruning distillation of IndicF5: https://arxiv.org/html/2607.18662
- Bengali IndicF5 fine-tune: https://huggingface.co/ehzawad/indicf5-bangla-tts

**Incumbent**
- https://huggingface.co/KittenML/kitten-tts-2
- https://betterstack.com/community/guides/ai/kitten-tts-2/
