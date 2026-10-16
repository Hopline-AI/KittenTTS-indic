# The backbone decision: keep the AR speech-LM, or move to flow matching?

**Session of 2026-10-16.** This document answers one question: what should the
central model be? Every claim carries a URL in §12. Numbers *we* measured this
week are marked **[ours]** and live in
`docs/indic/2026-10-16-findings-and-decisions.md`. Claims I could not verify are
marked **UNVERIFIED** and are not load-bearing.

---

## 1. Recommendation

**Move the backbone to F5-style non-autoregressive flow matching, with IndicF5 as
the research base starting now.**

But the single most important finding in this research is not about AR vs NAR at
all, and it changes how the whole project should be planned.

### 1.1 The finding that reframes the project

Voicebox ran the data-scaling experiment nobody else did — same model, same
recipe, 60 h to 60,000 h:

| training hours | zero-shot WER | zero-shot **SIM-r** |
|---|---|---|
| 60 | 2.30 | **0.151** |
| 600 | 2.11 | **0.417** |
| 6,000 | 2.08 | **0.573** |
| 60,000 | 2.05 | **0.645** |

**Intelligibility is almost data-independent — 2.30 → 2.05 across a 1000× range.
Zero-shot speaker similarity is almost entirely data-bound, and has not saturated
at 60,000 hours.**

Now put that next to IndicF5's ablation, where speaker similarity is *flat* at
SSIM 92.7–93.1 whether you fine-tune on 1 h or 100 h per language, while MUSHRA
and WER collapse between 10 h and 1 h.

The two tables reconcile exactly, and the conclusion is sharp:

> **Zero-shot cloning quality is bought by the pretrained base. Our 150–200 h per
> language buys Indic intelligibility and prosody, and buys essentially none of
> the voice cloning.**

Four independent papers confirm the price of ignoring this: any flow-matching
model trained from scratch on under ~1000 h lands at **SIM-o 0.58–0.61**, against
0.655–0.70 for 50–100K h models — while its WER is fine (1.5–2.4). F5-TTS's own
155M model on 585 h of LibriTTS reaches WER 2.20, *better than the 2.23 ground
truth*, at SIM-o 0.60.

**So the backbone question is really a base-model question**, and it should be
asked in this order:

1. Which available base has the best **zero-shot cloning** quality? (It must be
   pretrained on ≥50K h; nothing else is in the running.)
2. Of those, which has an **Indic prior**, a **clean licence**, and an **inference
   profile** that beats RTF 0.78?

### 1.2 Why that answer is a flow-matching model

| | |
|---|---|
| **Data efficiency** | F5-TTS adapts to a new language on **12.65 h**, one **TITAN RTX**, **1.72% of params**. IndicF5 needs **10 h/language**. Our 12 GB card fits LoRA and nothing else **[ours]** — which is exactly this regime. |
| **Inference** | RTF **0.15** @16 NFE vs our measured **0.78** **[ours]**. Free 4× available on day one (EPSS, training-free, 32→7 NFE, RTF 0.123 → **0.030**). A further 5× from 4-step distillation that *improves* both WER and SIM. |
| **Size** | 336M, or **123M** for ZipVoice, against our 1.7B ternary + 325M dense **[ours]**. |
| **Indic prior** | IndicF5 is the only open base with real coverage of all five of our languages. |

And the AR-vs-NAR comparison does exist, on matched data, contrary to the common
claim that it does not. DiFlow-TTS evaluates both on **the same ~500 h of
LibriTTS under one protocol**: F5-TTS SIM-O **0.52** against VALL-E (AR codec LM,
same 500 h) SIM-O **0.40**. IndicF5 beats VoiceCraft (AR) 80.5 vs 73.0 MUSHRA on
Indian languages. The NAR advantage at small data is real — about **+0.12 SIM-O**
— but it is one paper, one protocol, and it is not the main reason to switch.

### 1.3 The three caveats that must travel with this

**(a) AR is not inherently data-hungry.** NVIDIA's Align2Speak adds a new language
— including **Hindi** — to Koel-TTS, a genuinely autoregressive codec LM, from
**30 minutes**: Portuguese CER falls 33.00% → 3.94% with fine-tuning plus
ASR-guided GRPO. The determinant is base coverage plus alignment, not the AR/NAR
axis. This weakens the data-efficiency leg; it does not touch the inference or
licence legs.

**(b) Every sub-250 ms latency number in this document belongs to an AR or hybrid
model.** dots.tts 54 ms, GEPARD 50 ms, Spark-TTS 210 ms (measured, named GPU).
Non-causal flow matching has no path below "one utterance." If conversational
streaming is a hard requirement, this recommendation is wrong — see §9.

**(c) Do not ship IndicF5's weights.** It is listed MIT on HuggingFace and is, by
its own paper, a fine-tune of the English F5-TTS checkpoint, which its authors
released **CC-BY-NC-4.0** because of Emilia. Our brief makes licences a hard
filter. See §7.5.

### 1.4 What to do this week

The deciding experiment **is already written and costs no GPU-hours**.
`indic/eval/run_eval.py` already ships an `IndicF5Synthesizer`, and its module
docstring already names `--model ai4bharat/IndicF5` as a worked example. It needs
an `HF_TOKEN` and nothing else. See §8.

---

## 2. The recommended architecture, concretely

| | |
|---|---|
| Family | **NAR flow matching**, F5-TTS / DiT lineage |
| Research base **now** | `ai4bharat/IndicF5` — 337M DiT, 11 Indic languages, trained |
| Shipping base **later** | EN→IN run from `mrfakename/OpenF5-TTS` (Apache-2.0), §7 |
| Efficiency target | **ZipVoice's 123M** config — beats a retrained 155M F5-TTS on identical 555 h data (SIM-o 0.610 vs 0.584, WER 1.69 vs 1.78) |
| Adaptation | LoRA r=16 on DiT q/v + prompt adapter ≈ **5.81M trainable (1.72%)** |
| Audio | 100-mel @ 24 kHz → Vocos (13.6M); *replaces* S3 tokens + s3gen |
| Inference, day one | 16 NFE Euler, **+ EPSS → 7 NFE for a free 4×** |
| Inference, later | 4-step DMOSpeech-2-style distillation — *improves* WER **and** SIM |
| Cloning | In-context: reference audio + transcript prefixed to the mel sequence |

What changes: we stop predicting discrete codec tokens with a causal LM and start
predicting a mel spectrogram with a non-causal transformer that sees the whole
utterance at once. The reference voice stops being an x-vector injected at
position 0 and becomes *a prefix of the thing being generated*.

---

## 3. The comparison table

**RTF is not comparable across rows** — these papers use at least seven different
GPUs, and some include the vocoder while others do not. Nor are the similarity
columns: SIM-o (WavLM-TDNN, the standard), CosyVoice's WavLM SS, IndexTTS-2's
FunASR embeddings and IN-F5's S-SIM are four different metrics, and CosyVoice 2
reports WavLM 0.652 and ERes2Net 0.736 *on the same audio*. Read for orders of
magnitude.

| Model | Family | Params | RTF (GPU) | First-chunk latency | Training hours | SECS / SIM | CER / WER | Licence |
|---|---|---|---|---|---|---|---|---|
| **KittenTTS 2** *(incumbent)* | AR codec LM | 1.7B ternary + 325M | **0.78** (A100) | **17 s** (A100, 380-char chunk) | undisclosed | — | ppl en 60.9 · hi 68.3 · **ta 287.2** **[ours]** | Stellon Community — terminates >$1M |
| **IndicF5 / IN-F5** | NAR flow | **337M** | ~0.15 @16 NFE † | ≈ whole utterance | **1,417 h / 11 Indic langs** | S-SIM 97.3 (human 97.0) ‡ | WER 19.2 (human 18.4) ‡ | **MIT claimed** — NC upstream, §7 |
| F5-TTS Base v1 | NAR flow | 335.8M | 0.15 @16 (3090) · **0.0394** served (L20) | 253 ms (L20, conc. 2) | ~95K h Emilia | SIM-o **0.655** | WER **1.89** LS-PC | **CC-BY-NC-4.0** |
| F5-TTS *small*, from scratch | NAR flow | 155M | — | — | **585 h** LibriTTS | SIM-o **0.60** | WER **2.20** | CC-BY-NC-4.0 |
| F5-TTS *small*, from scratch | NAR flow | 155M | — | — | **945 h** Mandarin | SIM 0.54 | WER 4.17 (E2-TTS: **9.63**) | CC-BY-NC-4.0 |
| **ZipVoice** | NAR flow | **123M** | — | — | 100K h Emilia | SIM-o **0.668** | WER **1.64** LS-PC | UNVERIFIED |
| **ZipVoice-Distill** | NAR flow, 4 NFE | **123M** | **0.0125** (H20) = **23.7× F5-TTS** | — | + 62K distill steps | SIM-o 0.657 | WER **1.51** | UNVERIFIED |
| ZipVoice, from scratch | NAR flow | 123M | — | — | **555 h** LibriTTS | SIM-o **0.610** | WER 1.69 | UNVERIFIED |
| **DMOSpeech 2** | NAR flow, distilled | 0.3B | **0.0316** (H100) | — | F5-TTS teacher | SIM-o **0.698** *(teacher 0.662)* | WER **1.752** *(teacher 1.947)* | — |
| OpenF5-TTS | NAR flow | ~336M | (= F5-TTS) | (= F5-TTS) | Emilia-YODAS 114K h | *alpha; "inferior to official"* | — | **Apache-2.0** |
| Voicebox | NAR flow | 330M + 28M dur | ~20× VALL-E | — | 60K h EN | SIM-o 0.662 | WER 1.9 | **never released** |
| E2-TTS | NAR flow | 333M | 0.68 @32 (3090) | — | 50K h / 95K h | SIM-o 0.675–0.708 | WER 1.9–2.0 | no official weights |
| Matcha-TTS | NAR flow | **18.2M** | **0.015** @2 NFE (3090) | — | LJSpeech only | single-speaker | WER 2.34 @2 NFE | **MIT** |
| SupertonicTTS | NAR flow | **44M** | **0.02** (4090) | — | **945 h** EN | SIM 0.472 | WER 2.41 LS-PC | paper CC-BY-NC-SA |
| MegaTTS 3 | NAR latent diff | **339M** (*not* 0.45B) | 0.188 → **0.124** @8 (V100) | — | 60K h LibriLight | SIM-o 0.71 | WER 1.82 LS-clean | Apache-2.0 — **encoder withheld → cloning gated** |
| CosyVoice 2 | Hybrid AR→NAR | 0.5B | — | **formula only in paper** § | 166.8K h | SS 0.748 zh / 0.652 en | CER 1.45 / WER 2.57 | Apache-2.0 |
| Fun-CosyVoice3 | Hybrid | 0.5B | — | "as low as **150 ms**" | 1M h | SIM 78.0 / 71.8 | CER 1.16 / WER 2.02 | Apache-2.0 |
| Spark-TTS | AR codec LM | 0.5B | 0.136 (L20, TRT) | **210 ms P50** (named workload) | 102.5K h | SIM 0.672 / 0.584 | CER 1.20 / WER 1.98 | **weights CC-BY-NC-SA-4.0** |
| Llasa | AR codec LM | 1B / 3B / 8B | — | — | 250K h | SIM-o 0.669 → 0.684 | WER 3.22 → 2.97 en | CC-BY-NC-4.0 |
| Orpheus | AR codec LM | 3B | — | ~200 ms (no GPU) | 100K+ h | — | — | Apache-2.0, gated · **Hindi ckpt** |
| FireRedTTS-2 | AR dual-tf | unstated | — | 140 ms (L20) | 1.4M h | SIM 0.736 / 0.665 | CER 1.14 / WER 1.95 | Apache-2.0 |
| IndexTTS-2 | Hybrid | unstated (2.5 = 0.8B) | 0.326 fp16 (4090) | — | 55K h | SS 0.865 (FunASR) | CER 1.008 / WER 1.521 | bilibili — <100M MAU |
| Higgs Audio v2 | AR + adapter | 5.77B | — | — | **10M+ h** | SIM 67.70 | Seed-TTS WER 2.44 | Boson — <100k AAU |
| **dots.tts** | Continuous AR + FM head | 2B | — | **54 ms** dual-streaming | ~1.5M h | **SIM 81.0 / 77.1 / 79.5** | **CER 0.94 / WER 1.30 / hard 6.60** | **Apache-2.0** · Hindi high-WER |
| **VoxCPM2** | Hybrid AR + LocDiT | 2B | **0.13** (4090, Nano-vLLM) | streaming, no number | 2M+ h | SIM 79.5 / 75.3 · **hi 85.6** | CER 0.97 / WER 1.84 · **hi WER 19.7** | **Apache-2.0** · Hindi only |
| GEPARD 1.0 | AR codec LM | **556M** | **0.067** (5090) | **~50 ms TTFA** | unstated | SIM 0.585 | WER 0.036 (own protocol) | Apache-2.0 · no Indic |
| **Veena** | AR + SNAC | 3B (HF says 4B) | 0.05 self-reported | **<80 ms** H100 (self-rep.) | **unpublished** | "92%" — no methodology | none published | **Apache-2.0** · **hi+en, ta/te/bn/mr roadmap** |
| Indic-Speak | AR + SNAC | 3B (Llama-3.2) | — | ~200 ms | undisclosed | consistency 96.5 | **WER 26.5 / CER 17.9** | Indic Open Model v1.0 · **no cloning** |
| Hindi student (distilled IndicF5) | NAR flow, pruned | **190M** | 3.13 (Apple M4) | — | 17.6 h *synthetic* | SIM 0.750 (teacher 0.784) | CER 0.077 (teacher 0.030) | unstated |
| Sarvam Bulbul *(target)* | closed | — | — | — | — | — | LLM-WER 0.007 hi / 0.029 te | commercial |

† IndicF5 is F5-TTS Base at the same size, so F5-TTS's RTF should transfer;
**IndicF5's own RTF on a datacentre GPU is UNVERIFIED** — the only published
figure is 5.56 on an Apple M4.
‡ IN-F5's numbers are on its own IN11 test set with IndicConformer and a WavLM
cosine, on a scale where human recordings score 97.0 and 18.4. **Not comparable to
SIM-o or Seed-TTS-eval.**
§ **Correction worth carrying:** the widely-quoted "CosyVoice 2 = 150 ms" is not in
the CosyVoice 2 paper, which gives only a latency *formula*. The 150 ms claim is
attached to Fun-CosyVoice3's bi-streaming checkpoint.

Two further corrections to figures in common circulation: **MegaTTS 3 is 339M, not
0.45B** (the paper's Appendix A.1; tables label it "0.3B"). And **F5-TTS has two
sets of "official" numbers** — the paper's LS-PC WER 2.42 @32 NFE, and third-party
re-measurements of the *released v1 checkpoint* at 1.89. There is no published
v1-vs-base ablation (**UNVERIFIED**); 1.89 / SIM-o 0.655 is the realistic number
for what you would actually download.

---

## 4. Findings on data efficiency

### 4.1 Intelligibility is cheap; speaker similarity is not

This is §1.1 restated with its full evidence, because it is the finding that
should drive planning.

**From scratch, under ~1000 h, four independent confirmations:**

| work | corpus | hours | params | SIM-o | WER |
|---|---|---|---|---|---|
| ZipVoice | LibriTTS | **555 h** | 123M | **0.610** | 1.69 |
| F5-TTS retrained (same paper, same data) | LibriTTS | 555 h | 155M | 0.584 | 1.78 |
| F5-TTS Table 9 | LibriTTS | **585 h** | 155M | **0.60** | **2.20** |
| LFMD teacher | LibriTTS | 585 h | 160M | 0.582 | 2.23 |
| DiFlow-TTS | LibriTTS | 470 h | 164M | 0.45 | — |
| SupertonicTTS | mixed EN | **945 h** | **44M** | 0.472 | 2.41 |
| *F5-TTS Base, for contrast* | *Emilia* | *95K h* | *336M* | *0.655* | *1.89* |

WER lands at ground-truth level. **SIM-o lands in the 0.58–0.61 band and stops
there**, roughly 0.06–0.09 below a 95K-hour model. That gap is the price of small
data and it is remarkably consistent.

**Fine-tuned from a large base, speaker similarity is inherited and flat:**
IndicF5's SSIM sits at 92.7 / 93.1 / 93.1 for 1 h / 10 h / 100 h per language —
no trend at all — while MUSHRA goes 33.7 / 61.5 / 64.3 and WER goes 59.4 / 31.3 /
32.6. The language is learned from our data; the voice modelling is not.

**Planning consequence.** Set the cloning acceptance criterion against the base we
choose, not against our corpus size. And treat "train an Indic TTS from scratch on
our 900 h" as definitively ruled out — IN-F5 measured it at MUSHRA 43.2 against
73.4 for the fine-tune, and F5-TTS's own 585 h run needs ~200K updates before
alignment emerges at all.

### 4.2 Adapting to a new language is genuinely cheap

**F5-TTS on Korean, 12.65 h of a single speaker, one TITAN RTX:**

| | #trainable | CER | WER | SIM-O | NMOS |
|---|---|---|---|---|---|
| Ground truth | — | 6.519 | 12.679 | 0.625 | 4.74 |
| CosyVoice 2 (0.5B AR hybrid, *Korean in its pretraining*) | 0.5B | 6.812 | 17.956 | 0.583 | 3.49 |
| F5-TTS **from scratch** | 336M (100%) | **17.345** | **45.883** | 0.641 | 2.64 |
| F5-TTS pretrained, full fine-tune | 336M (100%) | 5.671 | 12.032 | 0.659 | 3.58 |
| **PEFT-TTS** (LoRA r=16 + prompt adapter) | **5.81M (1.72%)** | **4.906** | 12.625 | 0.640 | **3.88** |

Four things at once: the prior is worth 3× the CER; LoRA on 5.81M parameters
*beats* a full fine-tune on CER and NMOS; it ran on one consumer GPU; and a 336M
flow-matching model beat a 0.5B AR hybrid that already had Korean in its training
data, on CER, WER, SIM-O **and** NMOS.

**IndicF5's own curve** — 150K steps, same architecture and hyperparameters:

| hours/language | MUSHRA | SSIM-seen | SSIM-unseen | WER-M | WER-P |
|---|---|---|---|---|---|
| 100 h | 64.3 | 93.1 | 93.9 | 32.6% | 30.5% |
| **10 h** | **61.5** | **93.1** | **94.0** | **31.3%** | 31.8% |
| 1 h | 33.7 | 92.7 | 93.5 | 59.4% | 60.4% |

10 h/language "retains key emergent behaviors … with an average reduction of 0.8%
in performance across metrics." Below that, a cliff. Our budget is 15–20× the knee.

**Also at small per-language scale:** LoRA on VoxCPM2 lifted Khmer MOS 3.85 → 4.23
from **25.5 h**, training 0.19–3.03% of a 2.4B model — and gave **no gain for
Korean**, which was already in pretraining. A useful calibration for fallback B
(§9): adaptation only pays where the base is actually weak.

### 4.3 But AR is not inherently data-hungry — the honest counter-evidence

NVIDIA's Align2Speak adds a new language to **Koel-TTS**, an autoregressive decoder
over NanoCodec tokens at 21.5 FPS, pretrained on ~21K h. New languages — Polish,
Portuguese and **Hindi** — from **30 min / 1 h / 5 h** subsets: "with only 30
minutes of Portuguese data, CER drops from 33.00% in the baseline to 3.94% after
fine-tuning and GRPO," with the same trend reported for Hindi. (Per-language
figures are in a plot, not a table, so the **Hindi numbers are UNVERIFIED**.)

So the claim "NAR flow matching is more data-efficient than AR speech-LMs" should
be stated as: **well-supported for intelligibility and alignment, weakly supported
for speaker similarity, and not supported as a property of the architecture
independent of the base.** The one matched-data comparison that exists — DiFlow-TTS
evaluating both on ~500 h of LibriTTS under one protocol — gives F5-TTS SIM-O 0.52
against VALL-E's 0.40. One paper, one protocol.

Note also what Align2Speak's 30-minute result required: a 21K h multilingual base
*and* ASR-guided GRPO. We have neither. It argues that *an AR model with the right
base* is data-efficient. It is not an argument for *this* AR model.

### 4.4 Architecture matters most exactly where we live

F5-TTS and E2-TTS on the same 945 h of Mandarin at the same 155M size: **WER 4.17
vs 9.63.** The ConvNeXt text encoder plus DiT is 2.3× more robust than E2's flat
UNet at small data. And ZipVoice's 123M beats a retrained 155M F5-TTS on identical
555 h data (SIM-o 0.610 vs 0.584, WER 1.69 vs 1.78) at 1/3 fewer parameters.

Low-data robustness is an architecture property within the NAR family, and it is
*not* bought by size. For a rebuild (§7), **ZipVoice is the best-evidenced
small-data recipe** found in this research.

---

## 5. Findings on inference

### 5.1 Our 17 s first-chunk latency is mostly a chunking policy

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
| IndicF5 / F5-TTS @16 NFE, no chunking needed | **~0.8 s** |
| F5-TTS + EPSS @7 NFE | **~0.2 s** |
| Spark-TTS (measured, named GPU + workload) | **0.21 s** |
| dots.tts dual-streaming | **0.054 s** |

NAR wins by ~5× over a fixed AR chunker and needs no chunking machinery at all.
But a reader who believes the architecture alone is worth 17 s → 1 s has been
misled, and **the cheapest latency fix available today needs no backbone change.**

Note where the floor is: every sub-250 ms number belongs to an AR or hybrid model.
Non-causal flow matching has no path below one utterance.

### 5.2 The distillation ladder — take the free rungs, stop before the last

Few-step inference is a second-order lever, available *after* the family choice,
and unusually well-measured:

| rung | method | NFE | RTF | quality cost |
|---|---|---|---|---|
| **1 — free, today** | **EPSS** (training-free) | 32 → **7** | **0.123 → 0.030** (3090), **4.1×** | WER 2.37→2.45; SIM unchanged. *Naive* 7-NFE without EPSS collapses to WER 4.16 / zh 7.96 |
| 2 — cheap | EPSS floor | → 6 | 0.027 | WER 2.41, fine |
| **floor** | EPSS | → 5 | 0.023 | **cliff**: SIM 0.59, UTMOS 3.84 → **2.91** |
| **3 — the good one** | **DMOSpeech 2** (DMD + CTC/SV + GRPO), F5-TTS teacher | 32 → **4** | **0.1671 → 0.0316** (H100), 5.3× | **student beats teacher on both**: WER 1.947 → **1.752**, SIM-o 0.662 → **0.698** |
| 3′ | ZipVoice-Distill | 16 → **4** | **0.0125** (H20) = **23.7× F5-TTS** | WER 1.64 → **1.51**, SIM-o 0.668 → 0.657 |
| 4 — **do not** | LFMD (2026 SOTA 1-step) | → **1** | 0.0163, 12.5× | WER 1.76 → 1.80 (free) but **SIM-o 0.667 → 0.607** |
| 4′ | MeanFlow on CosyVoice2 decoder | 10 → 1 | 0.0775 → **0.0046**, 16.8× | WER 3.18 → 3.41, SpkSim 0.940 → 0.932 |

**The pattern is consistent and directly relevant to us: one-step distillation costs
speaker similarity, not intelligibility.** Since §4.1 says speaker similarity is
the scarce resource in our whole project, **stop at 2–4 NFE.** Our existing
`student_flow` is a single-step decoder, and its measured mel L1 of 0.250 against
0.168 for the dense student **[ours]** is this same trade showing up in our own
repo.

Compounding rungs 1 and 3 against our baseline: RTF 0.78 → ~0.03, with *better*
intelligibility and speaker similarity than the undistilled teacher. That is the
realistic end state of the recommended path.

---

## 6. Findings on Indic quality

### 6.1 IndicF5 is the best open Indic base — and it is not Sarvam Bulbul

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

For scale: the only frontier model publishing a Hindi number is VoxCPM2, at **WER
19.699** against ~1.84 on English. A 10× Indic gap in a 2B model trained on 2M
hours. **Indic is genuinely hard for everyone, and nobody open is near Bulbul.**

**Read this as: IndicF5 is the best open starting point, not a finished product.**

### 6.2 Weaknesses we would inherit

- **English inside Indic text is effectively dropped.** Its character tokeniser
  accepts Latin, but its pretraining corpus has no Latin-script *audio*. Measured
  code-mix LLM-WER **0.80–0.85** without preprocessing. For a product that must do
  "Hindi + English," this is a first-class problem, not polish.
- **Tamil is its thinnest language**, ~80 h — cited as the reason for weak Tamil
  code-mixing. Tamil is also *our* worst language **[ours]**. Both systems are weak
  in the same place; switching base does not fix Tamil, it changes what we fine-tune
  from.
- **Reference audio is truncated to ~12 s.** F5-TTS clips long references (README
  "<12s"; community reports a 15 s hard cut). Our spec asks for **20–30 s**, and
  KittenTTS 2's card explicitly accepts "5–30 seconds". A real capability the
  incumbent has and the recommendation does not.
- **The reference transcript is required** at inference. We ship a
  `ReferenceTranscriber`; Cross-Lingual F5-TTS 2 and RTFree-F5 remove the
  requirement entirely. Solvable, but work. (Both read at abstract level only —
  **UNVERIFIED**.)
- **AI4Bharat now point elsewhere.** The IndicF5 card was updated ~10 days ago to
  recommend Indic-Speak — a 3B Llama-3.2 AR model over SNAC codes that **does not
  support voice cloning at all**, with WER 26.5 / CER 17.9. Disqualified for us on
  requirement 3. The redirect signals AI4Bharat's roadmap, not that IndicF5 is a
  bad base.

### 6.3 Adding English data *hurt* Indic quality

EN→EN+IN scored 66.2 against EN→IN's 73.4 — mixing English back in during Indic
fine-tuning cost 7 MUSHRA points. The paper recommends keeping the original English
checkpoint for English and the Indic fine-tune for Indic.

**This cuts against D1's reasoning** **[ours]**, which keeps English recordings in
every training phase. D1's concern was *tokeniser fertility* — Sarvam makes English
~14% more expensive — and IndicF5 has no BPE tokeniser at all, just 685 raw
characters. So D1 is not wrong; it is scoped to the AR arm. The English-retention
strategy must be re-decided per family, and here the published answer is "route to
two checkpoints."

---

## 7. The direct answer: keep KittenTTS 2, or switch?

**Switch the backbone. Keep the harness.**

### 7.1 What switching costs

The instinct to protect the KittenTTS 2 investment is right but mis-locates where
the investment is:

| Asset | Survives a switch? |
|---|---|
| Preflight orchestrator + overnight driver | **Yes** — architecture-independent |
| Eval harness (Set A/B/C, floor CER, WavLM SECS, D8–D11) | **Yes** — already has an `IndicF5Synthesizer` |
| HPC portability layer, Slurm templates, shard reader, preprocessing job | **Yes** |
| Danda / NFC / Indic-digit text fixes (D4) | **Yes** |
| Data acquisition plan (~900 h, IndicVoices-R, Rasa) | **Yes** — the same corpora IndicF5 used |
| Licensing analysis | **Yes**, and it gets sharper (§7.5) |
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

That last one flips from a constraint into an argument: PEFT-TTS adapted F5-TTS to
a new language with **5.81M trainable parameters on a single TITAN RTX**. Our 12 GB
card is not a limitation in the recommended path — it is the published setup.

### 7.2 The case for keeping KittenTTS 2, stated as strongly as it deserves

- **The codec gate passed, and passed well.** Every Indic language reconstructs
  *better* than English through S3 **[ours]**. Real, measured, and it removed the
  project's headline risk.
- **Arbitrary-length reference conditioning.** x-vector → `spk_proj` takes 5–30 s,
  matching our spec. F5-TTS truncates at ~12 s.
- **Streaming.** §5.1: every sub-250 ms number in this document belongs to an AR or
  hybrid model. Non-causal flow matching cannot get there.
- **AR/hybrid holds the quality frontier.** dots.tts (2B, Apache-2.0) at CER 0.94 /
  WER 1.30 / SIM ~79; VoxCPM2 at 0.97 / 1.84 / 79.5; against F5-TTS's 1.89 / 0.655.
- **Align2Speak shows AR adapting from 30 minutes** (§4.3).

### 7.3 Why they lose anyway

- The codec gate says *the decoder* is fine. It says nothing about whether a 1.7B
  ternary AR LM learns Tamil from 150 h. Our measurement on that is **Tamil
  perplexity 287.2, 4.7× English** **[ours]**, in the language where Qwen3 fertility
  is worst (10.13 tokens/word). IndicF5 has already *trained on* Tamil.
- The 12 s reference limit is real, but 12 s is empirically enough: IN-F5 holds
  SSIM > 92 on unseen speakers; the distilled Hindi student holds SIM 0.750.
- **We do not have streaming** — ours is explicitly "chunk-level, not token-level"
  **[ours]**. Every low-latency figure cited belongs to a chunk-aware causal flow
  decoder we would have to build either way.
- The frontier-quality models train on 1.5M–10M hours. §4.1 shows that regime does
  not transfer to <1000 h, where the prior dominates and SIM-o is the binding
  constraint.
- Align2Speak's 30-minute result needed a 21K h multilingual base plus ASR-guided
  GRPO. KittenTTS 2's training data is undisclosed, its Tamil perplexity is 287.2,
  and nobody has built the GRPO pipeline.
- **Nothing in the literature supports 1.7B for this task.** Llasa 1B→8B moves
  English WER only 3.22→2.97, while inference-time search on the 1B reaches 1.46.
  CosyVoice 2 matches models 3× its size at 0.5B. ZipVoice beats F5-TTS at 123M.

### 7.4 And the one that settles it

**KittenTTS 2's licence has an expiry date built in.** The Stellon Labs Community
License terminates above $1M revenue or $1M cumulative funding, requires "Powered
by Stellon Labs" to be displayed, and §4b forbids using the model or its outputs to
train other models **[ours]**. For a project whose brief says "commercial use
intended; licences are a hard filter," this is a backbone we must leave anyway. The
only question is whether we leave now or after spending the allocation.

### 7.5 Licensing — the sharpest finding in this document

The licence chain for the recommended base does not close cleanly.

```
Emilia (101.7K h)             CC-BY-NC-4.0  "non-commercial purposes only"
  └─> F5-TTS Base v1 weights  CC-BY-NC-4.0  (code is MIT; the weights are not)
        └─> IndicF5 / IN-F5   listed MIT on HuggingFace    <-- conflict
```

The IN-F5 paper states it fine-tunes "starting from the English checkpoint
pretrained on nearly 100K hours," and the F5-TTS repository states its pre-trained
models are CC-BY-NC "due to the training data Emilia." AI4Bharat's MIT label does
not visibly extinguish an upstream non-commercial term the upstream authors
attached deliberately. **This is a legal question and it is UNVERIFIED in both
directions** — the F5-TTS discussion thread records a community reading that the
Emilia team may not consider trained models to inherit NC, but that is a user's
opinion, not a published position.

**The escape hatch exists.** Every ingredient of the IN-F5 recipe has a permissive
substitute:

| Ingredient | NC version | Permissive substitute |
|---|---|---|
| Pretrained base | F5-TTS Base v1 (CC-BY-NC) | **`mrfakename/OpenF5-TTS` — Apache-2.0**, F5-TTS Base V1 config, 1M steps |
| Pretraining corpus | Emilia (CC-BY-NC) | **Emilia-YODAS, 113.9K h, CC-BY-4.0** |
| Indic corpus | — | IndicVoices-R (CC-BY-4.0), Rasa (CC-BY-4.0), IndicTTS, LIMMITS |
| Architecture | F5-TTS 336M | **ZipVoice 123M** — better SIM-o and WER on identical 555 h |
| Recipe | — | published in full: 150K steps, AdamW 5e-5, 30K frames/GPU, 48K warm-up, 32×H100 |

OpenF5-TTS is self-described **alpha**, English-only, "still inferior to the
official NC-licensed F5-TTS model," with no published benchmarks. Its quality is
**UNVERIFIED** and it is the main risk in the shipping plan — the reason the
research base and the shipping base are deliberately different models.

Full licence survey, since this is a hard filter:

| Model | Licence | Verdict for a commercial product |
|---|---|---|
| OpenF5-TTS, Matcha-TTS, StableTTS, CosyVoice 2/3, VoxCPM2, dots.tts, GEPARD, FireRedTTS-2, **Veena** | Apache-2.0 / MIT | **clean** |
| Orpheus | Apache-2.0, gated | clean, access-controlled |
| IndicF5 | MIT claimed, NC upstream | **research only** until resolved |
| KittenTTS 2 | Stellon Community | terminates at $1M; **outputs cannot train other models** |
| Higgs Audio v2 | Boson Community | commercial until >100k annual active users |
| IndexTTS-2 | bilibili Model Use | commercial until >100M MAU / RMB 1B; Chinese law |
| Indic-Speak | Indic Open Model v1.0 | commercially usable, but **no voice cloning** |
| **Spark-TTS** | **weights CC-BY-NC-SA-4.0** | code is Apache-2.0; **the weights were relicensed away from it** |
| Llasa (all sizes), Fish S1-mini | CC-BY-NC(-SA)-4.0 | non-commercial |
| MegaTTS 3 | Apache-2.0 but **encoder withheld** | cloning gated through ByteDance — **disqualified** |
| Voicebox | never released | unusable |
| ZipVoice | **UNVERIFIED** | must be checked before it enters the rebuild plan |
| Sarvam-1 tokeniser | non-commercial | already flagged **[ours]** |

Two traps worth naming. **Spark-TTS's weights were relicensed from Apache-2.0 to
CC-BY-NC-SA-4.0 after release** because of its training data — exactly the failure
mode IndicF5 may be in, and proof the risk is not theoretical. And **HuggingFace
parameter counts are inflated** by codec embedding tables (Llasa-3B reports 4B,
Veena-3B reports 4B); use paper figures for architecture and HF figures for VRAM.

---

## 8. The cheapest experiment that decides this

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
`cer_above_floor` against the same recognizer and a WavLM SECS drawn from
references that are *not* the target sentence — a number on **our** protocol, not a
paper's. Record wall-clock while you are there: it closes the UNVERIFIED
IndicF5-RTF gap for free.

**The decision rule, written before the data.** §4.1 says speaker similarity is the
scarce resource and is set by the base, so make SECS the primary criterion:

- **Switch, as recommended** — IndicF5's WavLM SECS lands near the 0.65+ band of a
  large-base model, and Tamil `cer_above_floor` is low while our Tamil perplexity
  sits at 287.2 **[ours]**.
- **Go to the fallback** — SECS lands in the 0.45–0.55 band of a *small-data*
  model, or Tamil CER is no better than the ASR floor. The Indic prior is then not
  the asset §4.1 claims, and the base-quality leg of the recommendation fails.
- English catastrophic (expected, §6.2) — does not change the verdict, but settles
  the routing decision: two checkpoints, not one.

Cost: hours of wall clock on one GPU. **This replaces the "highest-value remaining
hour" the findings doc currently nominates.**

The tie-breaker, only if the first run is ambiguous: fine-tune IndicF5 and
KittenTTS 2 on the *same* 10 h Tamil slice and score both. §4.2 says 10 h is enough
to see the difference, so even the tie-breaker is cheap — and it would be a cleaner
hours-matched AR-vs-NAR comparison than anything published (§4.3).

---

## 9. Fallback plan

**If this is wrong, it will be wrong in one of two specific ways.**

### Failure mode A — streaming is a hard requirement

NAR flow matching cannot stream. A 5 s sentence arrives in ~0.8 s (~0.2 s with
EPSS): fine for narration, marginal for conversation. §5.1 shows no amount of NAR
tuning reaches sub-250 ms.

**Fall back to the CosyVoice 2 hybrid, on our existing LM.** `s3gen.py` is vendored
from ResembleAI Chatterbox (Apache-2.0), "which in turn builds on Alibaba's
CosyVoice and Matcha-TTS", with `SPEECH_VOCAB_SIZE = 6561` at 25 Hz **[ours]** —
and CosyVoice 2's speech tokeniser is FSQ with a codebook of **6,561** at **25 Hz**.
Same token space. **KittenTTS 2 is structurally CosyVoice 2 with a ternary
Qwen3-1.7B in place of Qwen2.5-0.5B.** The migration is to replace the flow decoder
with a chunk-aware causal one and interleave text/speech tokens at N:M. That keeps
the ternary checkpoint, the codec gate, the vocabulary surgery and the QAT work —
the entire 40% a switch discards. Apache-2.0, WER 2.47 / SS 0.745 LibriSpeech
test-clean. A MeanFlow distillation of exactly this decoder is published: 10 NFE →
1, RTF 0.0775 → 0.0046.

If we are rebuilding for streaming anyway, **dots.tts** is the better target: 2B,
Apache-2.0, **54 ms dual-streaming**, CER 0.94 / WER 1.30 / SIM ~79 — the best open
numbers in this document — using continuous AR latents with a flow-matching head.
Hindi is flagged as higher-WER on its own card, so Indic work remains ours.

Cost of being wrong this way: the IndicF5 evaluation plus whatever fine-tuning had
started. Weeks, not months.

### Failure mode B — IndicF5's prior is weaker than its paper suggests

§6.1 already shows an 18× Hindi gap to Bulbul on one protocol. If §8 confirms
IndicF5's SECS sits in the small-data band, the "strong prior" premise fails.

**Fall back to a clean-licence base with real coverage**, ranked:

1. **VoxCPM2** — Apache-2.0, RTF 0.13 on one RTX 4090, streaming, SIM 75.3 / WER
   1.84 en, 2M training hours, and a published Hindi number (WER 19.699 / SIM 85.6).
   Crucially, **LoRA adaptation of VoxCPM2 is already demonstrated**: Khmer MOS
   3.85 → 4.23 from 25.5 h, training 0.19–3.03% of params — and *no gain* for
   Korean, which was already in pretraining. That is the right shape for us: Hindi
   is covered, the other four are not, and the adaptation cost is measured. 2B is
   larger than our efficiency goal wants.
2. **Veena** (Maya Research) — Apache-2.0, Hindi + English including code-mixed,
   with Tamil/Telugu/Bengali/Marathi on the published roadmap, and already a LoRA
   fine-tune (attention rank 192, FFN rank 96, 4-bit NF4 base), which matches our
   hardware. **But it publishes no training hours, no WER and no Seed-TTS-eval
   numbers**; its "92% speaker similarity / 98% intelligibility / <80 ms" are
   self-reported with no methodology. Everything about Veena's quality is
   **UNVERIFIED** and would need the §8 harness run before it could be trusted.
3. **Fun-CosyVoice3 0.5B** — Apache-2.0, best open quality at the size we want
   (CER 1.16 zh / WER 2.02 en), 150 ms bi-streaming. No Indic coverage at all.
4. **Stay on KittenTTS 2 and spend the allocation**, accepting Tamil ppl 287.2 as
   the start point and the Stellon licence as a future migration.

### The tripwire

Both failure modes are detected by the same experiment in §8, which is why it runs
first. The thing that would make this document wrong and *not* be caught is a
licence ruling — so treat §7.5 as a parallel track with a different owner.

---

## 10. Decisions proposed

Numbering continues from the findings doc, which ends at D12.

**D13 — Move the backbone to NAR flow matching.** Not because NAR is inherently
data-efficient (§4.3 says that is only partly supported), but because a 336M
flow-matching model adapts to a new language on ~12 h and one consumer GPU, runs
~5× faster than what we have before distillation and ~25× after, and IndicF5 is the
only open base with a real prior across all five of our languages.

**D14 — Plan around speaker similarity as the scarce resource.** §4.1: our
150–200 h/language buys intelligibility, not cloning. Set the cloning acceptance
criterion against the *base*, and never consider training from scratch.

**D15 — Research base is IndicF5; shipping base is a clean-licence rebuild.**
Deliberately different models. Use IndicF5 to learn and to measure; re-run the
published EN→IN recipe from OpenF5-TTS — ideally at ZipVoice's 123M — on CC-BY-4.0
Indic data for anything that ships. Do not let the two converge by accident.

**D16 — Run the IndicF5 evaluation before any further training.** Already
implemented; needs only an `HF_TOKEN`. Any week spent fine-tuning either backbone
before this number exists is a week spent on a guess.

**D17 — Take EPSS immediately, distil to 4 NFE later, never to 1.** Rung 1 is
training-free and worth 4× at no measurable quality cost. Rung 3 (DMOSpeech-2-style)
*improves* both WER and SIM-o. One-step distillation costs 0.06 SIM-o, which by D14
is the one thing we cannot spend.

**D18 — Re-decide English handling per family.** D1 keeps English in every phase.
IN-F5 measured that strategy costing 7 MUSHRA points and recommends routing English
to a separate checkpoint. D1's reasoning was about tokeniser fertility, which does
not apply to a 685-character vocabulary — so D1 is not wrong, it is scoped to the
AR arm.

**D19 — Fix the chunker before blaming the architecture for latency.** §5.1 shows
17 s is ~21 s of audio at RTF 0.78. A smaller first chunk buys ~4× with no backbone
change, and it makes the AR arm an honest comparison rather than a strawman. Do
this even if D13 proceeds.

**D20 — Treat the Emilia→F5→IndicF5 licence chain as a blocking, parallel track.**
Unresolved in both directions, and the single thing most likely to invalidate the
shipping plan. Spark-TTS's weights being relicensed from Apache-2.0 to CC-BY-NC-SA
*after release* is the precedent.

**D21 — Do not scale the LM.** Llasa 1B→8B moves English WER only 3.22→2.97, while
inference-time search on the 1B reaches 1.46. CosyVoice 2 matches models 3× its size
at 0.5B. ZipVoice beats F5-TTS at 123M. Our 1.7B is unsupported by any published
scaling evidence for this task, and our stated target is efficiency.

---

## 11. Honest gaps

- **The hours-matched AR-vs-NAR comparison is thin.** Exactly one exists
  (DiFlow-TTS, ~500 h LibriTTS, F5-TTS SIM-O 0.52 vs VALL-E 0.40), and Align2Speak
  is active counter-evidence. The §8 tie-breaker would be a cleaner comparison than
  anything published — which is itself a reason to run it.
- **IndicF5's RTF on a datacentre GPU is UNVERIFIED.** Only 5.56 on an Apple M4 is
  published. We infer F5-TTS's 0.15 @16 NFE from identical architecture and size.
- **OpenF5-TTS's quality is UNVERIFIED.** Self-described alpha, no benchmarks, its
  own card says it is inferior to the NC checkpoint. The clean-licence shipping plan
  rests on it being good enough, or on us training a replacement.
- **ZipVoice's licence is UNVERIFIED** and it has no Indic coverage. It enters this
  document as an architecture recommendation for a rebuild, not as a base.
- **Veena's everything is UNVERIFIED** — no hours, no WER, no methodology. It is
  ranked in fallback B on licence plus Indic roadmap alone, which is a weak basis.
- **Align2Speak's Hindi numbers are in a plot, not a table.** Only the Portuguese
  33.00% → 3.94% is quotable.
- **Per-language hours for IN-F5 are not published** — only 1,417 h across 11
  languages, plus a third-party ~80 h figure for Tamil.
- **The licence question (§7.5) is unresolved** and is the largest single risk.
- **No MOS or listening test in this document is ours.** Every quality number is
  self-reported by a system's authors or by a competitor. §8 produces the first
  number on our protocol.
- **Seed-TTS `test-hard` is not published for F5-TTS, E2-TTS or MegaTTS 3**, and is
  omitted by Spark-TTS, FireRedTTS-2, IndexTTS-2 and Higgs. It is the discriminating
  split and its absence is rarely accidental.
- **Training-hours disclosure is getting worse.** FreyaTTS, dots.tts,
  RobustSpeechFlow, Veena and Indic-Speak all decline to state corpus hours. Only
  2024–25 academic work gives clean data-scale ablations — which is why §4.1 leans
  on Voicebox (2023) and F5-TTS (2024).
- **The web-search budget for this session was exhausted** partway through, shared
  across all three researchers. Sarvam's and Krutrim's own 2026 Indic releases were
  not surveyed, so the commercial landscape in §6.1 may be incomplete.

---

## 12. Sources

**IndicF5 / IN-F5**
- https://arxiv.org/abs/2505.20693 · https://arxiv.org/html/2505.20693v1
- https://huggingface.co/ai4bharat/IndicF5

**F5-TTS family and adaptation**
- Paper: https://arxiv.org/html/2410.06885v3 · https://aclanthology.org/2025.acl-long.313/
- Repo, runtime benchmarks, licences: https://github.com/SWivid/F5-TTS
- Weights (CC-BY-NC): https://huggingface.co/SWivid/F5-TTS
- Licence discussion: https://github.com/SWivid/F5-TTS/discussions/129
- **PEFT-TTS (Korean, 12.65 h, 1.72% params):** https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf
- OpenF5-TTS (Apache-2.0): https://huggingface.co/mrfakename/OpenF5-TTS
- Reference-length behaviour: https://github.com/niknah/ComfyUI-F5-TTS
- Cross-Lingual F5-TTS 2: https://arxiv.org/abs/2609.15184 · RTFree-F5: https://arxiv.org/abs/2606.20266

**Data-scaling evidence (§4.1 — the core of this document)**
- **Voicebox 60 h → 60,000 h ablation:** https://arxiv.org/html/2306.15687v2
- **ZipVoice (123M, 555 h LibriTTS):** https://arxiv.org/html/2506.13053v3
- E2-TTS: https://arxiv.org/html/2406.18009v2
- LFMD: https://arxiv.org/html/2609.36324v1 · DiFlow-TTS: https://arxiv.org/html/2509.09631v3
- SupertonicTTS (44M, 945 h): https://arxiv.org/html/2503.23108v3
- **Align2Speak (AR, 30 min/language, incl. Hindi):** https://arxiv.org/html/2509.21718v1
- P-Flow: https://research.nvidia.com/labs/adlr/projects/pflow
- LoRA on VoxCPM2 (Khmer, 25.5 h): https://arxiv.org/abs/2606.26618

**Few-step / distillation (§5.2)**
- **EPSS / Fast F5-TTS:** https://arxiv.org/html/2505.19931v2
- **DMOSpeech 2 (F5-TTS teacher, 4 NFE, beats teacher):** https://arxiv.org/html/2507.14988v1
- DMOSpeech: https://arxiv.org/html/2410.11097v2
- MeanFlow on CosyVoice2 decoder: https://arxiv.org/html/2606.18072v1
- FlashSpeech: https://arxiv.org/html/2404.14700v4 · RapFlow-TTS: https://arxiv.org/html/2506.16741v1

**Data and licences**
- Emilia / Emilia-YODAS: https://huggingface.co/datasets/amphion/Emilia-Dataset
- IndicVoices-R: https://arxiv.org/html/2409.05356v2 · https://huggingface.co/datasets/ai4bharat/indicvoices_r
- Rasa: https://arxiv.org/html/2407.14056v2 · https://huggingface.co/datasets/ai4bharat/Rasa

**Hybrid / AR**
- CosyVoice 2: https://arxiv.org/html/2412.10117v3 · https://huggingface.co/FunAudioLLM/CosyVoice2-0.5B
- CosyVoice 3: https://arxiv.org/html/2505.17589v2 · https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- Llasa (scaling): https://arxiv.org/html/2502.04128v2 · https://huggingface.co/HKUSTAudio/Llasa-3B
- Spark-TTS: https://arxiv.org/html/2503.01710v1 · https://huggingface.co/SparkAudio/Spark-TTS-0.5B
- Orpheus: https://github.com/canopyai/Orpheus-TTS
- FireRedTTS-2: https://arxiv.org/html/2509.02020v2 · IndexTTS-2: https://arxiv.org/html/2506.21619v2
- Higgs Audio v2: https://github.com/boson-ai/higgs-audio
- **dots.tts:** https://arxiv.org/abs/2606.07080 · https://huggingface.co/dots-studio/dots.tts-soar
- **VoxCPM2:** https://github.com/OpenBMB/VoxCPM · https://huggingface.co/openbmb/VoxCPM2
- GEPARD 1.0: https://arxiv.org/abs/2609.04222 · MegaTTS 3: https://arxiv.org/html/2502.18924v4
- Matcha-TTS: https://arxiv.org/html/2309.03199v2 · GLM-TTS survey table: https://arxiv.org/html/2512.14291

**Indic-specific**
- **Veena:** https://huggingface.co/maya-research/Veena · https://huggingface.co/maya-research/maya1
- Indic-Speak: https://huggingface.co/bodhan-ai/indic-speak
- Praxy Voice (Bulbul / ElevenLabs / IndicF5): https://arxiv.org/html/2604.25441v1
- PSP accent benchmark: https://arxiv.org/html/2604.25476
- Hindi depth-pruning distillation of IndicF5: https://arxiv.org/abs/2607.18662
- Bengali IndicF5 fine-tune: https://huggingface.co/ehzawad/indicf5-bangla-tts

**Incumbent**
- https://huggingface.co/KittenML/kitten-tts-2
- https://betterstack.com/community/guides/ai/kitten-tts-2/
