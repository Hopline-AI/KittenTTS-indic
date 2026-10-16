# Data efficiency: how to get high quality from very little Indic speech

**Research pass of 2026-10-16.** Companion to
`docs/indic/2026-10-16-findings-and-decisions.md`. Nothing here was measured on
this machine — every number is from a published source and carries its URL. Claims
I could not verify in a primary source are marked **UNVERIFIED**; claims that are
mine rather than a paper's are marked **inference**.

Decisions continue the numbering in the findings doc, which ends at D12.

---

## 1. The answer, in one paragraph

**The architecture is not what makes a TTS system data-efficient. Pretraining is.**
The same flow-matching model, same size, same target language, same 12.65 hours:
trained from scratch it gets WER 45.9 and an unseen-speaker SMOS of 1.12; started
from an English/Chinese checkpoint it gets WER 12.6 and SMOS 2.81
([Kwon et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf)).
We already hold the expensive half of that: a 1.7B speech-LM pretrained on a large
English corpus, with a codec that we **measured** reconstructs Indic *better* than
English. The remaining question is not "how do we get 10,000 hours" but "which
900 hours, filtered how, mixed how". The published curve for exactly this
situation — English-pretrained TTS adapted to 11 Indian languages — says **10
hours per language retains most of what 100 hours per language buys, and 1 hour
per language falls off a cliff**
([Phir Hera Fairy / IN-F5, Table 4](https://arxiv.org/html/2505.20693v1)).

What *does* need hours is not intelligibility. It is **speaker generalisation and
hard-case robustness**, and both fail quietly — in a way the loss curve does not
show.

---

## 2. The evidence base

Seven studies carry most of the weight. Everything in sections 3–9 is drawn from
these plus supporting sources.

| # | Study | Why it matters here |
|---|---|---|
| **E1** | [Phir Hera Fairy / IN-F5](https://arxiv.org/html/2505.20693v1) (arXiv 2505.20693) | The only clean hours-per-language curve for Indic TTS: 1 / 10 / 100 h per language, 11 languages |
| **E2** | [PEFT-TTS, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/kwon25_interspeech.pdf) | Scratch vs full FT vs 1.72% PEFT at 12.65 h, *with an unseen-speaker test set* |
| **E3** | [chatterbox-indic-lora](https://huggingface.co/reenigne314/chatterbox-indic-lora) | Nearest architectural analogue we have: T3 Llama + S3 25 Hz + s3gen, LoRA, ~10 h/lang, Indic |
| **E4** | [Rasa](https://arxiv.org/html/2407.14056v1) (arXiv 2407.14056) | Text-selection ablation at 1 h: CER 0.812 → 0.074 from script choice alone |
| **E5** | [IndicOOV](https://arxiv.org/pdf/2407.13435) (arXiv 2407.13435) | 48–64 minutes of targeted volunteer audio cuts OOV error 22–41% relative |
| **E6** | [CosyVoice 3](https://arxiv.org/html/2505.17589v2) (arXiv 2505.17589) | 3,000 h vs 170,000 h on an identical 0.5B model — says *what* data buys |
| **E7** | [IndicVoices-R](https://arxiv.org/html/2409.05356v2) (arXiv 2409.05356) | Studio data vs many-speaker enhanced data, scored on unseen Indic speakers |
| **E8** | [Polyglot-Lion](https://arxiv.org/pdf/2603.16184) (arXiv 2603.16184) | **Qwen3, 1.7B, 968.83 h, 4 languages including Tamil, one GPU, 48 h.** Our exact scale and backbone family |
| **E9** | [Yong et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/yong25_interspeech.pdf) | Speaker count vs hours, with total duration held constant |

Supporting: [Cross-Lingual TTS benchmark for Indian languages](https://arxiv.org/html/2610.14826)
(arXiv 2610.14826, Oct 2026) for the ASR floors;
[UR-BERT](https://arxiv.org/html/2606.11681v1) (arXiv 2606.11681) for romanization;
[When Fine-Tuning Fails and when it Generalises](https://arxiv.org/html/2603.10904)
(arXiv 2603.10904) for breadth-vs-depth on an LLM-based TTS;
[Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts) for the
corpus-by-language accounting.

---

## 3. Architecture-level data efficiency — the question was mis-posed

### 3.1 The brief's hypothesis does not survive contact with the literature

> **Hypothesis under test:** "flow-matching models need far less data than AR
> speech-LMs."

**I could not find a single controlled comparison of flow-matching vs AR speech-LM
data requirements.** No paper trains both families on the same hours of the same
language and reports the curve. The claim circulates, but every citation chain I
followed terminates either in an inference-speed result (flow matching is faster
to sample, which is true and irrelevant here) or in a 2022 observation that AR
architectures have "higher data requirement compared to non-AR TTS"
([Do et al., arXiv 2306.00535](https://arxiv.org/pdf/2306.00535), citing Pine et
al., ACL 2022) — made about Tacotron 2 against FastSpeech 2, two architecture
generations before discrete-codec speech-LMs existed.

Mark it **UNVERIFIED**, and note it is also *not actionable*: we are not choosing
an architecture. We have a 1.7B ternary Qwen3 speech-LM with a proven codec and a
proven training loop. The useful question is what the evidence says about
adapting *this*.

### 3.2 What the evidence actually separates

Pretraining, not architecture. **E2** is the cleanest experiment in the whole
literature for our purposes because it holds everything constant except the
starting point. F5-TTS, 335.8M parameters, Korean, **12.65 hours, one speaker**
(KSS), one TITAN RTX:

| | #trainable | CER ↓ | WER ↓ | SIM-O ↑ | UTMOS ↑ | NMOS ↑ |
|---|---|---|---|---|---|---|
| Ground truth | — | 6.519 | 12.679 | 0.625 | 3.808 | 4.74 |
| CosyVoice 2 (0.5B, huge multilingual incl. Korean) | — | 6.812 | 17.956 | 0.583 | 3.447 | 3.49 |
| **F5-TTS from scratch** | 336M (100%) | **17.345** | **45.883** | 0.641 | 3.152 | **2.64** |
| F5-TTS pretrained, full fine-tune | 336M (100%) | 5.671 | 12.032 | 0.659 | 3.503 | 3.58 |
| **PEFT-TTS (pretrained + adapters)** | **5.81M (1.72%)** | **4.906** | 12.625 | 0.640 | 3.338 | **3.88** |

Three things to take from this table and carry into every later section.

1. **Pretraining is worth ~3.5× on CER and 3.6× on WER** at 12.65 hours. From
   scratch at this data scale is not a worse model; it is a broken one.
2. **A 1.72%-parameter adapter beat the 100% fine-tune** on CER (4.906 vs 5.671)
   and on human-rated naturalness (3.88 vs 3.58). At this data scale, capacity is
   not the binding constraint — *regularisation* is.
3. **The model beat its own ASR floor.** CER 4.906 against a ground-truth floor of
   6.519. That is not magic; a clean single-speaker TTS is easier for Whisper than
   a real recording. It is a direct demonstration of why D9 exists: a CER without
   its floor means nothing.

### 3.3 What hours actually buy — the decomposition that matters

**E6** scaled one architecture over 57× of data and reported three test sets. This
is the single most useful table in this document, because it says *which* quality
axes are data-bound.

CosyVoice 3.0-0.5B, SEED test sets (CER/WER ↓, speaker similarity SS ↑):

| | test-zh CER | SS | test-en WER | SS | **test-hard CER** | SS |
|---|---|---|---|---|---|---|
| 3,000 h | 1.68 | 0.710 | 6.60 | 0.614 | **27.60** | 0.679 |
| 170,000 h | 1.27 | 0.815 | 2.46 | 0.747 | **6.96** | 0.787 |
| change | −24% | **+0.105** | −63% | **+0.133** | **−75%** | +0.108 |

Read it as three separate statements:

- **Easy-sentence intelligibility saturates early.** 57× the data moved in-domain
  Chinese CER from 1.68 to 1.27. If your benchmark is short clean read sentences,
  3,000 hours already looks finished.
- **Speaker similarity does not saturate.** +0.11 absolute SS across the board.
  Zero-shot cloning is the thing that keeps paying for data.
- **Hard inputs are where data lives.** test-hard CER 27.60 → 6.96. Long,
  tongue-twisting, repetitive, numeric input is 4× better with 57× the data.

**Inference, and the organising idea of this document:** at our data scale we
should expect to land near-parity on Set A read sentences and to be visibly behind
on (a) unseen-speaker cloning and (b) Set C robustness and long-form. Those are
the two places to spend every extra hour, and the two places our evaluation must
be able to see. Section 13 turns this into a measurement plan; section 14 into a
risk register.

### 3.4 The Indic curve itself

**E1** fine-tuned English F5-TTS (~0.4B) on 11 Indian languages, 150K steps,
32×H100, lr 5e-5, and ablated hours per language (L):

| | MUSHRA ↑ | SSIM seen | SSIM unseen | WER monoglot ↓ | WER polyglot ↓ |
|---|---|---|---|---|---|
| 100 h/L | 64.3 | 93.1 | 93.9 | 32.6 | 30.5 |
| **10 h/L** | **61.5** | **93.1** | **94.0** | **31.3** | 31.8 |
| 1 h/L | **33.7** | 92.7 | 93.5 | **59.4** | 60.4 |

And the training-strategy comparison, same model, full 1,417 h IN11 set:

| Setup | Seen Nat. | Seen Sim. | Unseen Nat. | Unseen Sim. | Polyglot Nat. | **Overall MUSHRA** |
|---|---|---|---|---|---|---|
| Human | 75.9 | 84.6 | 74.6 | 83.2 | 70.6 | **77.4** |
| **EN → IN** (Indic only) | 78.0 | 86.8 | 76.6 | 85.8 | 68.2 | **73.4** |
| EN → EN+IN (mixed, equal) | 69.7 | 84.3 | 69.7 | 84.7 | 61.0 | **66.2** |
| ∅ → IN (from scratch) | 40.0 | 76.2 | 42.0 | 78.1 | 41.8 | **43.2** |

Four readings, and one of them is a trap.

1. **10 h/L is a real operating point.** 61.5 vs 64.3 MUSHRA, and monoglot WER is
   *lower* at 10 h than at 100 h (31.3 vs 32.6 — inside noise, but certainly not
   worse). The paper's own summary is "an average reduction of 0.8%".
2. **1 h/L is a cliff, not a slope.** MUSHRA 33.7, WER 59.4. Between 1 and 10 the
   model stops being a model. Whatever our per-language floor is, it is above 1 h.
3. **Speaker similarity barely moves across the whole sweep** (92.7 → 93.1 seen,
   93.5 → 94.0 unseen). This is *not* evidence that cloning is data-cheap. It is
   evidence that **SSIM at this operating point is saturated and therefore blind**
   — all three rows, including the one with WER 59.4, score ~93. Contrast E2 and
   E7, where unseen-speaker similarity moved 0.25 → 0.64 and 78.93 → 88.18. A
   metric that does not separate a working model from a broken one is not
   measuring the thing. **Do not use a saturating SSIM as our cloning gate.**
4. **From scratch on 1,417 hours (43.2) is worse than fine-tuning on 110 hours
   (61.5).** 13× less data, 18 MUSHRA points better, because of where it started.

> **The EN→IN vs EN→EN+IN result, and why I am not taking it at face value.**
> E1 found Indic-only fine-tuning beat an equal English/Indic mix by 7.2 MUSHRA,
> and the paper does not explain why. The obvious reading — "drop English replay"
> — is the wrong one for us, for two reasons. First, E1's mix was **equal
> weight**: ~1,417 h of English against 1,417 h of Indic, i.e. a 50% replay rate,
> which is not a replay ratio, it is co-training. Nobody tested 10% or 20%.
> Second, E1 did not need English afterwards; **we do** — English is a shipping
> language, and our tokenizer swap made English ~14% *more* expensive in tokens
> (findings §3.1), so English is exactly the capability most at risk. Their result
> argues against 50% English. It says nothing about 10%.

### 3.5 Where we actually stand against this

Our measured teacher-forced perplexities — English 60.9, Hindi 68.3, **Tamil
287.2** (findings §3.7) — say the base checkpoint already carries Hindi at near
parity and does not carry Tamil at all. That is a 4.7× perplexity gap and it maps
onto the Qwen3 fertility measurement (Tamil 10.13 tokens/word, worst of the five).

**Inference:** our hours are not fungible across languages. Hindi is close to the
10 h/L regime of E1. Tamil and Telugu are the languages that need the hours, the
oversampling, and the text-selection care, and they are also where a failure will
be loudest.

---

## 4. Cross-lingual transfer — script, romanization, and what transfers

### 4.1 Native script works, and is what the best Indic system uses

**E1** uses **685 native-script character tokens, no G2P, no romanization**, with
new embedding rows "initialized by randomly sampling existing English-checkpoint
token embeddings". That system reaches MUSHRA 80.5 against human 91.8, WER 19.2
against a human floor of 18.4, S-SIM 97.3 against human 97.0
([IN-F5 Table 6](https://arxiv.org/html/2505.20693v1)). Native script is not a
handicap.

**E3** does the same thing at the subword/grapheme level on an architecture that
is nearly ours — Chatterbox-Multilingual, a 0.5B Llama T3 over single-codebook S3
tokens at 25 Hz with an s3gen decoder — extending the vocabulary 2,454 → 2,871
rows (+417 graphemes) and **warm-starting each new Brahmic character from its
phonetically equivalent Devanagari character** (Telugu క from Hindi क), with the
original rows frozen by a gradient mask.

That is a better initialisation idea than ours. Our D2 `mean_rescaled` init
averages the Qwen3 subtokens a Sarvam piece decomposes into — a *textual*
average. E3's map is a *phonetic* one, and for a Brahmic abugida it is nearly
exact: the scripts are isomorphic at the aksara level by construction.

**D13 — add a Brahmic phonetic warm-start arm to the vocabulary surgery.**
For every Sarvam piece whose characters are Brahmic, initialise from the
corresponding Devanagari piece's row where the base checkpoint has one, falling
back to `mean_rescaled` otherwise. This costs an afternoon, is testable by the
same 12-check `verify_surgery.py` harness, and attacks precisely the Tamil/Telugu
gap that §3.5 says is our problem. Keep `mean_rescaled` as the control — this is
an arm, not a replacement.

### 4.2 Romanization helps when there is almost no data, and we are not in that regime

[UR-BERT](https://arxiv.org/html/2606.11681v1) pretrains a shared character-level
text encoder over `uroman`-romanized text from 495 languages with an auxiliary
speech-token-prediction objective (13K h, 8M sentences), and plugs it into VITS.
Against the stock VITS text encoder on genuinely tiny corpora:

| Language | hours | MOS (UR-BERT / VITS) | ΔCER (UR-BERT / VITS) |
|---|---|---|---|
| Sinhala | 1 h | 3.82 / 3.49 | 9.49 / 13.09 |
| Nepali | 2 h | 3.66 / 3.33 | 6.71 / 13.55 |
| Afrikaans | 2 h | 3.34 / 2.80 | 15.82 / 26.03 |
| Xhosa | 2 h | 3.48 / 3.05 | 10.74 / 17.70 |
| Javanese | 5 h | 3.05 / 2.80 | **28.03 / 23.50** (worse) |

Romanization roughly halves CER at 1–2 hours. At 5 hours (Javanese) the CER
advantage has already inverted. The paper has **no Indic TTS results** — Hindi,
Bengali, Tamil, Telugu and Marathi appear only in its pretraining list.

**Inference:** romanization is a *low-data* device. It buys a shared phonetic
space when you have too few examples to learn script-specific embeddings, and it
costs you orthographic information — Indic scripts are shallow and largely
phonemic, so ISO-15919 or `uroman` output throws away very little, but it also
adds a reversibility problem (romanizers normalise native digits to ASCII, which
would silently destroy our entire Set C digit-pair measurement; see
[ACL 2025 on reversible transliteration](https://aclanthology.org/2025.acl-long.795.pdf)).

**D14 — do not romanize. Use native script, with the Brahmic warm-start (D13) as
the cross-script transfer mechanism instead.** We are targeting 60–180 h/language,
not 1–2 h; the one system that reached near-human Indic quality (E1) used native
script; and romanization would break the Set C digit instrumentation that D11
depends on. Record this as reversible: if the Tamil arm fails at 10 h/L despite
D13, romanized input is the next thing to try, and the shard schema's
`text_ids` column makes it a re-preprocessing job, not a redesign.

### 4.3 Multilingual joint training beats per-language training

The [Cross-Lingual TTS benchmark](https://arxiv.org/html/2610.14826) (arXiv
2610.14826, Oct 2026) trained both bilingual (one Indian language + 30 h Indian
English) and multilingual (24 languages, 3,248.5 h) F5-based models and found
**multilingual models reach lower WER and CER than bilingual ones**, attributed to
exposure to diverse scripts and to duration-per-character variation. E3's
incremental curriculum reports the same direction anecdotally: Hindi CER
*improved* after further languages were added, with no catastrophic forgetting
observed (no before/after numbers given — **UNVERIFIED**).

**Inference:** train all five languages plus English in one run. Do not build five
adapters. This also matters for us specifically because related Indic languages
share the Brahmic aksara inventory that D13 exploits — Marathi and Hindi share a
script outright.

---

## 5. Parameter-efficient adaptation — and why it is a *quality* decision, not a memory one

Our findings doc chose LoRA because a full fine-tune does not fit in 11.5 GB
(findings §3.4). The literature says we would want LoRA even with unlimited
memory.

### 5.1 PEFT beats full fine-tuning at low data — on unseen speakers, by a lot

**E2**'s multi-speaker table is the one that should change how we think. Same
models as §3.2, now evaluated on **164 utterances from unseen speakers** of a
Korean multi-speaker corpus — the model having been fine-tuned on **one** speaker:

| | CER ↓ | WER ↓ | SIM-O ↑ | NMOS ↑ | **SMOS ↑** |
|---|---|---|---|---|---|
| Ground truth | 3.488 | 10.295 | 0.834 | 4.45 | **4.88** |
| CosyVoice 2 (large-scale) | 3.662 | 10.887 | 0.806 | 4.26 | 4.36 |
| F5-TTS from scratch | 36.482 | 107.541 | **0.250** | 1.13 | **1.12** |
| F5-TTS pretrained, **full fine-tune** | 13.454 | 39.413 | 0.627 | 2.24 | **2.35** |
| **PEFT-TTS (1.72%)** | **7.220** | **21.591** | 0.641 | 3.06 | **2.81** |

Set it against the single-speaker table in §3.2:

| | seen-speaker CER | unseen-speaker CER | **degradation** |
|---|---|---|---|
| Full fine-tune | 5.671 | 13.454 | **2.37×** |
| PEFT (1.72%) | 4.906 | 7.220 | **1.47×** |

**The full fine-tune did not merely fail to generalise — it actively destroyed
generalisation that the base checkpoint already had.** The same weights that
score 5.67 on the training speaker score 13.45 on new ones, and human-rated
speaker similarity collapses to 2.35 out of 4.88. The adapter, with 58× fewer
trainable parameters, keeps the damage to 1.47×.

This is the mechanism behind our D-series worry about "a model that ignores the
voice" (findings §4d): there are two independent ways to arrive at bad speaker
similarity, a wiring bug and an over-fit, and they look identical in a loss curve.

**LoRA rank is not monotone, and rank 32 wins on both test sets.** E2's ablation,
holding everything else fixed:

| LoRA rank | seen-speaker CER | unseen-speaker CER |
|---|---|---|
| 16 | 4.906 | 7.220 |
| **32** | **4.570** | **5.358** |
| 64 | 4.916 | 6.140 |

Rank 32 — what our memory probe already budgeted (findings §3.4, Phase B) and what
E3 independently chose — is the measured optimum in the one study that swept it,
on *both* the seen and the unseen set. Rank 64 over-fits.

> **A correction to how E2 is usually quoted, including by me on a first read.**
> E2's headline configuration is *not* its best-CER configuration. Its chosen row
> uses DropPath 0.3 on the prompt adapter; the ablation row that drops DropPath at
> the same rank scores **better** on both sets (seen CER 4.282 vs 4.906; unseen
> 6.662 vs 7.220). The paper selects DropPath for the naturalness/similarity
> trade-off it argues for in prose, not for CER. So: **take rank 32 from this
> paper, treat DropPath as unproven.**

**D15 — LoRA r=32 is the primary arm, and the reason is quality, not memory.**
Keep the embedding-only warm-up (Phase A) and LoRA joint (Phase B) plan exactly as
budgeted. The memory probe said r=32 fits; E2 says r=32 is also the best-scoring
rank. That the two agree is luck, but it removes a trade-off we thought we had.

### 5.2 The nearest analogue to our system, with numbers

**E3** is Chatterbox-Multilingual — T3 Llama backbone, single-codebook S3 at 25 Hz,
s3gen, speaker encoder frozen, **only the T3 text side adapted**. LoRA r=32 α=64 on
`q_proj,k_proj,v_proj,o_proj`, ~7.8M of ~544M trainable (1.4%), bf16, one GPU. CER
measured with Whisper large-v3 on 100 held-out samples per language:

| Language | data | CER | status |
|---|---|---|---|
| Hindi | ~10 h IndicTTS | **0.106** | stable |
| Kannada | ~7 h IndicTTS | 0.143 | trained |
| **Tamil** | ~10 h IndicTTS + Rasa | **0.161** | trained |
| Marathi | ~10 h IndicTTS + Rasa | 0.198 | trained |
| Gujarati | ~10 h IndicTTS + Rasa | 0.238 | trained |
| Bengali | ~15 h IndicTTS | 0.245 | trained |
| Telugu | ~52 h IndicTTS + Rasa | 0.285 | trained |
| Malayalam | ~10 h IndicTTS + Rasa | **0.859** | failed |

Read carefully, because the ordering is informative and partly negative:

- **~10 h per language in our architecture produces a working Hindi (CER 0.106)
  and a working Tamil (0.161).** This is the strongest direct evidence we have
  that our plan is feasible at low data.
- **Telugu got 5× the data of Tamil and came out worse (0.285 vs 0.161).** Hours
  are not the whole story; script, tokenizer fertility and corpus composition are.
  The sibling repo [`indic-chatter-train`](https://github.com/rkumar1310/indic-chatter-train)
  documents why the Telugu runs went the way they did: 20 more epochs on the ~8.7 h
  IndicTTS Telugu set "barely moves the loss", and only swapping in Rasa's ~26k
  utterances "actually moved the val loss". **More epochs on a small corpus
  saturate; more distinct material does not.**
- **Malayalam at CER 0.86 is a complete failure at the same 10 h as working
  Hindi.** Nobody knows why. This is the honest base rate: per-language outcomes
  at 10 h are high-variance, and at least one of our five could land here.
- **Two speakers per language** (one male, one female, from IndicTTS). The author
  explicitly flags that speaker generalisation is unevaluated. Given §5.1, assume
  it is bad.

Its replay recipe, from the sibling repo, is the only concrete Indic replay ratio
I found: **target language 0.7 / Hindi replay 0.3**, with an incremental curriculum
(Hindi → +Telugu → Telugu-heavy → +Kannada → all 8) and weighted sampling.

> **A bug in that lineage worth carrying.** The repo warns that upstream
> `T3.loss()` does not shift logits against targets, so it trains the model to
> predict the *current* token. Their trainer shifts. Our label shift is one of the
> things the teacher-forced gate confirmed at +4.7 nats (findings §3.7), so we are
> clear — but it is the same class of silent error as our `spk_proj` keyword-
> argument trap, and worth a named test rather than a comment.

---

## 6. Data quality over quantity

### 6.1 The thresholds that a 1,704-hour Indic corpus was actually built with

**E7** built IndicVoices-R by restoring IndicVoices in-the-wild recordings. The
pipeline is the most directly transferable artefact in this entire document,
because it was tuned on *our languages*:

1. keep 44.1 kHz, extempore + read speech, drop clips > 30 s
2. **HTDemucs** source separation (speech vs background)
3. **VoiceFixer** dereverberation
4. **DeepFilterNet3** to remove the artifacts VoiceFixer introduces
5. **filter:** C50 ≥ 30 dB · SNR ≥ 25 dB · duration 0.2–30 s · mean pitch ≤ 350 Hz
   · pitch std ≤ 150 Hz · speaking rate ≤ 30 chars/s · **drop any utterance whose
   two transcript levels disagree by more than 5% CER**
6. peak-normalise to 0.1 dB headroom; **text kept verbatim, no normalisation**

Step 4 is the detail nobody would guess: the dereverberation model introduces
artifacts that need a *third* model to clean up. Step 5's transcript-agreement
rule is the ASR-consensus filter in its cheapest form.

**E6** independently converged on the same shape at 1M-hour scale: diarization +
VAD, MossFormer2 denoise, **three independent ASR systems (Faster-Whisper
Large-V3, NVIDIA Canary-1B, SeamlessM4T-V2) with transcripts kept only when mean
pairwise WER < 15%**, MFA-driven punctuation insertion (comma at pauses ≥ 300 ms,
strip punctuation at pauses ≤ 50 ms), peak normalisation to 0.6, and a
**speech-to-text length-ratio filter dropping the smallest 1% and largest 5%**.

That last one is cheap and catches a failure our schema is exposed to: a row whose
`audio_tokens` and `text_ids` lengths are wildly inconsistent is a misalignment,
and misalignments are the single most damaging thing in a TTS corpus because the
model learns to hallucinate rather than to read.

**D16 — our quality filter is the IV-R thresholds plus CosyVoice 3's length-ratio
rule, applied in `prepare_shards.py`.** The shard schema already carries `cer`
(computed in `Encoders.encode` as one ASR against the corpus transcript) and
`duration_s`, and `ShardDataset` already filters on `max_cer=0.15` and on
audio-token bounds. Three quantities are missing and should be **columns, not a
separate filtering pass**, because a column is re-thresholdable without re-encoding
900 hours of audio: **SNR, C50, and the audio-token / text-token length ratio**
(the third is free — both lengths are already computed). Adding them means
bumping `SCHEMA_VERSION` from 1 to 2, which is exactly what that constant is for:
the reader refuses a version it does not know rather than reading a renamed column
as the old one.

> **A threshold mismatch to resolve before the preprocessing job runs.** Our
> `max_cer` default is 0.15; IV-R's transcript-agreement rule is 0.05 and
> CosyVoice 3's is pairwise WER 0.15. These are not the same quantity — 0.05 CER
> between two human transcript levels is far stricter than 0.15 WER between three
> ASR systems. Since our `cer` column is *one ASR against the corpus transcript*,
> and the Indic ASR floor is itself 2–6% CER (§13.2), a 0.15 CER threshold is
> roughly "throw away anything more than ~3× the recogniser's own floor". That is
> defensible, but it should be chosen from the measured distribution of the column,
> not from a default. **Plot the histogram before picking the number.**

### 6.2 Does a small clean corpus beat a large dirty one? Not for cloning.

This is where the naive "quality over quantity" slogan breaks, and **E7** is the
experiment that breaks it. VoiceCraft (830M, GigaSpeech-pretrained) fine-tuned
two ways, both evaluated on the IV-R **zero-shot** benchmark (unseen speakers):

| fine-tuned on | NORESQA ↑ | **speaker similarity ↑** |
|---|---|---|
| **IndicTTS** (studio, ~2 speakers/language) | **3.83** | 78.93 |
| **IndicVoices-R** (restored in-the-wild, 10,496 speakers) | 3.64 | **88.18** |

Studio data wins on signal quality by 0.19 NORESQA. **Restored many-speaker data
wins on unseen-speaker similarity by 9.25 points.** And IV-R's own quality table
is honest that it is the worst of the four corpora on naturalness MOS:

| | N-MOS | SNR | C50 |
|---|---|---|---|
| LJSpeech | 4.36 | 58.03 | 58.56 |
| IndicTTS | 4.29 | 65.38 | 58.78 |
| LibriTTS | 3.76 | 59.60 | 57.33 |
| **IndicVoices-R** | **3.38** | 60.47 | 53.45 |

**Inference, and it is the central trade-off of this project:** filtering is for
*alignment correctness and gross defects*, not for *acoustic purity*. Throwing
away a speaker because their recording is 3 dB noisier throws away the diversity
that zero-shot cloning is made of. Filter on "is the transcript right, is the
alignment right, is the clip usable" and be permissive about "is it studio-clean".

[When Fine-Tuning Fails](https://arxiv.org/html/2603.10904) sharpens this from the
other direction on an architecture close to ours (NeuTTS: Qwen2.5-0.5B over a
neural codec, LoRA r=8). It found the predictor of whether fine-tuning helps or
hurts was **the speaker's energy standard deviation in dB**, not hours: speaker 2
(13.11 dB std, 3.9 h) gained +0.424 DNS-MOS; speaker 1212 (9.91 dB std, **18.5 h**)
lost 0.414; speaker 1401 (11.89 dB, 17.6 h) lost 0.274 — **while training and
validation loss kept improving throughout**. Acoustically flat data is worse than
less data, and the loss curve says nothing about it.

**D17 — carry an acoustic-diversity statistic per speaker and per language, and
treat low diversity as a defect.** Energy std in dB is a two-line computation at
preprocessing time. A corpus that is uniformly flat is a corpus that will produce
a flat model, and we will not find out from the loss.

---

## 7. Curriculum, mixing, and language balancing

### 7.1 Temperature sampling — what our loader already does, and what to set it to

`indic/train/data.py` implements temperature sampling as `p_l ∝ count_l ** t`,
with `t = 1.0` the natural distribution and `t = 0.0` uniform. Note this is the
**exponent convention**, i.e. `t = 1/T` in the more common `p ∝ n^(1/T)` notation,
and the docstring's example is exactly our problem: "175 hours of Hindi against
9 of Tamil".

The multilingual-speech literature's usual operating range is α ≈ 0.3–0.5 in the
exponent convention (equivalently T ≈ 2–3.3). I did not find a **TTS-specific**
per-language sampling ablation — mark the specific value **UNVERIFIED**.

**D18 — start at `language_temperature = 0.3`, and make the realised hours per
language a logged artefact, not an assumption.** `ShardDataset.summary()` already
emits `hours_by_language`, `kept_by_language` and `sampled_by_language`. Those
three dicts are the contract: after temperature sampling, the *sampled* hours per
language is the number that the E1 curve applies to, not the hours on disk. A run
whose sampled Tamil hours are below the floor in §11 is a run that will fail, and
it is knowable before the first step.

> **Oversampling repeats rows; it does not create data.** The docstring is right
> that a repeated row draws a different reference each time, which is real
> regularisation. But E3's Telugu experience (§5.2) is the warning: 20 more epochs
> over 8.7 hours "barely moves the loss". Temperature sampling makes a small
> language *seen more often*; it does not make it *more varied*. If Tamil has 50
> hours and Hindi has 175, `t=0.3` is appropriate. If Tamil has 5 hours, no
> temperature will save it and the answer is more Tamil.

### 7.2 English replay

E1's finding (§3.4) rules out a 50/50 English/Indic mix. It does not rule out
replay. E3's sibling repo uses **0.7 target / 0.3 replay**, and E1's own result
that multilingual beats bilingual (§4.3) means the English in the mix is doing
work beyond preservation.

**D19 — English replay at 20–30% of sampled steps, treated as a measured quantity
with its own gate.** Our situation differs from E1's in the one way that matters:
the Sarvam tokenizer swap makes English ~14% more expensive in tokens (findings
§3.1) and replaces every English embedding row. English is not a capability we are
preserving from a frozen table; it is one we are partially *re-learning*. Gate it:
English Set A CER must not regress by more than a stated margin against the base
checkpoint, measured at the same checkpoints as the Indic numbers. If it does not
regress at 20%, try 10% and reclaim the steps for Tamil.

### 7.3 Curricula — the one with evidence, and the one without

**With evidence: breadth before depth.**
[When Fine-Tuning Fails](https://arxiv.org/html/2603.10904) ran the comparison
directly, measuring zero-shot DNS-MOS on held-out speakers:

| fine-tuned on | zero-shot DNS-MOS on unseen speakers |
|---|---|
| **Pure FT** — one speaker, 3.5–18.5 h | 3.513 |
| **1 h × 3 speakers** (3 h total) | 3.628 (+0.114) |
| **2 h × 3 speakers** (6 h total) | **3.806 (+0.293)** |

Six hours spread over three speakers beat up to 18.5 hours from one. A "Mix FT"
arm using 11–22% of the per-speaker data of the pure baselines landed within 7–9%
of its speaker similarity. **For zero-shot cloning, speaker count dominates hours
per speaker.** This corroborates E7 (§6.2) from a second architecture.

**Also with evidence: incremental language addition works.** E3's six-round
curriculum (Hindi → +Telugu → Telugu-heavy → refine → +Kannada → all 8) produced
no observed forgetting and an improving Hindi CER, though without before/after
numbers (**UNVERIFIED**).

**Without evidence: short→long utterance curricula.** I found no TTS study
measuring it. It is a sensible prior — our memory budget is a pure function of
tokens/step (findings §3.4), so starting at seq 512 and extending to 1024 is free
throughput early — but it is a convenience, not a quality technique, and should
not be described as one.

**D20 — order the run by breadth, not by language.** Phase A (embedding warm-up)
on everything at once; Phase B (LoRA joint) on everything at once. Resist the
instinct to perfect Hindi first and add Tamil later: §4.3 and §7.3 both say joint
is better, and a sequential curriculum spends its first phase teaching the model a
single-language prior we would then have to undo.

---

## 8. Synthetic data, augmentation, and mined ASR corpora

*(Section populated from the parallel research stream; see §8.4 for the
license constraint that governs all of it.)*

### 8.4 What our own license forbids

The KittenTTS 2 Stellon Labs Community License §4b forbids using the model **or
its outputs** to train other models (findings §7). Three consequences, stated
plainly so nobody has to re-derive them:

- **No self-distillation into a separate student.** Whatever §9 concludes about
  distillation, a KittenTTS-2-derived teacher cannot supervise a model that is not
  itself a KittenTTS 2 derivative.
- **KittenTTS 2 output must never enter the training mix** — not as augmentation,
  not as pseudo-labels, not as a "1 hour of synthetic seed" in the style of E1's
  Bhojpuri/Tulu procedure.
- **Other TTS systems' outputs are governed by *their* licenses, and several are
  permissive.** IndicF5 is MIT; Indic Parler-TTS is Apache 2.0; Chatterbox is MIT.
  Using those to generate seed data is a license question with a clean answer,
  not a blocked path — but it is a *different* question from §4b and should be
  recorded per-source in the NOTICE the license requires anyway.

---

## 9. Distillation

*(Section populated from the parallel research stream.)*

---

## 10. The recommended data strategy

### 10.1 What exists, by language, in commercially usable licenses

Studio and semi-studio, from the [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts)
accounting (aggregating IndicTTS CC BY 4.0, LIMMITS CC BY 4.0, Rasa CC BY 4.0):

| | hours | utterances |
|---|---|---|
| Bengali | 140.04 | 70,305 |
| Marathi | 122.47 | 54,894 |
| Hindi | 107.00 | 46,135 |
| Telugu | 95.91 | 37,405 |
| **Tamil** | **52.25** | 29,204 |
| **total (our 5)** | **517.67** | **237,943** |

Many-speaker, from [IndicVoices-R](https://arxiv.org/html/2409.05356v2) (CC BY 4.0,
gated, 10,496 speakers across 22 languages):

| | hours |
|---|---|
| Telugu | 136.40 |
| Bengali | 111.99 |
| Tamil | 99.47 |
| Hindi | 74.60 |
| Marathi | 50.90 |
| **total (our 5)** | **473.36** |

**Combined: ~991 hours across the five languages**, all CC BY 4.0, all
commercially usable with attribution. This corroborates the brief's "realistic
clean usable ~800–900 h" from an independent accounting, and it is 10× the
100 h/L operating point of E1's best data arm.

> **Hours and speakers are not the same resource, and the catalogue hides it.**
> The full [Rasa release](https://huggingface.co/datasets/ai4bharat/Rasa) is
> 1,145.44 h over 22 languages at 48 kHz — far more than the 288 h the Indic
> Parler-TTS table attributes to it — but it is **44 speaker-language pairs: one
> male and one female per language**, ~20–30 h each. IndicTTS is the same shape
> (2 speakers/language). So the entire *studio* spine for our five languages,
> however many hundreds of hours it totals, contains roughly **10–20 distinct
> voices**. Every distinct speaker we have beyond that comes from IndicVoices-R
> (10,496 speakers) and the OpenSLR crowdsourced sets. Given §6.2 and §7.3, that
> makes IV-R not a nice-to-have but **the only source of the property zero-shot
> cloning is trained on**. Both figures are also gated and need accepted HF terms
> — findings §6 already flags this as the long pole.

Speaker diversity top-ups, cheap and multi-speaker:
[OpenSLR crowdsourced sets](https://www.openslr.org/resources.php) — SLR 64
Marathi, 65 Tamil, 66 Telugu, 37 Bengali. **There is no Hindi crowdsourced set**,
which matters because Hindi then draws its speaker diversity entirely from IV-R.

Deliberately **not** in the recommended mix:

- **Shrutilipi** ([CC BY 4.0, 6,400+ h, gated](https://huggingface.co/datasets/ai4bharat/Shrutilipi)).
  All India Radio news bulletins, **document-level** mined alignment. Three
  strikes for TTS: document-level alignment is exactly the misalignment risk §6.1
  warns about; news-read register is acoustically and prosodically narrow (§6.2's
  energy-std failure mode); and the speaker inventory is small and professional.
  It is an ASR corpus. Revisit only if a measured hours-vs-quality curve (§13)
  shows we are data-bound after IV-R, and then only with forced realignment.
- **IndicF5 / Indic Parler-TTS outputs** as synthetic data in phase 1. Licensing
  is clean (§8.4) but it is a bootstrap we have not shown we need.

### 10.2 The recommended strategy

**Target: 120–180 sampled hours per language, drawn from ~990 available, in one
joint multilingual run with English replay.**

| | what | why |
|---|---|---|
| **Corpora** | IndicTTS + LIMMITS + Rasa (studio spine) **+ IndicVoices-R (speaker breadth)** + OpenSLR crowdsourced where it exists | §6.2: studio alone costs 9.25 points of unseen-speaker similarity |
| **Filtering** | IV-R thresholds (C50 ≥ 30 dB, SNR ≥ 25 dB, 0.2–30 s, pitch ≤ 350 Hz, pitch std ≤ 150 Hz, ≤ 30 chars/s) + length-ratio trim (bottom 1%, top 5%) + ASR-agreement CER threshold chosen from the measured histogram | §6.1; permissive on acoustics, strict on alignment |
| **Text selection** | Greedy syllable/aksara-balanced selection when subsampling; never random | §12 rank 1; E4 measured CER 0.812 → 0.074 at 1 h |
| **Mixing** | `language_temperature = 0.3`; English replay 20–30% of sampled steps | D18, D19 |
| **Speakers** | maximise distinct speakers per language before maximising hours; log speaker count as a first-class metric | §7.3: 6 h over 3 speakers > 18.5 h over 1 |
| **Order** | Phase A embedding-only warm-up on all 6 languages → Phase B LoRA r=32 joint on all 6 | D20; findings §3.4 budgeted both |
| **Never** | KittenTTS 2 output in the mix; Shrutilipi without realignment; romanized input | §8.4, §10.1, D14 |

---

## 11. The minimum viable corpus

**The smallest corpus I believe reaches acceptable quality:**

> **~10 hours per language of syllable-balanced, multi-speaker, alignment-verified
> audio for Hindi, Bengali, Tamil, Telugu and Marathi — 50 hours of Indic — plus
> 10–15 hours of English replay. Call it 65 hours total.**
>
> With the strong caveat that this buys **intelligibility, not cloning**: expect
> per-language CER in the 0.10–0.25 range, and expect unseen-speaker similarity to
> be the thing that is not good enough.

The evidence for the floor, in order of how close it is to our situation:

1. **E3, our architecture, our languages.** ~10 h/language on a T3 + S3 25 Hz +
   s3gen stack with LoRA r=32 gave Hindi CER 0.106 and Tamil 0.161.
2. **E1, our languages, best-in-class Indic system.** 10 h/L retained MUSHRA 61.5
   of the 64.3 that 100 h/L reached, with monoglot WER *lower* at 10 h.
3. **E2, controlled, with an unseen-speaker set.** 12.65 h of one speaker, via a
   1.72% adapter, reached CER 4.906 — below its own ASR floor — on seen speakers.
4. **E4, our hardest language.** 1 h of *greedily syllable-balanced* Tamil reached
   CER 0.074 where 1 h of randomly selected Tamil reached 0.812, in a far weaker
   (FastPitch) model with no large-scale pretraining.

And the evidence for where the floor is **not**:

- **1 h/language is below the cliff.** E1: MUSHRA 33.7, WER 59.4.
- **10 h/language has a real failure rate.** E3's Malayalam reached CER 0.859 at
  the same ~10 h that gave a working Hindi. At least one of our five could do
  this, and the minimum corpus has no margin to absorb it.
- **10 h/language from 2 speakers will not clone.** E3 used one male and one
  female per language and explicitly did not evaluate speaker generalisation; E2
  showed what single-speaker adaptation does to unseen speakers (SMOS 2.35/4.88
  under full fine-tune); E7 showed studio-only costs 9.25 similarity points.

So the minimum viable corpus has a **composition requirement that is as binding as
its size**:

| | minimum | why |
|---|---|---|
| hours/language | **10 h** | E1, E3 |
| **distinct speakers/language** | **≥ 30** | §7.3, §6.2 — this is the constraint E3 violated |
| text selection | greedy aksara-balanced | E4, 11× CER difference at low data |
| English replay | 10–15 h | D19 |
| alignment verification | ASR-agreement on 100% of rows | §6.1 |

**The honest version of the recommendation:** 10 h/language is the *floor for a
demo*. **60 h/language (300 h Indic) is the floor for a product**, because that is
roughly where E1's curve has flattened and where IV-R can supply enough distinct
speakers for cloning to work. We have ~990 hours available; the reason to go to
120–180 sampled hours per language is not that the curve demands it, but that it
costs nothing — compute is ~25–40 A100-hours (findings §6) and the data is
already licensed.

---

## 12. Techniques ranked by payoff per unit of effort

Ranked by *measured* effect divided by *engineering days*, for our specific
situation.

| rank | technique | measured payoff | effort | evidence |
|---|---|---|---|---|
| **1** | **Greedy aksara-balanced text selection when subsampling** | CER **0.812 → 0.074** at 1 h Tamil; syllable coverage 49% → 73% | hours — Indic NLP Library + a greedy loop | E4 |
| **2** | **Include many-speaker data (IV-R), not just studio** | unseen-speaker similarity **78.93 → 88.18** | days — it is a download and a filter pass | E7 |
| **3** | **PEFT instead of full fine-tune** | unseen-speaker CER **13.45 → 7.22**; seen/unseen degradation 2.37× → 1.47× | **zero** — already our plan, for the wrong reason | E2 |
| **4** | **Brahmic phonetic warm-start of new embedding rows** | not isolated in any ablation — **UNVERIFIED**; but it is what the one working Indic system on our architecture did | an afternoon; `verify_surgery.py` already tests it | E3, D13 |
| **5** | **Alignment-correctness filtering (ASR agreement + length ratio)** | the thresholds two 1,000 h+ corpora were built with; no isolated ablation found | days, at preprocessing time | E7, E6 |
| **6** | **Maximise speaker count before hours per speaker** | 6 h / 3 speakers beat 18.5 h / 1 speaker by **+0.293 DNS-MOS** zero-shot | a sampling-policy change | 2603.10904 |
| **7** | **Temperature sampling across languages** | no TTS ablation found — **UNVERIFIED** value; mechanism is sound and Tamil is 4.7× worse in our own perplexity | zero — implemented | D18 |
| **8** | **Targeted volunteer recordings for OOV coverage** | OOV intelligibility error **−22% to −41% relative** from 48–64 minutes; 85% cheaper than studio; 2 working days | a recording session + ethics | E5 |
| **9** | **English replay at 20–30%** | protects a capability we are partly re-learning; E1 says 50% is too much | zero — a mixing weight | D19, E1 |
| **10** | **Acoustic-diversity screening (energy std)** | predicted fine-tune failure where hours did not; loss curve blind to it | two lines at preprocessing | 2603.10904 |

Items 1–3 are the three to do first, and together they cost less than a week.
Item 3 costs nothing because it is already the plan — but §5.1 means we should
stop describing it as a memory compromise and start describing it as the quality
choice it is.

**Item 8 deserves a note.** E5 is the most *under-rated* technique in this list and
the only one that creates data rather than selecting it. AI4Bharat recorded
1,983 Hindi words (48:07) and 2,098 Tamil words (64:32) from **volunteers, not
voice artists**, chosen by a greedy algorithm maximising missing character-bigram
coverage with a per-bigram cap of k=6 — and cut OOV intelligibility error by
40.59% (Hindi FastPitch) and 36.33% (Tamil FastPitch). Critically, **the
volunteers' voices improved the OOV performance of the model's original speaker**:
phonetic coverage transfers across speakers. One hour of deliberately chosen audio
beat a great deal of undirected audio. It is ranked 8th only because it has a
physical-world dependency; on pure payoff-per-hour-of-audio it is first.

---

## 13. What to measure — building the quality-vs-hours curve cheaply

### 13.1 The curve

The subsample ablation is cheap for us in a way it is not for most projects,
because **preprocessing is the long pole, not training** (findings §6: ~25–40
A100-hours for the whole thing). So:

**Preprocess once at full size; subsample at read time; train several small runs.**

`ShardDataset` already supports this — `languages=`, `language_temperature=` and
`max_cer=` all filter in-memory over an mmap'd table. The missing piece is an
hours cap. Add one knob, `--max-hours-per-language`, applied *after* filtering and
*before* temperature sampling, and the curve becomes four runs of the same script:

```
1 h/L   →  the cliff check: does it reproduce E1's collapse?
10 h/L  →  the minimum-viable point (§11)
30 h/L  →  the knee, if there is one
100 h/L →  the E1 reference point we can compare against directly
```

Four points, ~log-spaced, each a short run with a fixed step budget and a fixed
seed. Report against the same Set A / Set C harness. Two rules make the curve
mean something:

- **Fix total optimisation steps across the arms, not epochs.** Otherwise the 1 h
  arm does 100× more passes over its data and the comparison measures overfitting,
  not data scale.
- **Subsample by speaker, not by utterance.** Dropping utterances uniformly keeps
  every speaker and understates how much the small-data regime hurts cloning —
  which §6.2 and §7.3 say is the axis that matters. Report **hours and speaker
  count** on every point.

### 13.2 What to score it with, and the floors

The harness exists (`indic/eval/`): Set A from FLEURS with per-sentence ASR floors
(D9), Set C digit-pair robustness reported as a gap (D11), WavLM-based SECS (D8),
references never drawn from the target sentence (D10). What this research adds is
**external reference values**, so a number can be judged rather than merely
tracked.

**ASR floors — IndicConformer-600M on ground-truth recordings**
([arXiv 2610.14826](https://arxiv.org/html/2610.14826)):

| | WER | CER |
|---|---|---|
| Marathi | 10.56 | 2.08 |
| Hindi | 14.40 | 5.21 |
| Telugu | 21.06 | 3.07 |
| Bengali | 24.78 | 6.48 |
| Tamil | 25.39 | 4.87 |
| English (Whisper large-v3) | 3.54 | 1.33 |

These confirm D9 quantitatively: **Tamil's floor WER is 1.8× Hindi's and 7.2×
English's.** A raw Tamil WER of 30% and a raw Hindi WER of 20% describe roughly
the same model. Report `cer_above_floor` or report nothing.

**Reference points to aim at:**

| system | number | source |
|---|---|---|
| IN-F5 / IndicF5 | MUSHRA 80.5, WER 19.2, S-SIM 97.3 (human: 91.8 / 18.4 / 97.0) | [E1 Table 6](https://arxiv.org/html/2505.20693v1) |
| chatterbox-indic-lora, ~10 h/lang | CER hi 0.106, ta 0.161, bn 0.245, te 0.285, mr 0.198 | [E3](https://huggingface.co/reenigne314/chatterbox-indic-lora) |
| CosyVoice 3-0.5B, 1M h | test-en WER 2.46, SS 0.747 | [E6](https://arxiv.org/html/2505.17589v2) |

### 13.3 Three measurements the current harness cannot make

1. **An unseen-speaker split.** Every cloning number in §5.1 and §6.2 is a
   *seen vs unseen* comparison, and that gap is the whole finding. Set A must be
   partitioned by speaker, with the reference pool drawn from speakers absent from
   training. Without this we cannot detect the failure mode that E2 shows is the
   dominant one.
2. **A long-form / hard-input set.** §3.3 says test-hard is where data lives
   (CER 27.60 → 6.96 across 57× data). Set C covers numerics; it does not cover
   multi-sentence stability, which is where an AR speech-LM at 25 Hz is most
   exposed. A "Set D" of 3–5 sentence paragraphs, scored for CER *and* for
   duration ratio against the reference reading, would catch runaway generation.
3. **A SECS metric that is not saturated.** E1's SSIM sat at 92.7–94.0 across a
   sweep that included a model with WER 59.4. If our WavLM SECS behaves the same
   way, it is not a gate. Calibrate it first: measure same-speaker (ceiling) and
   different-speaker (floor) similarity on held-out recordings — `metrics.py`
   already contemplates exactly this — and report where our model sits *between*
   them, not its raw cosine.

---

## 14. Risks — what degrades with little data, and how it shows up

Ordered by how likely they are to bite us, with the metric each one hides behind.

### R1 — Speaker generalisation collapses first, and silently

**Evidence.** E2: full fine-tune on one speaker, seen-speaker CER 5.671 → unseen
13.454 (2.37×), SMOS 2.35 against a ground-truth 4.88. E7: studio-only training
loses 9.25 points of unseen-speaker similarity. 2603.10904: breadth beat depth by
+0.293 DNS-MOS zero-shot.

**How it shows up.** It does *not* show up in training loss, in seen-speaker CER,
or in a saturated SSIM. It shows up only in a seen-vs-unseen gap, which §13.3
says we currently cannot measure.

**Mitigation.** PEFT over full fine-tune (D15); IV-R in the mix (§10.2); maximise
speaker count (§7.3); build the unseen-speaker split before the first real run.

**Our specific exposure.** Findings §3.7 could not confirm the reference layout —
`role_swapped` cost −0.016 nats and the whole reference prefix was worth 0.062.
That gate was blind to conditioning by construction (teacher forcing predicts
audio tokens from their predecessors). **R1 and the unresolved reference layout
are the same measurement.** Generating from both layouts and scoring unseen-speaker
similarity settles both at once, and nothing else settles either.

### R2 — Hard-input and long-form stability degrade while easy sentences look fine

**Evidence.** E6: 3,000 h → 170,000 h moved easy Chinese CER only 1.68 → 1.27 but
test-hard 27.60 → 6.96.

**How it shows up.** A headline Set A CER that looks acceptable, and a product
that falls apart on a paragraph. For an AR model over 25 Hz tokens the failure is
usually runaway or truncated generation, which a CER computed on a short clip can
even *flatter* — a truncated utterance has fewer tokens to get wrong.

**Mitigation.** The Set D long-form set in §13.3, scored on CER **and duration
ratio**. Duration ratio is the cheap detector: a generation 1.6× or 0.5× the
reference length is broken regardless of what its CER says.

### R3 — One of the five languages simply fails

**Evidence.** E3: Malayalam CER 0.859 at the same ~10 h that produced Hindi 0.106.
Telugu at 52 h came out worse than Tamil at 10 h. Indic Parler-TTS reports
native-speaker scores *dropping* after fine-tuning for Konkani (87.20 → 76.60) and
Manipuri (89.58 → 85.63).

**How it shows up.** A per-language CER an order of magnitude off the others. It
is only visible if every metric is reported per language, never pooled.

**Our specific exposure.** Tamil: perplexity 287.2 against English 60.9, worst
Qwen3 fertility at 10.13 tokens/word, and the least available data of our five
(52.25 studio hours). Tamil is the arm most likely to be R3. It is also the arm
where D13's Brahmic warm-start and E4's aksara-balanced selection should help most,
which makes it the right place to test both.

### R4 — Prosody and expressiveness flatten before intelligibility does

**Evidence.** E4's expressive curve: with 5 h neutral, a single emotion needs
≥ 15 min to be rated "Fair" and ≥ 30 min to be rated "Good"; at 7.5 min the scores
are 38–43 Exp-MUSHRA. And more *neutral* data raises expressive scores even with
expressive data held fixed (10 h + 0.5 h = 66.06 beats 5 h + 2 h = 65.50).

**How it shows up.** CER stays fine. The model becomes a competent newsreader with
one register. No objective metric in our harness detects this.

**Mitigation.** Rasa is already in the recommended mix and carries six Ekman
emotions with per-emotion tags, and our shard schema already has an `emotion`
column. Keep expressive data in from the start rather than adding it later — E4's
table says neutral hours and expressive hours are not substitutes but they are
complements.

### R5 — The loss curve lies

**Evidence.** 2603.10904: training and validation loss improved monotonically
while DNS-MOS fell 0.414 for a low-energy-variance speaker. E3's Telugu: 20 extra
epochs "barely moves the loss" on a saturated corpus.

**How it shows up.** It does not. That is the point.

**Mitigation.** Never gate on loss. Gate on the Set A/C/D harness at fixed
checkpoints, with per-language reporting, floors subtracted, and a seen/unseen
speaker split. This is the same discipline D5 applied to the training loop: a
check that cannot fail is worse than no check, and a loss curve that cannot
detect the failure mode is not a check.

### R6 — Oversampling a small language is mistaken for having more of it

**Evidence.** E3's Telugu saturation (§7.1). Mechanism: temperature sampling
repeats rows.

**How it shows up.** `sampled_by_language` looks balanced; `kept_by_language`
does not. Both are already emitted by `ShardDataset.summary()`.

**Mitigation.** D18 — log and gate on *distinct* hours and *distinct speakers* per
language after filtering, and treat the repeat factor (`sampled / kept`) as a
quantity with a ceiling. A repeat factor above ~4 means the honest statement is
"we do not have enough of this language", not "we balanced it".

---

## 15. Open questions

- **Is there any controlled flow-matching-vs-AR data-efficiency comparison?** I
  could not find one (§3.1). If one exists it would change nothing about our plan
  but would settle a claim the project has been carrying.
- **What temperature?** No TTS-specific per-language sampling ablation found
  (§7.1). `t = 0.3` is borrowed from multilingual ASR/MT practice.
- **Does the Brahmic warm-start actually help?** Nobody has ablated it (§12 rank 4).
  We can be the ones who do: it is an arm in a harness that already exists.
- **Why did Malayalam fail at 10 h in E3, and is Tamil at risk of the same?**
  Unanswered anywhere. This is the single most important unknown for our Tamil arm.
- **What is the real English cost of the tokenizer swap after fine-tuning?**
  Findings §3.1 measured +14% tokens; nobody has measured the quality cost, and
  D19's replay ratio is set against an unmeasured risk.
