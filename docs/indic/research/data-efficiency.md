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

**And the brief's architecture hypothesis is backwards, in our favour.** The one
study that trains an AR speech-LM and a flow-matching model over the same
5k→134k-hour sweep found the **AR model reached 95% of its speaker-similarity
ceiling at 3.7% of the data** (0.587 → 0.620) while the flow-matching model
needed all of it (0.418 → 0.606)
([Emilia](https://arxiv.org/html/2501.15907v2)). For a low-data zero-shot cloning
project, ours is the right family.

**Three things follow, and they organise everything below.**

1. **Speaker similarity is bought with *speakers*, not hours** — it saturates
   around 500 distinct voices, and 15 speakers × 10 minutes beats 5 speakers
   × 30 minutes by 34% relative CER at identical total duration
   ([Yong et al.](https://www.isca-archive.org/interspeech_2025/yong25_interspeech.pdf)).
   Our studio spine holds roughly **10–20 voices across all five languages**.
   That, not hours, is the binding constraint.
2. **Hours buy hard-case robustness**, which is our architecture's characteristic
   weakness: 57× the data moved easy CER 1.68 → 1.27 and hard CER
   **27.60 → 6.96** (E6). We will not fix this with 900 hours; we will detect
   it and work around it.
3. **Everything that fails here fails quietly.** Validation loss is compatible
   with degrading naturalness, destroyed English, and an exhausted corpus. Section
   13 is therefore not an afterthought — it is the deliverable that makes the
   rest of the plan falsifiable.

---

## 2. The evidence base

Nine studies carry most of the weight. Everything in sections 3–9 is drawn from
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

**There is exactly one controlled comparison, and it points the other way.**
[Emilia's scaling study](https://arxiv.org/html/2501.15907v2) trained an AR
speech-LM (AR + SoundStorm) and a flow-matching model (VoiceBox) over the same
5k → 134k-hour sweep. At 5,000 hours the AR model scored **S-SIM 0.587** and the
flow-matching model **0.418**; at 134k they were 0.620 and 0.606. The AR model
reached 95% of its ceiling at 3.7% of the data; the flow-matching model needed all
of it. §3.3 unpacks this.

Every other citation chain I followed terminates either in an inference-speed
result (flow matching is faster to sample, which is true and irrelevant here) or
in a 2022 observation that AR architectures have "higher data requirement compared
to non-AR TTS" ([Do et al., arXiv 2306.00535](https://arxiv.org/pdf/2306.00535),
citing Pine et al., ACL 2022) — made about Tacotron 2 against FastSpeech 2, two
architecture generations before discrete-codec speech-LMs existed.

**Treat "flow matching is more data-efficient" as refuted for speaker similarity
and unproven elsewhere.** It is also *not actionable*: we are not choosing an
architecture. We have a 1.7B ternary Qwen3 speech-LM with a proven codec and a
proven training loop. The useful question is what the evidence says about adapting
*this* — and the answer, happily, is that it is the right family for a low-data
cloning project.

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

### 3.3 What hours actually buy — and the finding that inverts the brief

Five independent scaling studies agree on an asymmetry, and it is the opposite of
what the brief assumed.

| study | architecture | data range | **SIM gain** | **WER/CER gain** |
|---|---|---|---|---|
| [Emilia](https://arxiv.org/html/2501.15907v2) AR+SoundStorm | **AR speech-LM** | 5k → 134k h (27×) | **0.587 → 0.620** | 5.2% → 4.2% |
| [Emilia](https://arxiv.org/html/2501.15907v2) VoiceBox | **flow matching** | 5k → 134k h (27×) | **0.418 → 0.606** | 7.1% → 6.4% |
| [MELLE](https://arxiv.org/html/2407.08551v1) | AR | 960 h → 50k h (52×) | 0.591 → 0.625 | 2.80 → 2.10 |
| [ZipVoice](https://arxiv.org/html/2506.13053v3) | flow matching | 555 h → 100k h (180×) | 0.610 → 0.668 | 1.69 → 1.64 |
| [NaturalSpeech 3](https://arxiv.org/html/2403.03100v3) | diffusion | 1k → 200k h (200×) | 0.64 → 0.73 | **3.94 → 2.11** |
| **E6 CosyVoice 3-0.5B** | AR LM + CFM | 3k → 170k h (57×) | 0.710 → 0.815 | **1.68 → 1.27; hard 27.60 → 6.96** |

> **The brief's hypothesis — "flow-matching models need far less data than AR
> speech-LMs" — is contradicted by the one study that trains both on the same
> data at the same sizes.** Emilia trained an AR speech-LM and a flow-matching
> model over the identical 5k → 134k sweep. The **AR model started at SIM 0.587
> and ended at 0.620 — 95% of its ceiling at 3.7% of the data.** The
> flow-matching model started at **0.418** and needed the full 134k hours to
> reach 0.606. On this axis the AR model is the *more* data-efficient of the two,
> not the less.

**But the real variable is speaker count, not hours, and that reconciles every
row above.** ZipVoice's "555 hours" is LibriTTS, which carries ~2,456 speakers —
already past the ~500-speaker knee that §7.4 identifies. Emilia and MELLE scale
within corpora that are speaker-rich at both ends. CosyVoice 3's 3k → 170k and
Emilia's VoiceBox row are the two that move most on SIM, and both are adding
speakers along with hours. **Hours that add speakers buy similarity; hours that
add more of the same speakers do not** — which is exactly what E9 measured
directly (§7.4).

So the decomposition, stated correctly:

- **Easy-sentence intelligibility saturates early.** E6: 57× the data moved
  in-domain CER 1.68 → 1.27. On short clean read sentences, a few thousand hours
  already looks finished.
- **Speaker similarity is bought with speakers, not hours**, and saturates around
  500 distinct voices. An AR speech-LM reaches most of its similarity ceiling
  early *provided the speaker inventory is deep*.
- **Hard inputs are where hours genuinely live.** E6 test-hard CER **27.60 →
  6.96**. Long, repetitive, numeric, code-mixed input is 4× better with 57× the
  data, and this is the **AR-specific repetition/looping failure mode**. VALL-E 2's
  repetition-aware sampling and CosyVoice 3's RL stage exist solely to fix it.

**Inference, and the organising idea of this document.** Our architecture is the
data-efficient one for cloning *if* we feed it enough distinct voices. Our
exposure is not speaker similarity — it is **hard-case robustness and the
per-language intelligibility of the languages the base model does not already
carry**. Section 13 turns that into a measurement plan; section 14 into a risk
register; and §10.1's speaker accounting is why IndicVoices-R and Vaani are not
optional.

> **A warning aimed squarely at D1.** [CosyVoice's own tokenizer
> ablation](https://arxiv.org/html/2407.05407v2) (Table 7) trained the same 300M
> model on **585 h of LibriTTS** with two text front-ends: English-only BPE
> (Exp-3) scored **WER 3.93 / SS 67.85**; **multilingual BPE + multilingual speech
> tokenizer (Exp-4) scored 4.76 / 65.94 — worse on both.** The multilingual setup
> *won* at 171K h. **A multilingual tokenizer costs you at low data and pays off
> only at scale**, and this is the only controlled evidence on the question.
> We are swapping to a multilingual tokenizer at ~900 hours. D1 already keeps a
> Qwen3 control arm for exactly this reason; this result says that control arm is
> not a formality.

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

### 6.2 How hard to filter — three retention curves that agree

The slogan "quality over quantity" is directionally right and quantitatively
dangerous. Three independent ablations put the optimum in the same place.

**[Raon-OpenTTS](https://arxiv.org/html/2605.20830v1) — filtered vs unfiltered,
fixed update steps** (0.3B models, so the gap is data quality, not training
length). Average rank across five test conditions, lower better:

| filtering | Seed-TTS WER | Seed SIM | CV3-Hard WER | avg rank |
|---|---|---|---|---|
| **combined rank, prune 15%** | **2.00** | **0.672** | 8.14 | **3.40 (best)** |
| DNSMOS only, prune 15% | 1.97 | 0.669 | 7.33 | 3.60 |
| **no filtering** | 2.19 | 0.661 | 8.53 | 4.90 |
| combined rank, **prune 50%** | 2.32 | 0.672 | 7.56 | **4.90 (tied with none)** |
| WER only, prune 50% | 2.20 | 0.655 | 9.88 | 7.60 (worst) |

Filtering 15% bought WER 2.19 → 2.00 and SIM +0.011. **Pruning 50% was no better
than not filtering at all.** Their conclusion is verbatim: *"moderate filtering
(15%) outperforms more aggressive filtering (50%)."*

**[Confidence-based filtering](https://arxiv.org/html/2601.12254) — the sharpest
retention curve published** (280,130 utterances, enhanced, then kept by rank):

| retained | UTMOS ↑ | DNSMOS ↑ | WER ↓ |
|---|---|---|---|
| 100% (unfiltered) | 3.64 | 3.10 | 20.45 |
| 90% | 3.76 | 3.14 | 19.29 |
| **80%** | **3.80** | **3.17** | 18.79 |
| **70%** | 3.76 | 3.15 | **18.14** |
| 60% | 3.73 | 3.12 | 18.78 |
| **50%** | 3.68 | 3.11 | **20.18 — back to unfiltered** |

**At 50% retention the entire benefit of filtering is gone.** Not "diminished" —
gone.

**A third, from the other direction:** a [flow-matching zero-shot
TTS](https://arxiv.org/html/2406.05699v1) swept the DNSMOS threshold and found
2.6 / 2.8 / 3.0 produced SIM-o of 0.746 / 0.749 / 0.746 — **within noise of each
other**. *Having* a threshold moved SIM-o +0.042; the exact value did not. The
cheapest single filter in that study was **multi-speaker detection alone**
(discarding 23.3% of pre-training data for +0.012 SIM-o clean, +0.019 noisy).

> **And the counterweight: skipping the gate entirely is expensive.**
> [TITW](https://arxiv.org/html/2409.08711v2) built two corpora from the *same*
> source at the *same* ~180 h, differing only by DEMUCS denoising plus
> DNSMOS ≥ 3.0. Training on the ungated version: GradTTS **UTMOS 2.18 → 1.29,
> WER 11.9% → 26.2%**; TransformerTTS and MQTTS **failed to converge at all.**
> The floor matters enormously. The ceiling barely does.

**D16a — target 75–85% retention via a combined rank, not independent gates.**
Raon's method is the only approach shown to beat every single-criterion filter:
rank each segment on DNSMOS, ASR disagreement and VAD speech-ratio, average the
ranks, cut the bottom 15–20%. It is also self-adapting — under one global
threshold their retention ran 97.7% on LibriTTS-R and 48.2% on the dirtiest
source. **Set one global threshold; do not hand-tune per corpus.**

> **Two threshold traps that would cost us real data.**
>
> **(1) P.808 is not P.835.** WenetSpeech4TTS's famous 3.6 / 3.8 / 4.0 tiers are
> **DNSMOS P.808**, scored *after* enhancement. On the same audio, P.808 reads
> roughly 0.5–0.8 higher than P.835 OVRL — Emilia's table puts the WenetSpeech4TTS
> corpus at P.835 OVRL **3.18** while its P.808 tiers sit at 3.6–4.0. Copying
> 3.8 into a P.835 pipeline would discard nearly everything.
>
> **(2) DNSMOS is not reliable per clip.** [Its own
> paper](https://arxiv.org/html/2110.01763v2) reports per-**model** PCC 0.98 for
> OVRL but per-**clip** PCC **0.74**, against a human-human upper bound of 0.77.
> Per-clip DNSMOS is barely better than the noise floor of human agreement. Use
> it pooled, or as one term in a combined rank — never as a hard per-clip gate.

Published P.835 OVRL thresholds, for calibration: Emilia-Pipe **≥ 3.0** ("the
lowest 'fair' quality anchor"), Emilia-YODAS lowered to **2.4** "to preserve more
data", FireRedTTS **> 3.3**, MOSS-TTSD **≥ 2.8** stage 1 / **≥ 3.4** stage 2,
flow-matching optimum **2.8**. Corpus reference values to check our own scorer
against: MLS 3.33 · LJSpeech 3.30 · Emilia 3.26 · LibriTTS 3.25 · VCTK 3.20 ·
WenetSpeech4TTS 3.18 · GigaSpeech 2.52.

### 6.3 Does a small clean corpus beat a large dirty one? Not for cloning.

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

### 6.4 The two-stage quality curriculum — train on most, polish on the cleanest

This is the only protocol with two independent confirmations, and it resolves the
tension between §6.2 (don't over-filter) and §6.3 (don't over-purify).

[WenetSpeech4TTS](https://arxiv.org/html/2406.05763v3) trained on Basic (7,226 h,
DNSMOS P.808 > 3.6), then fine-tuned on Standard (4,056 h, > 3.8), then on Premium
(945 h, > 4.0). On **unseen speakers**:

| | CER unseen | SECS unseen | NMOS | SMOS |
|---|---|---|---|---|
| VALL-E, Basic only | 14.96 | 0.781 | 3.27 | 3.37 |
| VALL-E, → Premium FT | **9.82** | 0.781 | 3.45 | 3.58 |
| NaturalSpeech 2, Basic only | 14.06 | 0.756 | 2.64 | 2.95 |
| NaturalSpeech 2, → Premium FT | **7.89** | 0.783 | 3.36 | 3.45 |

**Unseen-speaker CER fell 34% and 44% — and SECS barely moved** (0.781 → 0.781;
0.756 → 0.783). That is the same asymmetry as §3.3, from a fourth angle:
**filtering buys intelligibility, not speaker similarity.**
[MOSS-TTSD](https://arxiv.org/abs/2603.19739) does the same thing with explicit
thresholds — stage 1 DNSMOS ≥ 2.8, stage 2 DNSMOS ≥ 3.4 **and** true sample rate
≥ 24 kHz at a lower learning rate.

> **Stated honestly, because it is widely over-claimed:** this is *sequential
> fine-tuning*, not a controlled comparison. **Nobody ever trained on Premium's
> 945 h alone**, so WenetSpeech4TTS does not show that a small clean corpus beats
> a large dirty one. It shows that you can extract value from both.
>
> And **cap the polish stage.** The [flow-matching
> study](https://arxiv.org/html/2406.05699v1) found noisy-prompt SIM-o *declining*
> from 0.659 to 0.633 as fine-tuning ran 0.16M → 1.6M steps, attributed to the
> narrower speaker distribution of the clean set. Over-polishing costs cloning
> generality — which is R1 arriving by a different road.

**D17a — two-stage quality curriculum: train on the ~80% kept by D16a, then
fine-tune at reduced LR on the cleanest ~15–25%, with a hard step cap on the
second stage and the unseen-speaker split (§13.3) watching it.**

---

## 7. Curriculum, mixing, and language balancing

### 7.1 Language balancing — flatter than I first thought, and with a cap

`indic/train/data.py` implements `p_l ∝ count_l ** t`, with `t = 1.0` the natural
distribution and `t = 0.0` uniform. This is the **exponent convention**: our `t`
is the literature's α in `p ∝ (n/N)^α`, equivalently `t = 1/T`. The docstring's
example is exactly our problem — "175 hours of Hindi against 9 of Tamil".

Published values cluster at α ≈ 0.3–0.5 (XLM-R and mT5 at 0.3; XLS-R and
Omnilingual ASR pretraining at 0.5; NLLB-lineage MT at 0.2). **But the two
controlled sweeps both favour going flatter than that**, and one of them is at our
exact model scale.

[Omnilingual ASR](https://arxiv.org/pdf/2511.09690) swept (β_corpus, β_lang) on a
1B CTC model over 200k steps, CER by resource bucket:

| (β_corpus, β_lang) | High | Med | **Low** | Avg |
|---|---|---|---|---|
| **(0.0, 0.0) — uniform** | **6.28** | **6.12** | **21.14** | **11.18** |
| (0.5, 0.25) — what they shipped | 6.54 | 7.86 | 23.09 | 12.89 |
| (1.0, 1.0) — no upsampling | 6.75 | 8.09 | **26.40** | 13.75 |

**Uniform beat no-upsampling by 20% relative on low-resource *and* was
simultaneously better on high-resource.** No trade-off at this scale. They shipped
(0.5, 0.25) anyway — not for quality, but because uniform over-indexed on the
narrow audio domain of one corpus that dominated their low-resource languages.
That caveat lands directly on us (see the warning below).

[UniMax (ICLR 2023)](https://arxiv.org/pdf/2304.09151) explains why flat wins and
adds the missing guardrail: at τ=3.33 the lowest-resource languages get **repeated
over 100 times**, and overfitting from repetition "only emerges with scale… it
becomes conspicuous at XXL size" and with longer training. They quote Hernandez et
al.: **"even repeating 0.1% of data 100 times can be as harmful as halving model
size."** UniMax's answer is uniform-subject-to-a-hard-epoch-cap, and its ablation
says **N=1 epoch (no repeats) was best** (82.2 TyDi QA) over N=5 (81.5) and N=10
(81.8). Also striking: τ=1 allocates 47.7% of training to English and buys only 1%
over UniMax, which allots English **1%** — massively down-weighting the strong
language costs almost nothing.

**E8 is the decisive one, because it is our model.** Polyglot-Lion fine-tuned
**Qwen3-ASR-1.7B on 968.83 h across 4 languages including Tamil, on one GPU in
48 hours**. They *explicitly rejected* temperature sampling — "introduces a
sensitive temperature hyper-parameter and still does not guarantee exact language
parity" — for a deterministic two-stage upsample (equalise corpora *within* a
language, then equalise languages), landing at **exactly 25% per language** from a
raw distribution that was 65% English+Mandarin and <8% Malay:

| | base Qwen3-ASR-1.7B | Polyglot-Lion-1.7B |
|---|---|---|
| Tamil CommonVoice WER | 139.96 | **39.19** (−72%) |
| Tamil FLEURS WER | 147.23 | 37.28 |
| Malay Mesolitica WER | 39.00 | 21.51 (−45%) |
| **English LibriSpeech WER** | 2.31 | **2.10 (improved)** |
| **English CommonVoice WER** | 7.50 | **4.91 (improved)** |
| **Mandarin AISHELL-1 CER** | 1.52 | **1.45 (improved)** |
| average over 12 benchmarks | 53.76 | **14.85** |

Their conclusion: "a 72% relative reduction **without any degradation on English
or Mandarin**… balanced upsampling does not introduce negative transfer on
high-resource languages."

> **This supersedes my first draft, which recommended `t = 0.3`.** α=0.3 is the
> text-LM convention (mT5, XLM-R) and I reached for it by habit. Every controlled
> *speech* sweep says flatter, and the one experiment at our scale and backbone
> flattened all the way to uniform and lost nothing on its strong languages.

**D18 — go near-uniform across the five Indic languages (`language_temperature`
0.0–0.2), hold English at a fixed 20–25% share, and cap epochs per language.**
Three parts, all of which need code:

1. **Near-uniform Indic.** `t = 0.0` is the E8/Omnilingual recommendation;
   `t ≤ 0.2` if we want a little of the natural distribution back.
2. **English as a fixed share, not as a temperature outcome.** English is not
   one of five peers — it is replayed at a ratio we are choosing on different
   grounds (§7.2). Our current loader folds it into the same exponent, which
   couples two decisions that should be independent. **Treat English's share as a
   separate parameter.**
3. **An epoch cap, which we do not have.** `_resample_languages` draws
   `replace=True` with no ceiling. UniMax says this is the failure mode that
   appears *at large model size and long training* — i.e. ours, not the regimes
   where α≈0.3 was validated. Add a cap and refuse to exceed it: if Tamil would
   need 6 repeats to reach parity, the honest output is a warning, not a silent 6×.
4. **Balance within a language before balancing between languages** (E8 stage 1).
   Otherwise whichever Hindi corpus is largest silently *defines* Hindi.

`ShardDataset.summary()` already emits `hours_by_language`, `kept_by_language` and
`sampled_by_language`. Those three dicts are the contract, and
`sampled / kept` is the repeat factor the cap applies to.

> **Carry Omnilingual's warning, because it is sharper for us than for them.**
> Flat balancing makes the model over-fit the *recording domain* of whichever
> corpus dominates each language. Our Indic data is mostly studio/read (Rasa,
> IndicTTS, LIMMITS) while IndicVoices-R is in-the-wild and English is something
> else again. Uniform *per language* can therefore be badly non-uniform *per
> recording condition*. Consider balancing per (language × corpus-type) bucket —
> which is also exactly what E8's stage-1 within-language equalisation does.

### 7.2 English replay — 20–25%, and the number is well supported

E1's finding (§3.4) is the only TTS datapoint and it is an outlier, for the reason
§3.4 gives: its mix was 50/50, and it **never measured English at all** (the paper
says so: "It is possible that this setup reduces English performance, but that is
not a concern, as we can always use the original checkpoint for English"). We do
not have that option.

The adjacent literature is unusually consistent, and spans speech-LMs specifically:

| source | domain | replay | effect |
|---|---|---|---|
| [Hsiao et al., Interspeech 2025](https://www.isca-archive.org/interspeech_2025/hsiao25_interspeech.pdf) | **speech-LM, ASR→TTS→SQA** | **0.5%** | LLaMA-Q 70.0 → 14.3 (no replay) → **66.3** |
| [Lifelong multilingual TTS, ICASSP 2022](https://arxiv.org/pdf/2110.04482) | **multilingual TTS** | ~1–4% (0.6 h buffer) | 87% of full joint training's MCD benefit |
| [LoRA-Whisper, Interspeech 2024](https://arxiv.org/html/2406.06619v1) | ASR language expansion | 5 h per base language | forgetting **+20.6 → +1.7 WER** |
| [Ibrahim et al. 2024](https://arxiv.org/html/2403.08763v4) | LLM continual pretrain, **strong shift** | **25%** (their choice) | forgetting +1.39 → **+0.16** loss |
| [Elhady et al., ACL 2025](https://arxiv.org/html/2506.00288v1) | LLM language adaptation | **20%** | source retention **20.12 → 43.43** |
| **E8 Polyglot-Lion** | **ASR, Qwen3 1.7B** | **25%** (uniform share) | **English improved** |
| E1 / IN-F5 | TTS, 11 Indic | **50%** | −7.2 MUSHRA on Indic; English unmeasured |

Two results deserve to be stated at full force.

**The forgetting hazard is large and fast.** LoRA-Whisper: naive full fine-tuning
on new-language data alone took base-language WER **11.68 → 32.29 (2.8×)**, with
the worst language going **14.37 → 68.93**. Five hours of replay per base language
took that from +20.6 WER points to **+1.7**. And Elhady measured the parameter
drift: cumulative L2 from init was **7× larger without English by step 100, 15×
larger by step 1000**. The damage happens in the first few hundred steps, which
matches Hsiao's finding that **"the most serious forgetting appears in the TTS
stage"** — our stage.

**LoRA may dominate replay entirely.** LoRA-Whisper's full table:

| | base languages WER | new languages WER |
|---|---|---|
| full FT on new data only | 32.29 | 28.33 |
| full FT + 5 h replay | 13.35 | 30.13 |
| **LoRA, no replay at all** | **9.52 (unchanged)** | **22.97** |

LoRA beat full-FT-plus-replay **on every axis simultaneously**, and left base-
language WER bit-identical because the backbone never moved. Caveat: that is
encoder-decoder ASR with one-at-a-time language expansion, not our
decoder-only-TTS-with-five-languages-at-once. But it is a second independent
reason (after §5.1) that our LoRA plan is the right one.

**D19 — English replay at 20–25% of sampled steps, declared as its own parameter,
gated at every checkpoint, and evaluated early.** Our situation is a *strong*
distribution shift — five new scripts and new phonology — so Ibrahim's 25%, not
his 5%. Two amplifiers specific to us: the Sarvam swap replaces every English
embedding row and makes English ~14% more expensive in tokens (findings §3.1), so
English is partly being *re-learned*, not merely preserved; and ternary weights
mean less spare capacity than any of these full-precision studies assumed. Both
argue for the high end.

**Gate it:** English Set A CER must not regress beyond a stated margin against the
base checkpoint, measured at **step 100, 250 and 500** as well as at the end —
because Elhady and Hsiao both say that is when it happens.

> **And do not use validation loss as the forgetting signal.** Elhady's clearest
> result: with vs without English replay, validation perplexity was 3.35 vs 3.58 —
> essentially identical — while downstream accuracy differed by 5.3 points and
> source-language retention by **23 points (43.43 vs 20.12)**. A healthy Indic
> loss curve is fully compatible with an English capability that has been
> destroyed. Nor can it be repaired afterwards: Hsiao tested model merging (Linear,
> TIES, DARE) and LoRA-scaling as post-hoc fixes and **all of them recovered
> essentially nothing** — DARE made it worse (4.0 against a 70.0 original).

**Two cheap alternatives worth an arm each**, both of which matched full mixing in
Elhady's setting at zero cost in Indic tokens: **English-only for the first ~10%
of steps, then Indic-only** (PPL 3.08, retention 42.94 vs mixing's 43.43); and a
**weight EMA with no English at all** (α=0.92 → retention 42.66). The EMA interval
needed per-language tuning, which is a real cost, but neither spends Indic data.

### 7.3 Curricula

**Breadth before depth, and it is now triply confirmed.**
[When Fine-Tuning Fails](https://arxiv.org/html/2603.10904), zero-shot DNS-MOS on
held-out speakers:

| fine-tuned on | zero-shot DNS-MOS on unseen speakers |
|---|---|
| one speaker, 3.5–18.5 h | 3.513 |
| 1 h × 3 speakers (3 h total) | 3.628 (+0.114) |
| **2 h × 3 speakers (6 h total)** | **3.806 (+0.293)** |

See §7.4 for E9, which runs the same comparison with total duration held exactly
constant and reaches the same answer much more sharply.

**Order by difficulty, and difficulty is not length.**
[Karakasidis et al., Interspeech 2022](https://www.isca-archive.org/interspeech_2022/karakasidis22_interspeech.pdf)
argues explicitly that "the standard practice of assuming that longer utterances
are harder to learn is not the best solution", and scores difficulty from model
loss or recognition error instead. A paced loss-ordered curriculum **matched a
full-data baseline having seen 44% fewer hours** (test WER 30.78 vs 30.69) and
crushed the equal-compute baseline (43.56 → 30.78). It also cut seed-to-seed
std from 0.53 to **0.37**. Scoring overhead was 15% self-scored, 4% transferred,
net −8% to −41% once pacing is included.

**This is nearly free for us: we already run ASR over every clip** and store the
result in the `cer` column. The same number that filters the corpus (§6.1) orders
it. Use duration only for padding efficiency.

**Short→long, deferred long-context: convergent practice, no controlled ablation.**
[MOSS-TTS](https://arxiv.org/pdf/2603.18090) (discrete-token AR, 12.5 fps) runs
four phases and states the principle plainly: *"long-context extension is deferred
until the short-context model has largely converged, which substantially reduces
optimization instability and preserves short-utterance quality"*, and *harder
conditioning tasks such as voice cloning are introduced while the LR is still
high "so that they become native behaviors rather than narrow later-stage
patches"*. [JoyVoice](https://arxiv.org/pdf/2512.19090) and
[MoonCast](https://arxiv.org/pdf/2503.14345) describe the same shape. **None
reports a non-curriculum control** — these are design principles, not results.

MOSS-TTS's one transferable *warning* is worth more than its schedule:
*"oversampling timbre-cloning data for too long biases the model toward prompt
copying and can suppress the relative influence of standard multilingual TTS."*
An over-cloned model copies its reference instead of reading its text, and that
shows up as good SECS and bad CER — the exact opposite failure from R1.

**Embeddings first, then the rest — and this is our Phase A.**
[LRSpeech](https://arxiv.org/pdf/2008.03687) (KDD'20): *"we first fine-tune the
character embeddings and speaker embeddings, and then fine-tune all parameters. It
can help prevent the TTS and ASR models from overfitting on the limited paired
data in a low-resource language."* Our Phase A (embedding-only warm-up) / Phase B
(LoRA joint) split was chosen on memory grounds (findings §3.4). It turns out to
be the published low-resource recipe, and it also mechanically limits the early
parameter drift Elhady measured.

**D20 — the curriculum, assembled.** Each stage names its source so a later
disagreement can be traced:

| stage | what | why |
|---|---|---|
| **0** (~2–5% of steps) | new text-embedding rows only, backbone frozen, English-heavy batches | LRSpeech; Elhady's English-first warm-up; our Phase A |
| **1** (LR held high) | short clean Indic + 20–25% English, **near-uniform across the five**, ASR-loss-ordered easy→hard with pacing | MOSS-TTS P1, Karakasidis, E8 |
| **2** (LR still high) | introduce reference-conditioned cloning pairs **oversampled**, plus code-mixed and ultra-short (word / number / date) items | MOSS-TTS P2; findings' Set C motivation |
| **3** (LR decaying) | **restore the mixture to natural proportions** — corrects the prompt-copying bias stage 2 induces | MOSS-TTS P3 |
| **4** (LR at floor) | extend sequence length, upsample multi-sentence / long-form | MOSS-TTS P4, JoyVoice |

Stage 4 is where our memory model pays off: peak memory is a pure function of
tokens/step (findings §3.4), so going seq 512 → 1024 is exactly a batch-8 → batch-4
trade with no re-probing.

**Train all six languages jointly, not one at a time.** §4.3's multilingual-beats-
bilingual result, E3's observation that Hindi *improved* as languages were added,
and E8's uniform-mix result all point the same way. A sequential curriculum spends
its first phase teaching a single-language prior that later phases must undo.

### 7.4 Speaker count beats hours per speaker — the cleanest result in this document

**E9** held total duration constant and varied only the number of speakers
(MMS-300M, evaluated on **out-of-distribution accents** — the ASR analogue of
unseen-speaker cloning):

| language | total | speakers | per speaker | CER ↓ |
|---|---|---|---|---|
| Mandarin | 150 min | 5 | 30 min | 60.3 ± 5.8 |
| Mandarin | 150 min | 10 | 15 min | 51.6 ± 3.2 |
| **Mandarin** | **150 min** | **15** | **10 min** | **39.9 ± 1.9** |
| English | 300 min | 5 | 60 min | 17.3 |
| **English** | **300 min** | **15** | **20 min** | **15.7** |

**15 speakers × 10 minutes beat 5 speakers × 30 minutes by 34% relative, at
identical total hours** — and tripling the speaker count cut the seed-to-seed
standard deviation from 5.8 to 1.9. In one case *less total data with more
speakers won outright*: 10 speakers × 10 min (100 min) beat 5 × 30 min (150 min).

**The single-speaker scaling experiment is the clincher.** Scaling *one* English
speaker from 5 h to 40 h: in-distribution CER falls monotonically (FLEURS 6.1 →
4.4), while **out-of-distribution CER stays pinned at ~20% across the entire 8×
increase (20.7 → 19.9)**. Their sentence: *"without sufficient speaker diversity,
scaling up training data helps ASR for seen accents but not for unseen accents."*

The two metrics have different knees. From the
[synthetic-data speaker sweep](https://arxiv.org/pdf/2512.17356v1) (1 → 1000
speakers at fixed hours): **WER and UTMOS plateau around ~50 speakers; SIM keeps
improving until ~500.** Intelligibility is cheap in speakers; cloning is not.

**D21 — cap hours per speaker and spend the freed budget on more speakers; report
per-language speaker counts as a first-class gate.** A ~1–2 h per-speaker cap is
directly supported. The gate: **if any of Tamil, Telugu or Marathi has fewer than
~50 distinct speakers after filtering, that language's cloning quality is
speaker-limited, and no amount of additional audio from the same voices will fix
it.** Given §10.1 — Rasa and IndicTTS are two speakers per language — this gate
will fail on the studio spine alone and is the concrete reason IndicVoices-R
(10,496 speakers) is not optional.

> **One dissenting source, and it dissents about a different regime.**
> [ZMM-TTS](https://arxiv.org/html/2406.08911v1) found the opposite — more
> utterances per speaker beat more speakers — but at **20–100 total utterances per
> language**, which is pure speaker adaptation, not language adaptation. It also
> found that fine-tuning improved *seen*-speaker similarity while leaving
> *unseen*-speaker similarity roughly at its zero-shot value, which is the same
> story as E2 told in §5.1. Regime matters: at ~100 utterances, depth; at our
> ~900 hours with zero-shot cloning as the goal, breadth.

---

## 8. Synthetic data, augmentation, and mined ASR corpora

### 8.1 TTS-for-TTS works better than expected, and does not collapse

I went looking for the ratio at which synthetic data poisons a TTS model and
**could not find one**. Three independent papers push synthetic well past 50% and
get *better* models. What breaks things is not the ratio — it is teacher
**style**, teacher **quality**, and speaker-identity **drift**, and the last of
those has a published fix.

**The directly applicable one.**
[ZeSTA](https://arxiv.org/abs/2603.04219) (Interspeech 2026) trained at **10%
real + 90% synthetic** (1:9), with synthetic from Fish-Speech and CosyVoice 2.
LibriTTS, SECS / CER / WER:

| | SECS | CER | WER |
|---|---|---|---|
| real 10% only | 0.818 | 5.932 | 12.520 |
| real 100% | 0.832 | 6.539 | 13.645 |
| **naive 10% real + 90% synth** | **0.765** | 4.738 | 10.348 |
| **+ domain embedding + 3× real oversampling** | **0.815** | 4.765 | 10.563 |

The naive mix **improves WER by 2.2 points absolute and costs 0.053 SECS**. A
**64-dimensional embedding marking each sample real-vs-synthetic**, plus
oversampling the real data 3×, recovers **0.050 of that 0.053** while keeping the
WER gain. (64 beat 16 and 256.) ABX preference for the fixed version over naive:
70.8%. Also: synthesising with **your own target speakers'** voices beat
mismatched speakers (SECS 0.807 vs 0.792).

**At scale.** [HAM-TTS](https://arxiv.org/html/2403.05989v1) trained on 150k h
real + **500k h synthetic (77% synthetic)**, the synthetic produced by voice
conversion to **1,000 randomly selected target speakers**:

| training data | CER | SMOS | MOS |
|---|---|---|---|
| 150k real | 4.0% | 4.12 | 4.27 |
| 150k real + 150k synth | 3.6% | 4.26 | 4.32 |
| **150k real + 500k synth** | **2.8%** | **4.32** | **4.49** |
| 500k synth only | 3.3% | 4.25 | 4.35 |

**500k hours of purely synthetic beat 150k hours of real.** No collapse at 3.3×.

**And at our kind of scale.** The [purely-synthetic
study](https://arxiv.org/html/2512.17356v1) trained a 20.9M Matcha-TTS on 100 h:
synthetic from XTTS-v2 beat real LibriSpeech train-100 on WER (**3.493 vs
4.428**) and SIM (0.404 vs 0.389), because the synthetic was *cleaner* (DNSMOS
3.99 vs 3.91).

**Three hard constraints come out of the same study, and they are the actionable
part:**

1. **Teacher style is decisive.** ChatTTS (oral/conversational register) as
   teacher drove the student to **WER 9.007 / SIM 0.276** — near-collapse — while
   XTTS-v2 gave 3.493 / 0.404. *Use a read/standard-register teacher, never a
   conversational one.* You inherit the teacher's error floor either way: a
   20.9M student on XTTS-v2 output beat XTTS-v2 itself on WER, but nothing
   rescued the ChatTTS arm.
2. **Noise tolerance is brutally tight.** Degradation sharpens **below SNR 30 dB**,
   and sweeping the *proportion* of noisy utterances, "impact can only be
   mitigated when the proportion of noisy speech is maintained at or below
   **0.1%**." Reverberation degrades monotonically with no safe fraction. Enhance
   before training, not after.
3. **Phoneme coverage beats transcript count.** At fixed duration, a 99% text
   reuse rate — only **285 distinct sentences** — was still fine *as long as all
   69 English phonemes were covered*. Below that (99.9% reuse → 60 phonemes) the
   quality fell off a cliff. **Audit aksara/phoneme coverage, do not count
   sentences.** This is the same finding as E4's syllable-balance result (§12
   rank 2) arriving from the synthetic side.

**No speech-domain model-collapse study exists.** Targeted searches returned only
text-LM work. The ChatTTS result above is a one-generation style mismatch, not
recursive collapse. Mark recursive-training risk **UNVERIFIED for speech** — and
note that nobody has published a synthetic:real *mixing-ratio curve* for TTS
either. ZeSTA tests one point, HAM-TTS tests three.

### 8.2 Speaker augmentation — what works and what is a clean null

**Does not work: VTLP / speed / formant pseudo-speakers.** The only controlled TTS
study ([Interspeech 2020](https://ar5iv.labs.arxiv.org/html/2005.01245)) tripled
VCTK's 100 speakers via SoX speed ×0.9/×1.1 treated as new identities. Every
delta in naturalness MOS and speaker-similarity DMOS fell between −0.3 and +0.2
and **none was significant** (Mann-Whitney U, p=0.01). Their conclusion: VTLP "did
not improve naturalness or speaker similarity." **Do not spend time on this.**

**Marginal: voice conversion purely to multiply speaker identities.** An
[exhaustive ASR augmentation
study](https://arxiv.org/html/2503.08954v1) swept VC speaker counts from 2,457 to
8,000 on 50 h real + 100 h synthetic: WER moved 46.30 → **46.03–46.09**, i.e.
~0.3 points, and adding speakers beyond 2,457 was not significant. **TTS-generated
augmentation over the same budget gave 40.21 — 6.1 points.** Their diagnosis is
the right one: VC "lack[s] augmentation of phonetic diversity" because it
preserves the source utterance's content. One useful sub-result: when choosing
which synthetic speakers to add, picking at **median** speaker-embedding distance
from the real set beat random, minimum and maximum distance (WER 41.65 vs 42.02 /
42.15 / 41.79).

**Works, and at scale:** HAM-TTS's 1,000 VC-generated target speakers (§8.1), and
— far more relevant to us — simply *acquiring* more real speakers. **E7's +9.25
S-SIM from IndicVoices-R's 10,496 speakers over IndicTTS's 27 is the largest
single speaker-diversity number in this document**, and it required no generation
at all.

**Untested:** kNN-VC has never been used for TTS or ASR data augmentation — an
exhaustive enumeration of the nine arXiv papers mentioning it found only voice
conversion, singing VC, and speaker anonymisation. If we try it, we are first.

**Existence proof for Indic, with a licensing catch:**
[IndicSynth](https://aclanthology.org/2025.acl-long.1070/) (ACL 2025 Outstanding
Paper) built **4,000+ h of synthetic speech across 12 Indian languages with 989
target speakers** using XTTS-v2 + VITS + FreeVC24. It is **CC BY-NC 4.0** and was
built for deepfake detection, so it is not usable for us — but it demonstrates
that FreeVC-based Indic speaker multiplication runs at this scale.

### 8.3 Mined ASR corpora — the gate is not optional, and the Indic inventory is larger than we thought

**The cost of skipping the gate, measured.**
[TITW](https://arxiv.org/html/2409.08711v2) built two corpora from the same
VoxCeleb1 source at the same ~180 h, differing only by DEMUCS denoising plus a
DNSMOS OVRL ≥ 3.0 cut. Training on the ungated version: **GradTTS UTMOS 2.18 →
1.29, WER 11.9% → 26.2%**, and **TransformerTTS and MQTTS failed to converge at
all.** At identical hours. This is why §6.1's pipeline is written down.

**The reference pipeline's yield.** [Emilia-Pipe](https://arxiv.org/html/2501.15907v2)
keeps **38.75%** of in-the-wild audio (666.94 h → 258.44 h) and lifts mean DNSMOS
from 2.52 ± 0.68 to **3.26 ± 0.14** — note the variance collapse, which matters
more than the mean. Stages: standardise (mono 24 kHz, −20 dBFS, gain clamped ±3
dB) → UVR-MDX-Net source separation → PyAnnote 3.1 diarization → Silero-VAD (keep
3–30 s) → Whisper ASR → filter on LID confidence ≥ 80%, DNSMOS ≥ 3.0, and
duration-per-character outside 1.5×IQR.

**Indic mined corpora, with licences verified on the primary page:**

| corpus | hours | speakers | licence | gated |
|---|---|---|---|---|
| **SYSPIN** | ~900 total, **~50–57 h per speaker** | **18** (1M+1F × 9 langs) | **CC BY 4.0** | no |
| **Vaani** | 31,278 audio / **2,122 transcribed** | **~156,000**, 165 districts | **CC BY 4.0** | yes |
| IndicVoices | 23.7K / 11.2K transcribed | 51K, 400+ districts | CC BY 4.0 | yes |
| Shrutilipi | 6,400+ | — | CC BY 4.0 | yes |
| YODAS (Indic portion) | ~632 (Hindi ~483) | — | **CC BY 3.0** | **no** |
| Emilia | 101,654 | — | **CC-BY-NC — unusable** | yes |
| WenetSpeech4TTS | 12,800 | — | **card says non-commercial** | no |

**Two of these change the plan.**

**SYSPIN is the corpus we were not counting.** ~50–57 h of *studio* audio per
speaker, 1 male + 1 female for each of 9 languages, **CC BY 4.0, ungated**, and it
covers four of our five: **Hindi ~110 h, Bengali ~105 h, Marathi ~100 h, Telugu
~106 h**. **No Tamil.** That is ~420 h of clean commercially-usable studio data
from 8 distinct professional speakers — a material addition to §10.1's accounting,
and another reason Tamil is the outlier.

**Vaani is the speaker-diversity source nobody mentions.** CC BY 4.0, ~156,000
speakers across 165 districts, image-prompted spontaneous speech. Transcribed
hours for our five: **Hindi 1,003.78 · Telugu 184.48 · Bengali 162.43 · Marathi
68.09 · Tamil 24.35.** Given §7.4 — that cloning quality is bound by speaker count
and our studio spine has ~10–20 voices — a corpus with six-figure speaker counts
is worth more per hour than anything else on the list, even at spontaneous
register and after a 60% filter loss.

> **An operational blocker worth knowing before the preprocessing job is written.**
> `mozilla-foundation/common_voice_17_0` on HuggingFace now reads *"The dataset is
> currently empty"* — **effective October 2025 Common Voice is distributed
> exclusively through the Mozilla Data Collective.** Any pipeline step that assumes
> an HF-hosted Common Voice will fail, and the licence terms at the new
> distribution point need re-reading.

**Licences still unresolved and load-bearing** (findings §7 makes licences a hard
filter): the **IndicTTS** terms PDF and the **LIMMITS** terms PDF both failed to
load. Indic Parler-TTS's model card characterises both as CC BY 4.0, but that is
AI4Bharat's characterisation, not the licensor's. **Resolve both before shipping.**
Separately, [arXiv 2410.14197](https://arxiv.org/html/2410.14197) records that the
LIMMITS organisers themselves noted "many errors including distortion in the
audio, wrong pronunciation of words, mismatch between text and audio" — so LIMMITS
needs the §6.1 filter applied to it, not trusted as studio-clean.

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

## 9. Distillation — mature for speed, second-tier for data

The word covers two unrelated things and conflating them is the main way this
topic wastes time.

### 9.1 Inference-speed distillation: excellent, and saves zero data

| system | teacher → student | WER/CER | SIM | speed |
|---|---|---|---|---|
| [DMOSpeech](https://arxiv.org/html/2410.11097v2) | 128 NFE → 4 NFE | **9.51 → 1.94** | 0.55 → **0.69** | RTF 0.96 → 0.07 (13.7×) |
| [DMOSpeech 2](https://ojs.aaai.org/index.php/AAAI/article/download/40450/44411) | F5-TTS 32 → 4 steps | en 1.947 → **1.752** | 0.662 → **0.698** | RTF 0.167 → 0.032 (5.3×) |
| [FlashSpeech](https://arxiv.org/pdf/2404.14700) | **no teacher** (adversarial consistency *training*) | WER 2.7 | 0.52 | RTF 0.02 |

The student beating the teacher is real but is **not** the distillation doing it:
DMOSpeech's ablation shows DMD2 alone at 4 steps gives WER 5.67 / SIM 0.53, and it
is the auxiliary **CTC loss** (→ 1.79 / 0.55) and **speaker-verification loss**
(→ 6.62 / **0.70**) that produce the gains. Teacher and student trained on the
same 57,706 h. **No data was saved.**

Two things worth carrying even though we are not doing this yet:

- **"Mode shrinkage."** DMOSpeech 2 identifies that a 4-step student has a visibly
  narrower F0 distribution than its teacher — **prosodic diversity collapses
  before spectral quality does**. Their fix is teacher-guided sampling: teacher for
  the early high-noise steps that set prosody, student for the late steps that set
  timbre. If we ever distil `s3gen`, expect flattening before we expect noise.
- **FlashSpeech is the counter-example to needing a teacher at all** — adversarial
  consistency training reaches 1–2 steps from scratch.

### 9.2 Data-efficiency distillation: real, but it is not the top lever

**The strongest claim is [SPADE](https://arxiv.org/abs/2509.20802)** (ICASSP 2026):
structured pruning of an LLM-TTS by a WER-based layer-importance index — *halving
Transformer depth* — plus multi-level distillation "to restore autoregressive
coherence", reaching near-parity perceptual quality on zero-shot benchmarks using
**"less than 5% of the original training data"**, with VRAM down 20% and RTF up
1.7×. That is a compelling shape for us. **Caveat: the abstract names neither the
base model nor the exact data amounts, and I could not retrieve the tables.** Mark
the 5% figure **UNVERIFIED**.

**The Indic datapoint ranks it fourth of four.**
[ELAICHI](https://arxiv.org/html/2410.17901v1) (AI4Bharat, Hindi) used a VoiceCraft
880M teacher trained on IndicVoices-R to generate **27.67 h of synthetic Hindi
targeted at low-frequency character bigrams** (939 new bigrams, 7.92× relative
growth). Intelligibility rate on low-frequency-bigram utterances, and MOS:

| condition | IR | MOS |
|---|---|---|
| baseline: 20.16 h IndicTTS Hindi, 2 speakers | 0.51 | 3.97 |
| + proximal language (Chhattisgarhi, 80.11 h) | 0.74 | 4.16 |
| + **synthetic KD** (27.67 h, 1 speaker) | 0.71 | 4.08 |
| + denoised ASR data (72.74 h IV-R, 368 spk) | 0.66 | **3.85 (↓)** |
| + multilingual pooling, 14 langs (224.57 h) | 0.85 | 4.42 |
| **+ multilingual pooling, 15 langs (852.17 h)** | **0.90** | **4.51** |

**Multilingual pooling (0.90) > related-language borrowing (0.74) > synthetic KD
(0.71) > denoised ASR data (0.66).** Distillation helped — but it was the weakest
of the three big levers, and the thing that beat everything was simply *more
languages pooled together*, which is §4.3 and D20 again.

Note also the denoised-ASR row: **IR rose 0.51 → 0.66 while MOS fell 3.97 → 3.85.**
That is E7's +9.25 S-SIM / −0.19 NORESQA trade-off appearing a third time. It is
real and it is consistent.

**The extreme reference point.** [LRSpeech](https://arxiv.org/pdf/2008.03687)
(KDD'20) got a usable TTS from **50 utterances (5 minutes) of high-quality paired
data + 3.5 h of low-quality paired + 21 h unpaired**, pretrained on Mandarin.
Stage-by-stage: pretrain/finetune → IR 93.09 / MOS 2.84; + dual transformation →
96.70 / 3.28; **+ knowledge distillation → 98.08 / 3.57.** KD alone bought +1.38
IR and **+0.29 MOS** at the very bottom of the data range.

**And the honest Indic self-training number.** E1's zero-resource procedure — 1 h
of self-generated speech per language, human-verified, then self-trained — moved
Bhojpuri MUSHRA 82.0 → **83.9** and Tulu 93.62 → **93.12**. Both inside the ±6
confidence interval, with one expert rater per language. **"Does not hurt" is the
defensible reading; "+1.9 MUSHRA" is not.**

### 9.3 Not distillation, but in the same budget line: reward optimisation

CosyVoice 3's **DiffRO** (Gumbel-Softmax over LLM speech tokens, reward from an
ASR-like Token2Text posterior, backpropagated rather than an RL loop, plus a KL to
a reference) is the largest single intervention in E6's whole paper:

| | zh | en | ja | **ko** | de | it |
|---|---|---|---|---|---|---|
| CosyVoice3-0.5B | 3.89 | 5.24 | 10.4 | **12.8** | 7.41 | 6.68 |
| **+ DiffRO** | 2.89 | **3.68** | 5.15 | **4.02 (−68.7%)** | 4.51 | 2.94 |

The gains are largest on the *weakest* languages — Korean fell 68.7%. **The cost is
a small, consistent drop in speaker similarity** (en SS 0.718 → 0.695).

**Inference:** for a project whose problem is a language the base model carries
badly (Tamil, perplexity 287.2), an ASR-posterior reward looks like a better
use of the same engineering week than a distillation student. It needs no new
data, it targets exactly the metric we are behind on, and its cost is on the axis
§3.3 says we have margin on. Filed as a lead, not a decision — we have not
measured it and it is out of scope for the first run.

### 9.4 Verdict

**Distillation is not in our top three.** Inference-speed distillation saves no
data; data-efficiency distillation is real but ranked fourth of four by the one
Indic study that compared it against the alternatives. **If we use a teacher at
all, use it the way ELAICHI did — to generate 20–50 h targeted at coverage holes
(rare aksharas and conjuncts, code-mixed English-in-Indic spans, numbers and
dates), not as a bulk data multiplier.** ELAICHI's bigram targeting, E4's
syllable balancing, E5's volunteer OOV recordings and the synthetic study's
phoneme-coverage finding are four independent routes to the same conclusion:
**coverage of the rare units is what low-data TTS is short of, and all four of the
cheapest wins in this document attack it.**

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
> (10,496 speakers), Vaani and the OpenSLR crowdsourced sets. Given §6.3 and §7.4, that
> makes IV-R not a nice-to-have but **the only source of the property zero-shot
> cloning is trained on**. Both figures are also gated and need accepted HF terms
> — findings §6 already flags this as the long pole.

**Two corpora the original catalogue under-weighted** (§8.3):

| | SYSPIN (studio, CC BY 4.0, **ungated**) | Vaani transcribed (CC BY 4.0, gated) |
|---|---|---|
| Hindi | ~110 h (2 speakers) | **1,003.78 h** |
| Bengali | ~105 h (2) | 162.43 h |
| Marathi | ~100 h (2) | 68.09 h |
| Telugu | ~106 h (2) | 184.48 h |
| **Tamil** | **none** | **24.35 h** |

SYSPIN adds ~420 h of *studio* audio from 8 professional speakers across four of
our five, with no gate to clear. Vaani adds spontaneous speech from **~156,000
speakers across 165 districts** — by far the deepest speaker inventory available
in these languages, and therefore (per §7.4) the highest-value hours per hour for
the cloning objective, even after a 60%+ filter loss.

Speaker diversity top-ups, cheap and multi-speaker:
[OpenSLR crowdsourced sets](https://www.openslr.org/resources.php) — SLR 64
Marathi, 65 Tamil, 66 Telugu, 37 Bengali. **There is no Hindi crowdsourced set.**
Also [YODAS](https://huggingface.co/datasets/espnet/yodas), **CC BY 3.0 and
ungated** — the only unencumbered mined Indic audio found — but thin for us:
Hindi ~483 h, Tamil 59, Bengali 27, Marathi 7, Telugu 6.

**Notice what every one of these tables says about Tamil.** No SYSPIN. 24 h of
Vaani against Hindi's 1,004. The smallest studio allocation (52.25 h). The worst
Qwen3 fertility (10.13 tokens/word). The worst base-model perplexity (287.2 vs
English 60.9). **Tamil is short of data on every axis simultaneously**, and that
is the single most important fact in this section.

Deliberately **not** in the recommended mix:

- **Shrutilipi** ([CC BY 4.0, 6,400+ h, gated](https://huggingface.co/datasets/ai4bharat/Shrutilipi)).
  All India Radio news bulletins, **document-level** mined alignment. Three
  strikes for TTS: document-level alignment is exactly the misalignment risk §6.1
  warns about; news-read register is acoustically and prosodically narrow (§6.3's
  energy-std failure mode); and the speaker inventory is small and professional.
  It is an ASR corpus. Revisit only if the hours-vs-quality curve (§13) shows we
  are data-bound after IV-R and Vaani, and then only with forced realignment.
- **Emilia and WenetSpeech4TTS.** Both would be useful English/Chinese replay
  material and both are **non-commercial** (Emilia CC-BY-NC-4.0; the
  WenetSpeech4TTS HF card says non-commercial). Only the **Emilia-YODAS** portion
  is CC BY 4.0. Findings §7 makes licences a hard filter, so English replay must
  be sourced from something else — GLOBE (CC V1, 535 h, 164 accents) is what
  Indic Parler-TTS and RASMALAI both used.
- **IndicSynth** (4,000 h synthetic, 12 Indian languages, 989 speakers) —
  **CC BY-NC 4.0**, so unusable despite being exactly the resource §8.2 describes.
- **IndicF5 / Indic Parler-TTS outputs** as synthetic data in phase 1. Licensing
  is clean (§8.4) but it is a bootstrap we have not shown we need.

### 10.2 The recommended strategy

**Target: 120–180 sampled hours and ≥ 200 distinct speakers per language, drawn
from ~1,400 h available, in one joint multilingual run with 20–25% English
replay, under a two-stage quality curriculum.**

| | what | why |
|---|---|---|
| **Corpora, tier 1** (studio spine) | IndicTTS + LIMMITS + Rasa + **SYSPIN** | ~940 h of clean material; defines pronunciation |
| **Corpora, tier 2** (speaker breadth — *not optional*) | **IndicVoices-R** + **Vaani** + OpenSLR crowdsourced | §6.3, §7.4: studio alone costs 9.25 points of unseen-speaker similarity, and our studio spine holds ~10–20 voices |
| **English replay** | GLOBE (CC V1, 535 h, 164 accents) | Emilia and WenetSpeech4TTS are non-commercial (§10.1) |
| **Filtering** | **Combined rank** (DNSMOS + ASR disagreement + VAD speech ratio), cut bottom 15–20% → **75–85% retention**; plus hard gates: duration 3–30 s, LID confidence ≥ 0.80 with audio/transcript agreement, roll-off > 7 kHz, speaker-change detection, length-ratio trim (bottom 1% / top 5%), volume to −20 dBFS | D16, D16a; §6.2's three retention curves |
| **Enhancement** | HTDemucs → VoiceFixer → DeepFilterNet3 on in-the-wild sources only, **before** training | E7; §8.1's "reverberation degrades monotonically" |
| **Text selection** | Greedy **aksara-balanced** selection when subsampling; audit grapheme/phoneme coverage, never sentence count | §12 rank 2; E4 CER 0.812 → 0.074 at 1 h; §8.1's 285-sentences-is-enough result |
| **Mixing** | **near-uniform across the five Indic languages** (`language_temperature` 0.0–0.2) with a **hard epoch cap**; English a separate fixed **20–25%** | D18, D19 |
| **Speakers** | cap hours per speaker at ~1–2 h; spend the freed budget on more speakers; **gate on ≥ 50 distinct speakers per language, target ≥ 200** | D21; §7.4 |
| **Order** | Stage 0 embeddings-only → Stage 1 short + balanced → Stage 2 cloning oversampled → Stage 3 mixture restored, LR decaying → Stage 4 long-form; then **D17a's clean-subset polish at reduced LR with a step cap** | D20, D17a |
| **Never** | KittenTTS 2 output in the mix; Shrutilipi without realignment; romanized input; Emilia / WenetSpeech4TTS / IndicSynth (non-commercial); conversational-register synthetic teachers; VTLP pseudo-speakers | §8.4, §10.1, D14, §8.1, §8.2 |

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
| **distinct speakers/language** | **≥ 50** | §7.4 — WER and UTMOS plateau at ~50 speakers; **this is the constraint E3 violated with 2** |
| text selection | greedy aksara-balanced, full grapheme coverage | E4 (11× CER difference at 1 h); §8.1 (coverage, not sentence count) |
| English replay | 10–15 h, from a commercially-licensed source | D19, §10.1 |
| alignment verification | ASR-agreement on 100% of rows | §6.1 |
| enhancement gate | DNSMOS on in-the-wild sources | §8.3 — TITW: skipping it took WER 11.9% → 26.2% at identical hours |

**The corroborating sub-1000-hour evidence is now substantial**, from outside
Indic and outside our architecture, which is why it is worth listing:

| system | data | result |
|---|---|---|
| [ZipVoice](https://arxiv.org/html/2506.13053v3) 123M | **555 h LibriTTS** | LibriSpeech-PC **WER 1.69 / SIM-o 0.610** (GT 1.87 / 0.690) |
| [MELLE-limited](https://arxiv.org/html/2407.08551v1) | **960 h LibriSpeech** | WER-H 2.80 / SIM-o 0.591 (vs 2.10 / 0.625 at 50k h) |
| [CosyVoice Exp-3](https://arxiv.org/html/2407.05407v2) 300M | **585 h LibriTTS** | WER 3.93 / SS 67.85 (GT 3.01 / 69.67) |
| [F5-TTS small](https://arxiv.org/html/2410.06885v3) 155M | **945 h WenetSpeech4TTS Premium** | Seed test-zh WER 4.17 / SIM 0.54 |
| [NaturalSpeech 3](https://arxiv.org/html/2403.03100v3) 500M | **1,000 h** | SIM-o 0.64 / WER 3.94 (vs 0.73 / 2.11 at 200k h) |

Note what these share: **ZipVoice's 555 h is LibriTTS with ~2,456 speakers**, and
it reaches SIM-o 0.610 — within 0.058 of its own 100k-hour version. The sub-1000-h
systems that do well on similarity are the ones that are speaker-rich. That is the
whole thesis of §7.4 restated as a table.

**The honest version of the recommendation:** 10 h/language is the *floor for a
demo*. **60 h/language (300 h Indic) is the floor for a product**, because that is
roughly where E1's curve has flattened and where IV-R and Vaani can supply enough
distinct speakers for cloning to work. We have ~1,400 hours available; the reason
to go to 120–180 sampled hours per language is not that the curve demands it, but
that it costs nothing — compute is ~25–40 A100-hours (findings §6) and the data is
already licensed.

> **Tamil will not reach the recommendation and may not reach the minimum.**
> After filtering at 80% retention, Tamil's realistic pool is roughly 42 h studio
> (IndicTTS + Rasa + LIMMITS, no SYSPIN) + ~80 h IV-R + ~20 h Vaani + OpenSLR SLR
> 65 ≈ **140 h**, against Hindi's ~1,000 h of Vaani alone. The hours are adequate;
> **the speaker count is the open question**, and §13.1 says measure it before
> training rather than discovering it afterwards.

---

## 12. Techniques ranked by payoff per unit of effort

Ranked by *measured* effect divided by *engineering days*, for our specific
situation.

| rank | technique | measured payoff | effort | evidence |
|---|---|---|---|---|
| **1** | **Many-speaker data (IndicVoices-R + Vaani), not just studio** | unseen-speaker similarity **78.93 → 88.18** (+9.25); 15 spk × 10 min beats 5 spk × 30 min by **34% rel. CER** at equal hours | days — downloads, gate acceptance, a filter pass | E7, E9 |
| **2** | **Greedy aksara-balanced text selection when subsampling** | CER **0.812 → 0.074** at 1 h Tamil; syllable coverage 49% → 73% | hours — Indic NLP Library + a greedy loop | E4 |
| **3** | **Enhancement + a DNSMOS floor on in-the-wild sources** | skipping it: **WER 11.9% → 26.2%, UTMOS 2.18 → 1.29** at identical hours; two models failed to converge | days at preprocessing | TITW |
| **4** | **PEFT instead of full fine-tune** | unseen-speaker CER **13.45 → 7.22**; degradation 2.37× → 1.47×; base-language WER **unchanged** vs 11.68 → 32.29 for full FT | **zero** — already our plan, for the wrong reason | E2, LoRA-Whisper |
| **5** | **English replay at 20–25%** | forgetting +20.6 → **+1.7** WER points (LoRA-Whisper); retention **20.12 → 43.43** (Elhady); E8 saw English *improve* | zero — a mixing weight | D19 |
| **6** | **Near-uniform language balancing with an epoch cap** | Tamil WER **139.96 → 39.19** on a Qwen3-1.7B at 968 h, with no cost to English or Mandarin | a sampling change + a cap we must write | E8, D18 |
| **7** | **Combined-rank filtering to 75–85% retention** | WER 2.19 → **2.00**, SIM +0.011; and 50% pruning is worth exactly nothing | days at preprocessing | Raon, §6.2 |
| **8** | **Two-stage quality curriculum (most → cleanest)** | unseen-speaker CER **14.96 → 9.82** and **14.06 → 7.89** | a second short run | D17a |
| **9** | **Brahmic phonetic warm-start of new embedding rows** | not isolated in any ablation — **UNVERIFIED**; but it is what the one working Indic system on our architecture did | an afternoon; `verify_surgery.py` already tests it | E3, D13 |
| **10** | **Order training by ASR loss, not by utterance length** | matched a full-data baseline at **44% fewer hours seen**; seed std 0.53 → 0.37 | low — we already compute the score | Karakasidis |
| **11** | **Cap hours per speaker (~1–2 h), spend the budget on speakers** | single-speaker scaling 5 h → 40 h left out-of-distribution CER **pinned at ~20%** | a sampling-policy change | E9 |
| **12** | **Targeted volunteer recordings for OOV coverage** | OOV intelligibility error **−22% to −41% rel.** from 48–64 minutes; 85% cheaper than studio; 2 working days | a recording session + ethics | E5 |
| **13** | **Teacher-generated audio targeted at coverage holes** | ELAICHI IR 0.51 → 0.71 from 27.67 h; ranked **4th of 4** levers in its own paper | a teacher + verification | §9.2 |
| **14** | **Acoustic-diversity screening (energy std)** | predicted fine-tune failure where hours did not; loss curve blind to it | two lines at preprocessing | 2603.10904 |
| — | ~~VTLP / speed / formant pseudo-speakers~~ | **clean null**: all deltas −0.3 to +0.2, none significant | — | [Interspeech 2020](https://ar5iv.labs.arxiv.org/html/2005.01245) |
| — | ~~Voice conversion purely to multiply speakers~~ | ~0.3 WER vs TTS augmentation's 6.1; "lacks augmentation of phonetic diversity" | — | [2503.08954](https://arxiv.org/html/2503.08954v1) |

**Items 1–3 are the three to do first, and together they cost about a week.**
Item 4 costs nothing because it is already the plan — but §5.1 and LoRA-Whisper
together mean we should stop describing it as a memory compromise and start
describing it as the quality choice it is.

**Item 12 deserves a note.** E5 is the most *under-rated* technique in this list
and the only one that creates data rather than selecting it. AI4Bharat recorded
1,983 Hindi words (48:07) and 2,098 Tamil words (64:32) from **volunteers, not
voice artists**, chosen by a greedy algorithm maximising missing character-bigram
coverage with a per-bigram cap of k=6 — and cut OOV intelligibility error by
40.59% (Hindi FastPitch) and 36.33% (Tamil FastPitch). Critically, **the
volunteers' voices improved the OOV performance of the model's original
speaker**: phonetic coverage transfers across speakers. One hour of deliberately
chosen audio beat a great deal of undirected audio. It is ranked 12th only because
it has a physical-world dependency; on payoff per hour of audio recorded it is
first, and Tamil is where it would pay.

**The pattern across items 2, 12 and 13.** Three of the cheapest wins are the same
technique wearing different clothes: *cover the rare units*. E4 selects text for
syllable coverage; E5 records words for bigram coverage; ELAICHI synthesises
audio for bigram coverage; and the synthetic-data study found 285 distinct
sentences sufficed **provided every phoneme appeared**. Low-data TTS is not short
of hours. It is short of coverage of the rare units, and coverage is purchasable
far more cheaply than hours.

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
seed. Report against the same Set A / Set C harness. Three rules make the curve
mean something:

- **Fix total optimisation steps across the arms, not epochs.** E1 found this
  dominates in the data-poor regime: *"a model fine-tuned on 10 hours for 150K
  steps achieving a WER of 31.3% compared to fine-tuning on 100 hours for 120K
  steps achieving a WER of 40.0%."* **Ten times less data and a 25% larger step
  budget won by 8.7 WER points.** A curve run at fixed epochs would measure the
  step budget and call it data scale.
- **Run two subsampling policies, not one.** *By utterance* keeps every speaker
  and isolates the effect of hours; *by speaker* removes voices and isolates the
  effect of breadth. §7.4 says these are different curves with different knees,
  and only running both tells us which constraint we are actually under. Report
  **hours and distinct-speaker count** on every point.
- **Audit speaker counts before spending a single GPU-hour.** The gate in D21 —
  ≥ 50 distinct speakers per language after filtering — is answerable from the
  shard manifest alone. If Tamil comes out at 12 speakers, the curve will say
  "Tamil is data-bound" when the truth is "Tamil is speaker-bound", and the
  remedy is a different corpus, not more hours.

> **One additional arm, run once, that nothing in the literature covers.** E1
> tested 0% and 50% English replay; nobody has published the 10/20/25/30% region
> (§7.2). Four short runs at a fixed step budget, scoring **English Set A CER and
> SECS *and* Indic CER at every checkpoint**, would settle the single
> highest-leverage unmeasured parameter in our plan — and would be a publishable
> result, since it does not exist for TTS anywhere.

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
| Indic Parler-TTS, 1,806 h | CER 12%, WER 24%, NORESQA-MOS 4.14 | [RASMALAI](https://arxiv.org/html/2505.18609v2) |
| CosyVoice 3-0.5B, 1M h | test-en WER 2.46, SS 0.747 | [E6](https://arxiv.org/html/2505.17589v2) |

**And the market position, which is the number that decides whether this ships.**
[Preferences of a Voice-First Nation](https://arxiv.org/html/2604.21481) (AI4Bharat,
Apr 2026) ran 120,000+ pairwise comparisons by 1,900+ native raters over 7 systems
× 10 Indic languages. Bradley-Terry scores:

| system | BT | win rate |
|---|---|---|
| Gemini 2.5 Pro TTS | 1128.53 | 70% |
| ElevenLabs v3 | 1056.28 | 57% |
| Sonic 3 | 1050.83 | 56% |
| Sarvam Bulbul v3 Beta | 1021.91 | 52% |
| GPT-4o-mini TTS | 942.76 | 40% |
| **IndicF5 (best open)** | **805.75** | **19%** |

**The best open Indic system sits ~320 BT points below the commercial frontier**,
which is the gap this project exists to close. Their SHAP analysis says
**expressiveness and intelligibility dominate preference; hallucinations and noise
contribute least** — so R4 (prosody flattening) matters more to a user than R2's
occasional glitch, which is the opposite of how an engineering team naturally
prioritises.

> **Measure the codec's ceiling before training anything. It costs zero GPU-hours
> and it is the number that makes every SECS result interpretable.**
> For a discrete-codec system the achievable speaker similarity is bounded not by
> the ground truth but by **codec resynthesis**: published gaps are SIM-o
> **0.710–0.715 against a GT of 0.779** on LibriSpeech cross-sentence
> ([MELLE](https://arxiv.org/html/2407.08551v1),
> [VALL-E 2](https://arxiv.org/html/2406.05370v1)) and **0.700–0.720 against human
> 0.734–0.755** on Seed-TTS-eval ([CosyVoice 2](https://arxiv.org/html/2412.10117v3)).
> We already have 160 S3 round-trip pairs on disk under `.cache/codec_roundtrip/`
> from the Gate 1 run (findings §3.6) — scoring them with the WavLM embedder gives
> us our own ceiling in an afternoon. **If our codec resynthesis SIM-o is 0.70, a
> generation SIM-o of 0.62–0.65 is a good result and 0.72 is impossible.**
> Published systems land 0.05–0.09 below their codec ceiling.

One more comparability warning, because it would otherwise silently corrupt every
cross-paper comparison we make: **SECS depends on the speaker-verification model
by up to 0.20 absolute.** CosyVoice 2 reports 0.748 (WavLM-ft) and 0.806
(ERes2Net) for the *same system on the same test set*; IndexTTS2 reports 0.865 on
that set using FunASR's SV model. D8 already fixed our choice to
`microsoft/wavlm-base-plus-sv`; the point here is that **IN-F5's "97.3 S-SIM" and
IndicVoices-R's "88.18" are not on our scale and never will be** — IV-R uses a
Wav2vec2 model fine-tuned via IndicSUPERB. Compare deltas within a study, never
levels across studies.

### 13.3 Five measurements the current harness cannot make

1. **An unseen-speaker split.** Every cloning number in §5.1, §6.3 and §6.4 is a
   *seen vs unseen* comparison, and that gap is the whole finding. Set A must be
   partitioned by speaker, with the reference pool drawn from speakers absent from
   training. Without this we cannot detect the failure mode that E2 shows is the
   dominant one.
2. **A hard-input set, per language, built on day one.** §3.3 says test-hard is
   where hours live (CER 27.60 → 6.96 across 57× data). This is the AR-specific
   repetition/looping failure mode and it is **invisible on ordinary test sets** —
   CosyVoice 3's easy CER barely moved over the same sweep. Tongue-twisters,
   repeated words, very long noun phrases.
3. **A long-form set scored as worst-of-N over seeds.**
   [Taming Long-form TTS](https://arxiv.org/abs/2609.16989) (Sep 2026) measures
   the right thing: **Qwen3-TTS-0.6B scores worst-of-10-seeds WER 35.2% on
   >1500-word prompts against 5.4% on <500-word prompts (6.5×)**, and catastrophic
   generations (WER > 30%) occur on **26%** of long prompts. Mean WER hides all of
   this. They also introduce **sliding-window SIM**, because plain SIM hides
   mid-generation speaker drift: worst-of-N wSIM on a 120 s reference was **0.01**
   where mean SIM looked fine. **Adopt worst-of-N-over-seeds and sliding-window
   SIM, or the long-form numbers will lie.** Add duration ratio against the
   reference reading as the cheap detector — a generation 1.6× or 0.5× the expected
   length is broken whatever its CER says.
4. **A SECS metric that is not saturated, reported against the codec ceiling.**
   E1's SSIM sat at 92.7–94.0 across a sweep that included a model with WER 59.4.
   Calibrate first: measure same-speaker (ceiling), different-speaker (floor), and
   **codec-resynthesis** similarity on held-out recordings — `metrics.py` already
   contemplates the first two — and report where the model sits *between* them.
5. **A Text Risk Score over both the training text and the eval set.**
   [Complex-Text Robustness for Low-Resource Multilingual
   TTS](https://arxiv.org/abs/2609.11545) (NCMMSC 2026) measured CER by text
   category across three systems and four low-resource languages (Thai,
   Vietnamese, Swahili, Indonesian — **not Indic**, so treat as indicative):

   | category | CER | duration-abnormal rate |
   |---|---|---|
   | **Long sentences** | **2.06%** | 0.00% |
   | Named entities | 6.78% | 1.67% |
   | *Ordinary* | *8.60%* | *7.92%* |
   | Punctuation | 9.65% | 5.42% |
   | **Mixed-script / code-switched** | **17.77%** | 1.67% |
   | **Numbers / dates** | **23.40%** | **17.92%** |

   **Numbers and dates are 2.7× ordinary CER; code-mixing is 2.1×; and long
   sentences are the *easiest* category of all.** Length per se is not the
   problem — high-risk *tokens* are. (These figures already include unified text
   normalisation and format-robust reference matching, so they are synthesis
   failures, not ASR inverse-normalisation artifacts.) Their 4-feature rule-based
   score — text length, digit ratio, punctuation ratio, mixed-script ratio,
   min-max normalised within language — separates high-risk CER 16.89% from
   low-risk 8.80% with **no training required**.

   This is a direct vindication of Set C's design (D11) and of the Indic
   number-to-words gap in findings §6: **the untranslated number expansion is not
   a cosmetic gap, it is the single highest-CER category in TTS.** It also gives
   us a free sampling knob — score the training text and use it for the
   difficulty-ordered curriculum in D20.

---

## 14. Risks — what degrades with little data, and how it shows up

Ordered by how likely they are to bite us, with the metric each one hides behind.

### R1 — Speaker generalisation collapses first, and silently

**Evidence.** E2: full fine-tune on one speaker, seen-speaker CER 5.671 → unseen
13.454 (2.37×), SMOS 2.35 against a ground-truth 4.88. E7: studio-only training
loses 9.25 points of unseen-speaker similarity. 2603.10904: breadth beat depth by
+0.293 DNS-MOS zero-shot. **And the clincher, E9:** scaling a *single* speaker
from 5 h to 40 h drove in-distribution CER down monotonically (6.1 → 4.4) while
out-of-distribution CER stayed **pinned at ~20% across the entire 8×
increase** — *"without sufficient speaker diversity, scaling up training data
helps for seen accents but not for unseen accents."*

**How it shows up.** It does *not* show up in training loss, in seen-speaker CER,
or in a saturated SSIM. It shows up only in a seen-vs-unseen gap, which §13.3
says we currently cannot measure.

**Mitigation.** PEFT over full fine-tune (D15); IV-R in the mix (§10.2); maximise
speaker count (§7.4); build the unseen-speaker split before the first real run.

**Our specific exposure.** Findings §3.7 could not confirm the reference layout —
`role_swapped` cost −0.016 nats and the whole reference prefix was worth 0.062.
That gate was blind to conditioning by construction (teacher forcing predicts
audio tokens from their predecessors). **R1 and the unresolved reference layout
are the same measurement.** Generating from both layouts and scoring unseen-speaker
similarity settles both at once, and nothing else settles either.

### R2 — Hard-input and long-form stability degrade while easy sentences look fine

**This is our architecture's characteristic failure and the most data-hungry
property we have.**

**Evidence.** E6: 3,000 h → 170,000 h moved easy Chinese CER only 1.68 → 1.27 but
**test-hard 27.60 → 6.96**. [Taming Long-form
TTS](https://arxiv.org/abs/2609.16989): Qwen3-TTS-0.6B worst-of-10-seeds WER
**35.2% on >1500-word prompts vs 5.4% on <500-word**, with **26% of long prompts
producing a catastrophic generation (WER > 30%)**. [Complex-text
robustness](https://arxiv.org/abs/2609.11545): numbers/dates **23.40% CER** vs
ordinary 8.60%, with a **17.92% duration-abnormal rate** — the highest of any
category.

**How it shows up.** A headline Set A CER that looks acceptable, and a product
that falls apart on a paragraph or on a phone number. For an AR model over 25 Hz
tokens the failure is runaway or truncated generation, which a CER computed on a
short clip can even *flatter* — a truncated utterance has fewer tokens to get
wrong. **Mean-over-seeds hides it almost completely**; the published 35.2% is a
worst-of-10 figure.

**Mitigation.** The hard set and the long-form set in §13.3, scored **worst-of-N
over seeds**, with sliding-window SIM and duration ratio. Duration ratio is the
cheap detector: a generation 1.6× or 0.5× the expected length is broken regardless
of its CER. Note the published fixes are all inference-side or post-training —
VALL-E 2's repetition-aware sampling, CosyVoice 3's RL stage, LACI's detect-and-
regenerate (which took 35.2% → 3.4%). **We will not fix this with data at 900
hours; we will detect it and work around it.**

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

**And it is the risk that matters most commercially.** The
[Voice-First Nation](https://arxiv.org/html/2604.21481) study ran SHAP over six
perceptual axes against 120,000+ native-rater preferences and found
**expressiveness and intelligibility dominate preference, while hallucinations and
noise contribute least.** That is the opposite of how an engineering team
naturally prioritises: we will instrument the glitches, because glitches are
measurable, and under-instrument the flatness that actually decides whether anyone
prefers our output.

**How it shows up.** CER stays fine. The model becomes a competent newsreader with
one register. **No objective metric in our harness detects this**, and §9.1's
"mode shrinkage" says a distilled or few-step decoder would make it worse —
prosodic diversity collapses before spectral quality does.

**Mitigation.** Rasa is already in the recommended mix and carries six Ekman
emotions with per-emotion tags, and our shard schema already has an `emotion`
column. Keep expressive data in from the start rather than adding it later — E4's
table says neutral hours and expressive hours are complements, not substitutes
(10 h neutral + 0.5 h expressive beat 5 h + 2 h). And add an **F0 standard
deviation** comparison against the reference recording to the eval harness: it is
the cheapest available proxy for prosodic range, and it is what DMOSpeech 2 used
to detect mode shrinkage.

### R5 — The loss curve lies, in three independent ways

**Evidence.**
[2603.10904](https://arxiv.org/html/2603.10904): training *and validation* loss
improved monotonically while DNS-MOS **fell 0.414** for a low-energy-variance
speaker. [Elhady et al.](https://arxiv.org/html/2506.00288v1): with vs without
English replay, validation perplexity was **3.35 vs 3.58** — essentially
identical — while downstream accuracy differed by 5.3 points and source-language
retention by **23 points (43.43 vs 20.12)**. E3's Telugu: 20 extra epochs "barely
moves the loss" on a saturated corpus, and only new material moved it.

**How it shows up.** It does not. That is the point. A loss curve is compatible
with a model whose naturalness is degrading, whose English has been destroyed, and
whose corpus has been exhausted.

**And it cannot be repaired afterwards.** Hsiao et al. tested model merging
(Linear, TIES, DARE) and LoRA-scaling as post-hoc fixes for a forgotten
capability: **all recovered essentially nothing**, and DARE made it worse (4.0
against a 70.0 original). Only replay worked, and only if it was there during
training.

**Mitigation.** Never gate on loss. Gate on the Set A/C/hard/long-form harness at
fixed checkpoints — **including step 100, 250 and 500**, because Elhady measured
7× parameter drift by step 100 — with per-language reporting, floors subtracted,
and a seen/unseen speaker split. This is the same discipline D5 applied to the
training loop: a check that cannot fail is worse than no check, and a loss curve
that cannot detect the failure mode is not a check.

### R6 — English is destroyed in the first few hundred steps

**Evidence.** LoRA-Whisper: full fine-tuning on new-language data alone took
base-language WER **11.68 → 32.29**, worst language **14.37 → 68.93**. Elhady:
cumulative parameter L2 drift from init **7× larger without English by step 100**,
15× by step 1000; source-language retention "collapses to near zero in the first
few steps". Hsiao et al.: **"the most serious forgetting appears in the TTS
stage"** — the exact stage we are running.

**How it shows up.** Indic metrics improve on schedule; nobody measures English
until the end; the English checkpoint is gone. E1 shipped into this and said so
explicitly ("we can always use the original checkpoint for English") — an option
we do not have, because our tokenizer swap means there is no original checkpoint
on our vocabulary.

**Our specific exposure is worse than any of these studies'.** The Sarvam swap
replaces every English embedding row and makes English ~14% more expensive in
tokens (findings §3.1), so English is being partly *re-learned*, not merely
preserved. And ternary weights mean less spare capacity than the full-precision
studies assumed.

**Mitigation.** D19's 20–25% replay, gated at steps 100/250/500. **Nothing in
this document is cheaper insurance.**

### R7 — The tokenizer swap costs more than it buys at 900 hours

**Evidence.** [CosyVoice's Exp-3 vs Exp-4](https://arxiv.org/html/2407.05407v2):
the same 300M model on **585 h** scored WER 3.93 / SS 67.85 with an English-only
BPE and **4.76 / 65.94** with a multilingual BPE plus multilingual speech
tokenizer — worse on both axes. The multilingual setup won at 171K h. This is the
only controlled evidence on the question and it is at our data scale, not theirs.

**How it shows up.** The Sarvam arm underperforms the Qwen3 control despite a
3.5–5.4× fertility advantage, and the natural reading ("the swap was wrong") is
confounded with ("we ran it at the wrong data scale").

**Mitigation.** D1 already mandates the Qwen3 control arm. This result means that
arm is load-bearing rather than ceremonial, and that **a Sarvam loss should be
reported as "at 900 hours" rather than as a verdict on the tokenizer.** It also
raises the value of D13's Brahmic warm-start, which is precisely an attempt to buy
back the low-data cost of a larger, less-trained vocabulary.

### R8 — Oversampling a small language is mistaken for having more of it

**Evidence.** E3's Telugu saturation (§7.1): 20 more epochs over 8.7 h "barely
moves the loss", and only new material moved it.
[UniMax](https://arxiv.org/pdf/2304.09151) puts a number on the harm: at
τ=3.33 the lowest-resource languages are repeated **over 100 times**, and
overfitting from repetition *"only emerges with scale"* — conspicuous at XXL,
invisible at the sizes where α≈0.3 was validated — and worsens with
longer training. They quote Hernandez et al.: **"even repeating 0.1% of data 100
times can be as harmful as halving model size."**

**How it shows up.** `sampled_by_language` looks balanced; `kept_by_language`
does not. Both are already emitted by `ShardDataset.summary()`. The damage then
looks like overfitting that nobody attributes to the sampler.

**Our exposure is at the bad end of UniMax's warning**: 1.7B parameters, a long
fine-tune, and a near-uniform target (D18) over languages whose raw hours differ
by an order of magnitude.

**Mitigation.** D18's epoch cap — which does not exist in
`_resample_languages` today, where `replace=True` has no ceiling. Log and gate on
*distinct* hours and *distinct speakers* per language after filtering, and treat
the repeat factor (`sampled / kept`) as a quantity with a hard limit. UniMax's own
ablation preferred **N=1 epoch** over N=5 and N=10; N≤3 is a defensible
operating range. A language that would need more than that is one we do not have
enough of, and the honest output is a warning, not a silent 6×.

---

## 15. Open questions, and what is genuinely unpublished

Five of these are gaps in the literature, not gaps in my search. Where that is the
case, running the experiment would produce a publishable result.

**Unpublished anywhere — experiments we could run:**

- **No TTS paper has ablated the English-replay ratio for Indic adaptation.** E1
  tested 0% and 50%. The 10–30% region, where every adjacent literature converges
  (§7.2), is unmeasured in TTS. **This is the single highest-value experiment in
  this document**: four short runs, and nobody has the answer.
- **No TTS paper runs a per-language sampling-temperature ablation.** All the
  α/T evidence in §7.1 is ASR, SSL or text. XTTS says only "a language batch
  balancer"; Indic Parler-TTS, RASMALAI and IN-F5 state nothing at all.
- **No published Indic zero-shot TTS result reports WER/CER *and* SECS on a
  standard benchmark.** IndicF5 and Indic Parler-TTS both report neither. In the
  500–2,000 h band there is nothing to compare against. **We would be setting the
  reference numbers.**
- **No measured CER impact of Indic text normalisation, number expansion or
  romanisation.** Given §13.3's finding that numbers and dates are the single
  highest-CER category, and findings §6's note that our number-to-words backend is
  still English, this is a gap with a direct bearing on our Set C.
- **Nothing on catastrophic forgetting under ternary or low-bit quantization.**
  Every replay number in §7.2 assumes full-precision fine-tuning. Reduced effective
  capacity is a reason to bias toward the high end of the replay range, but that
  is an inference, not a measurement.

**Unknowns that bear on our plan:**

- **Why did Malayalam fail at 10 h in E3 (CER 0.859) where Hindi succeeded
  (0.106)?** Unanswered anywhere, and it is the base rate for R3. This is the
  single most important unknown for our Tamil arm.
- **Does the Brahmic warm-start actually help?** Nobody has ablated it (§12 rank 9). We can be the ones who do — it is an arm in a harness that already exists.
- **What is the real English cost of the tokenizer swap?** Findings §3.1 measured
  +14% tokens; nobody has measured the quality cost, and D19's replay ratio is set
  against an unmeasured risk.
- **Is SPADE's "<5% of the original training data" real?** The abstract names
  neither base model nor data amounts (§9.2). If it holds, prune-and-distil is a
  much larger lever than this document gives it credit for.
- **Licences: IndicTTS and LIMMITS.** Both terms PDFs failed to load. Indic
  Parler-TTS characterises both as CC BY 4.0, but that is a third party's
  characterisation. Findings §7 makes licences a hard filter, and **these two
  corpora are a large fraction of §10.1's studio spine.** Resolve before shipping.

**One methodological note on this document.** The evidence base is overwhelmingly
*not* Indic, *not* our architecture, and *not* our data scale — the three studies
that are any two of those (E1, E3, E8) carry disproportionate weight here, and E3
is a HuggingFace model card with no formal evaluation beyond CER. Read the
confidence accordingly: the *direction* of every recommendation in §10 and §12 is
well supported by multiple independent sources; the *magnitudes* are borrowed.
