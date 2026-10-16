# The audio codec: keep S3, and stop treating it as the bottleneck

**Research stream, 2026-10-16.** Literature survey with URLs for every claim.
Nothing here was measured on this machine; the one measurement we *do* own — the
Gate 1 round trip in `docs/indic/2026-10-16-findings-and-decisions.md` §3.6 — is
cited as such. Claims I could not verify are marked **[unverified]**.

---

## 1. Recommendation

**Keep S3TokenizerV2. Do not switch.** Not as a default-by-inertia — as the
conclusion of the arithmetic.

The reason is one line of arithmetic that most codec papers do not put on the
front page: **for a flat autoregressive LM, sequence length is frame rate ×
codebooks, not frame rate.** Our model is a flat single-stream AR over one shared
text+audio vocabulary. It has no depth transformer, no delay pattern, no local
per-frame decoder. So S3's 25 Hz × 1 codebook costs **25 AR steps per second of
audio**, and that number is at or below *every released codec in this survey
except one*:

| "12.5 Hz" codec | actually costs |
|---|---|
| Mimi (8 codebooks) | 100 tok/s — **4× worse than S3** |
| DualCodec-12.5Hz (8 cb) | 100 tok/s — 4× worse |
| NanoCodec 1.78 kbps (13 cb) | 162.5 tok/s — 6.5× worse |
| Qwen3-TTS-Tokenizer-12Hz (16 cb) | 200 tok/s — 8× worse |
| SNAC 24 kHz (3 cb @ 12/23/47 Hz) | 82 tok/s — 3.3× worse |

Halving the *frame* rate while octupling the codebooks is not a sequence win. It
is a sequence loss that reads as a win because the headline number went down.

The single exception is **TaDiCodec** at 6.25 Hz × 1 codebook = 6.25 tok/s, and
it is the runner-up for exactly that reason.

**Runner-up: TaDiCodec** ([arXiv 2508.16790](https://arxiv.org/html/2508.16790v1),
Apache-2.0, [amphion/TaDiCodec](https://huggingface.co/amphion/TaDiCodec)). 4×
shorter sequences than S3 at equal or better reconstruction — on *its* test set,
in *its* languages.

**The condition under which the runner-up wins:** when we are paying for a
from-scratch audio-LM pretrain anyway — i.e. the KittenTTS 2 checkpoint is being
abandoned, or its Stellon Labs license (§7 of the findings doc, terminates at $1M
revenue) forces a clean-room rebuild — **and** a 6.25 Hz round trip on our own
harness shows Indic holding against the English control the way 25 Hz already
does. Short of both, it loses, and §2 says why.

### The trade-off, stated plainly

Keeping S3 means accepting that we will not get a 2–4× sequence reduction, and
therefore that RTF and training cost must come from somewhere else: the ternary
kernels, the KV cache (112 KiB/token — a 30 s reference prefix is 750 tokens =
82 MiB before a word is generated), chunk granularity, and batching. Those are
all real levers and none of them requires retraining the audio vocabulary.

Switching means throwing away the 6561 trained audio-token embedding rows. The
vocabulary swap (§3.5) deliberately kept them **bit-identical** because they are
the part of the checkpoint that knows what speech is. Replace the codec and those
rows are noise, and a 1.7B speech LM must learn audio modelling from scratch on
**<1000 h**. The project's stated primary constraint is data efficiency on a small
corpus. A codec switch is the single most data-expensive change available.

---

## 2. The three numbers that decide it

### 2.1 Halving the frame rate at a fixed 6561-code vocabulary costs 12 points of speaker similarity

This is the closest thing in the literature to a controlled experiment on our
exact question, and it happens to use **our exact codebook size**. Qwen-Audio-3.0-TTS
took CosyVoice3's supervised-FSQ tokenizer — same family as ours, same 3⁸ = 6561
codes — and moved it from 25 Hz to 12.5 Hz
([arXiv 2607.23938](https://arxiv.org/html/2607.23938v1), Table 2, SEED-TTS-Eval):

| Tokenizer | Codes | Rate | test-en WER | test-en SIM | test-zh CER | test-zh SIM |
|---|---|---|---|---|---|---|
| CosyVoice3 | 6,561 | 25 Hz | 2.57 | **73.60** | 1.45 | 80.60 |
| Qwen-Audio-3.0 | 6,561 | **12.5 Hz** | 3.21 | **61.64** | 2.59 | 72.44 |
| Qwen-Audio-3.0 | 19,683 | 12.5 Hz | 2.56 | 77.58 | 1.48 | 83.25 |
| Qwen-Audio-3.0 | 59,049 | 12.5 Hz | 2.37 | 77.49 | 1.23 | 83.09 |

The naive halving costs **12 points of speaker similarity** and 25% relative WER.
For a product whose whole proposition is zero-shot cloning from 20–30 s, 73.6 →
61.6 is not a trade, it is the feature breaking.

The third row is the interesting one: **3⁹ = 19,683 codes recovers all of it and
more** — WER 2.57 → 2.56, SIM 73.60 → **77.58** — at half the sequence. One extra
FSQ dimension. That is the only published recipe for a real 2× win on our
architecture, and §5 says what it would take.

The same paper's ASR table (Table 1) shows the loss is upstream of synthesis:
FLEURS-en 5.43 → 5.33 but CV-en 13.07 → 15.40, CV-ja 15.61 → 18.68. The tokens
stop carrying phonetic detail, and no decoder recovers it.

### 2.2 Single-codebook costs ~1.5 WER points and buys a 4× shorter sequence — and costs *nothing* in speaker similarity

The best cross-codec table in the field is TaDiCodec's Table 1, resynthesis on
SeedTTS test-en ([arXiv 2508.16790](https://arxiv.org/html/2508.16790v1)). Our
incumbent is in it, under its upstream name:

| System | Hz | **tok/s** | cb | kbps | WER↓ | SIM↑ | UTMOS↑ |
|---|---|---|---|---|---|---|---|
| **CosyVoice 2 Tokenizer (= our S3V2)** | 25 | **25** | 1 | 0.325 | 4.10 | **0.68** | 3.65 |
| DualCodec (8 cb) | 12.5 | 100 | 8 | 1.225 | **2.57** | 0.64 | 3.78 |
| Mimi (8 cb) | 12.5 | 100 | 8 | 1.1 | 3.99 | 0.57 | 3.21 |
| TaDiCodec (w. dct) | 6.25 | **6.25** | 1 | 0.0875 | 2.73 | 0.69 | 3.73 |

Against DualCodec's 8-codebook stack at **4× the tokens**, S3 gives up 1.5 WER
points — and *wins* on speaker similarity, 0.68 vs 0.64. It has the **highest SIM
of any single-codebook codec in that table**, beating X-Codec2 (0.62), BiCodec
(0.61), BigCodec (0.61) and TAAE (0.41).

That is not an accident, and it is the architectural reason to keep it. S3 tokens
are *supervised semantic* tokens — speaker identity is deliberately not in them;
it arrives through the decoder's reference conditioning. For zero-shot cloning
that is the correct factorisation: the LM predicts content, the decoder supplies
the voice, and 20–30 s of user audio feeds the decoder rather than having to be
memorised into the token stream. Acoustic RVQ codecs put timbre *in the tokens*,
which is why they need 8–16 codebooks to sound right and why their SIM is worse
anyway.

### 2.3 Our 17 s first-chunk latency is a chunking problem, not a token-rate problem

From our own README (`README.md:233`): *"On an A100, a 936-character passage
yielded its first 21 s of audio after 17 s."*

**The first chunk is 21 seconds of audio.** At 25 Hz that is ~525 AR steps in 17 s
— 32 ms/step, consistent with the published RTF 0.78. The latency is not the token
rate; it is that the chunker emits ~380-character chunks and streaming is
chunk-level, not token-level (`README.md:232`).

A perfect 2× codec takes 17 s → 8.5 s. Still unusable. Cutting the chunk to ~2 s
of audio takes it to ~1.6 s **with the codec unchanged.** The codec is the wrong
lever for the stated latency problem, and that single fact removes the main
argument for disrupting it.

Corroborating: Qwen3-TTS measured both rates in one system
([arXiv 2601.15621](https://arxiv.org/html/2601.15621v1), Table 2). At concurrency
1 the 12 Hz model has *higher* RTF than the 25 Hz model (0.313 vs 0.253), because
its 16 codebooks mean 200 tok/s against the 25 Hz tokenizer's 25. Frame rate alone
predicted the opposite.

---

## 3. The comparison table

tok/s is **AR steps per second for a flat single-stream LM** = frame rate ×
codebooks. That is the column that sets our inference *and* training cost.
Quality is WER / SIM / UTMOS on SeedTTS test-en resynthesis from
[TaDiCodec Table 1](https://arxiv.org/html/2508.16790v1) where available, so the
column is internally comparable; other sources are labelled.

| Codec | tok/s | codebooks | kbps | quality (WER↓/SIM↑/UTMOS↑) | streaming | license | Indic evidence |
|---|---|---|---|---|---|---|---|
| **S3TokenizerV2 / CosyVoice 2** (incumbent) | **25** | 1 (FSQ 3⁸=6561) | 0.325 | 4.10 / **0.68** / 3.65 | encoder causal-capable; our s3gen decoder is 2-step meanflow, chunkable | Apache-2.0 ([S3Tokenizer](https://github.com/xingchensong/S3Tokenizer), [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)); our s3gen weights MIT ([chatterbox-turbo](https://huggingface.co/ResembleAI/chatterbox-turbo)) | **No Indic in training data**; but **measured on our data**: all 5 Indic languages round-trip at MCD 0.70–0.82× the English control (§3.6) |
| **TaDiCodec** (runner-up) | **6.25** | 1 (BSQ 2¹⁴) | 0.0875 | **2.73 / 0.69 / 3.73** | **No** — bidirectional attention, 10–32 diffusion steps | Apache-2.0 ([repo](https://github.com/HeCheng0625/Diffusion-Speech-Tokenizer)) | None. Emilia: en/zh/de/fr/ja/ko only. Its own fr/ja WER ≈ 20 |
| Qwen-Audio-3.0 tokenizer | 12.5 | 1 (FSQ 3¹⁰=59049) | ~0.198 | 2.37 / 0.775 (SEED-TTS) | chunked flow + causal vocoder | **proprietary, API-only** — not downloadable | None listed (16 langs, no Indic) |
| Qwen-TTS-Tokenizer-25Hz | 25 | 1 (32768) | ~0.375 | not reported as resynthesis | yes, block-wise DiT, ~190 ms first packet | Apache-2.0 | None mentioned |
| Qwen3-TTS-Tokenizer-12Hz | **200** | 16 RVQ (2048) | ~2.2 | PESQ-WB 3.21 / STOI 0.96 / UTMOS 4.16 / SIM 0.95 (LibriSpeech) | yes, fully causal, no lookahead | Apache-2.0 | None mentioned |
| Mimi (Kyutai/Moshi) | **100** | 8 RVQ (up to 32) | 1.1 | 3.99 / 0.57 / 3.21 | yes; 80 ms frame, 160 ms theoretical | CC-BY-4.0 ([kyutai/mimi](https://huggingface.co/kyutai/mimi)) | not stated |
| DualCodec-12.5Hz | **100** | 8 (sem 16384 + ac 4096) | 1.225 | 2.57 / 0.64 / 3.78 | no | MIT ([repo](https://github.com/jiaqili3/DualCodec)) | Emilia; en/zh evaluated only |
| DualCodec-25Hz | **300** | 12 | 3.1 | TTS WER 3.40 / SIM 0.57 | no | MIT | same |
| NanoCodec (NVIDIA) 1.78 kbps | **162.5** | 13 FSQ (2016) | 1.78 | CER 2.42 / SECS 0.862 (MLS) | causal decoder, non-causal encoder | NVIDIA Open Model License — **commercial OK** ([card](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps)) | **105 languages, 28.7 k h** — best declared coverage here, but language list unverified for Indic |
| NanoCodec 0.6 kbps | 50 | 4 FSQ (4032) | 0.6 | CER 7.79 / SECS 0.701 | same | same; card says *not* for general-purpose use | same |
| SNAC 24 kHz | **82** | 3 @ 12/23/47 Hz | 0.98 | not in a comparable table | no | MIT ([card](https://huggingface.co/hubertsiuzdak/snac_24khz)) | **Real Indic use**: [snorTTS-Indic-v0](https://huggingface.co/snorbyte/snorTTS-Indic-v0) ships 9 Indic languages on SNAC-24k — existence proof, no quality numbers |
| X-Codec2 | 50 | 1 (65536) | 0.8 | 2.63 / 0.62 / 3.68 | no | **CC-BY-NC-4.0 — fails our license filter** ([card](https://huggingface.co/HKUSTAudio/xcodec2)) | "multilingual semantic support", unlisted |
| X-Codec2-25TPS-24k | 25 | 1 (65536) | 0.40 | best UTMOSv2 among 25 Hz codecs on CV-17 | not mentioned | **[unverified]** — fork of a CC-BY-NC parent ([arXiv 2601.20185](https://arxiv.org/html/2601.20185v2)) | Paper claims training on 100+ langs **naming Hindi, Tamil, Bengali**; no Indic row in any table |
| BiCodec (Spark-TTS) | 50 (25 variant exists) | 1 (8192) + global | 0.65 | 3.05 / 0.61 / 3.68 | no | **[unverified]** ([arXiv 2503.01710](https://arxiv.org/html/2503.01710v1)) | none |
| BigCodec | 80 | 1 (8192) | 1.04 | 3.25 / 0.61 / 3.59 | no | **[unverified]** | none |
| WavTokenizer | 40 / 75 | 1 (4096) | 0.48 / 0.90 | 6.65 / 0.48 / 3.36 | no | **[unverified]** | none |
| TAAE / Stable Codec | **25** | 1 | 0.4 | **11.08** / 0.41 / 3.87 | no | **[unverified]** | none |
| FocalCodec-Stream | 50 | 1 binary (2k–65k) | 0.55–0.80 | dWER 3.68, SIM 97.0, UTMOS 3.85 (LibriSpeech) | **yes — 80 ms theoretical** | **[unverified]** ([arXiv 2509.16195](https://arxiv.org/html/2509.16195v1)) | **English audiobooks only** (LibriTTS/Libri-Light) |
| ZipCodec | 6.25 Hz | **[unverified]** | 0.80 | self-reported SOTA among streaming | yes, 160 ms, 842M params | **[unverified]** ([arXiv 2609.11642](https://arxiv.org/abs/2609.11642)) | not stated |
| SpeechTokenizer | **400** | 8 | 4.0 | 7.98 / 0.46 / 2.47 | no | **[unverified]** | none |
| DAC | 75–774 | 1–9 | 0.75–7.74 | 12.74 (VQ) / 20.08 (RVQ) | no | MIT **[unverified]** | none |
| EnCodec | 150–2400 | 2–32 | 1.5–24 | 5.36 / 0.48 / 1.54 | yes | MIT **[unverified]** | none |
| FACodec (NaturalSpeech 3) | ~480 | 6 + timbre | ~4.8 | **[unverified]** | no | **[unverified]** | none |
| Vocos / BigVGAN | n/a | n/a | n/a | vocoders, not tokenizers | — | MIT (Vocos) **[unverified]** | IndicF5 uses Vocos — mel pipeline, **no discrete codec at all** |

Two rows are the whole story: our incumbent sits at 25 tok/s with the best SIM of
any single-codebook codec, and the only row below it on tok/s is TaDiCodec.

---

## 4. The evidence, by question

### 4.1 Semantic vs acoustic disentanglement, and whether it helps low-data training

It does, and it is the main reason S3 is the right *kind* of codec for this
project rather than merely an adequate one.

- A supervised-semantic tokenizer derives its codes from an **ASR encoder's**
  intermediate representation ([CosyVoice 2, arXiv 2412.10117](https://arxiv.org/html/2412.10117v3)).
  The LM therefore learns a near-phonetic sequence, not a compressed waveform.
  On <1000 h that is the difference between learning *what to say* and learning
  *what speech sounds like*.
- The cost shows up as speaker info being absent from the tokens — which is
  precisely what 2.2 shows is an advantage for cloning, because the decoder
  supplies it from the reference instead.
- DualCodec quantifies how much semantic content survives frame-rate reduction by
  decoding **RVQ-1 only** ([arXiv 2505.13000](https://arxiv.org/html/2505.13000v2),
  Table 2): ASR WER 2.98 at 25 Hz vs **6.94 at 12.5 Hz** at the same 16384
  codebook. Same result as 2.1 from a different lab and a different architecture.
- OmniCodec asserts "a single-layer codebook is insufficient at low frame rate"
  ([arXiv 2603.20638](https://arxiv.org/html/2603.20638v1)) but — by its own
  text — gives no numbers for it. Marked as an assertion, not evidence.
- Counter-evidence worth knowing: TaDiCodec gets a single codebook down to 6.25 Hz
  by **conditioning the decoder on the transcript**. Its own ablation shows that
  without text conditioning, WER exceeds 10 at 12.5 Hz. So "single codebook at
  very low frame rate" is real but is bought with a text-conditioned,
  non-streaming diffusion decoder.

### 4.2 Streaming and decoder latency

- Our s3gen decoder is **already the optimised case**: ResembleAI chatterbox-turbo
  with a meanflow flow-matching head converging in **2 CFM steps**
  (`kittenml/kittentts2/vocoder.py:33`), plus an optional BitNet-quantised student
  flow at 1 step. Most "25 Hz flow-matching is slow" arguments assume 10–32 steps.
- Qwen3-TTS is the one source that measures the decoder separately
  ([Table 2](https://arxiv.org/html/2601.15621v1)): their 25 Hz DiT decoder costs
  **25–147 ms per packet** against the 12 Hz causal ConvNet's 4–5 ms. That is a
  real argument — but against *expensive* decoders, not against 25 Hz. Ours is 2
  steps.
- The genuinely streaming low-token-rate options are FocalCodec-Stream (80 ms
  theoretical, but 50 tok/s and English-only) and ZipCodec (160 ms, structure
  **[unverified]**). Neither beats S3 on tok/s *and* has Indic evidence.
- NanoCodec's ablation ([arXiv 2508.05835](https://arxiv.org/html/2508.05835v1),
  Table 1) prices causality honestly: at 1.1 kbps / 12.5 fps, going fully causal
  costs CER 3.617 → 5.490 and SECS 0.788 → 0.735. A **causal decoder with a
  non-causal encoder** is nearly free. That is the configuration to copy if we
  ever fine-tune s3gen.

### 4.3 Indic coverage in codec training data

Thin everywhere, and this is the finding that most favours staying put.

- **Nobody in this survey reports a codec reconstruction number on an Indic
  language.** I searched for one specifically and did not find it.
- NanoCodec declares the broadest coverage — 105 languages, 28.7 k h from Common
  Voice + MLS — but does not publish the list, so Indic presence is
  **[unverified]**. Common Voice does carry Hindi/Tamil/Bengali, so it is likely
  non-zero and small.
- The X-Codec2-25TPS-24k paper names Hindi, Tamil and Bengali among 100+ training
  languages but reports results only on 7 European languages
  ([arXiv 2601.20185](https://arxiv.org/html/2601.20185v2)).
- TaDiCodec, DualCodec, FocalCodec, BigCodec, WavTokenizer, X-Codec2, BiCodec,
  TAAE: **no Indic**.
- The only field evidence of a codec carrying Indic is
  [snorTTS-Indic-v0](https://huggingface.co/snorbyte/snorTTS-Indic-v0), a LLaMA-3.2-3B
  TTS over SNAC-24k covering 9 Indic languages — an existence proof with no
  published quality numbers.
- IndicF5, the quality bar we are chasing, **uses no discrete codec at all**: it
  is F5-TTS flow-matching over mel with a Vocos vocoder. So "what codec does the
  Indic SOTA use" has no answer; the question is ours to settle.

**Against all of that, we hold the only Indic codec measurement in the room**:
§3.6, 16 FLEURS clips × 5 languages, every Indic language reconstructing at MCD
0.70–0.82× the English control with duration preserved exactly. Trading a measured
PASS for an unmeasured codec with no Indic data and no Indic numbers is trading
information for hope.

### 4.4 Licenses — a short hard filter

Commercial use is a hard filter, and it eliminates one serious candidate outright:

- **X-Codec2 is CC-BY-NC-4.0.** Out. Its 25 Hz/24 kHz fork inherits an unresolved
  license question (**[unverified]**) and should be treated as out until the
  LICENSE file is read.
- **Qwen-Audio-3.0-TTS is API-only / proprietary.** The 19,683-code result in 2.1
  is a recipe we could reproduce, not a checkpoint we can download.
- Clean: S3Tokenizer/CosyVoice **Apache-2.0**, our s3gen weights **MIT**,
  TaDiCodec **Apache-2.0**, DualCodec **MIT**, SNAC **MIT**, Mimi **CC-BY-4.0**,
  NanoCodec **NVIDIA Open Model License** (card states commercial use is ready).
- The binding license risk in this project is not the codec. It is the KittenTTS 2
  checkpoint itself (§7: terminates above $1M revenue/funding, §4b forbids using
  its outputs to train other models).

---

## 5. What would have to be measured to validate this on our data

We have a working harness: `indic/bench/codec_roundtrip.py`. It encodes real
FLEURS audio, decodes through s3gen, and scores against the original with English
as the control. Five runs, in order of value. **Runs 1–3 require no new code.**

**Run 1 — close the existing gate. Highest value, cheapest.**

```
export HF_TOKEN=...   # IndicConformer is gated
python -m indic.bench.codec_roundtrip --clips 20 --asr auto --device cuda \
       --out indic/data/codec_roundtrip_asr.json
```

The §3.6 PASS is **spectral-only**, and the harness says so in its own verdict
text. MCD cannot distinguish a clean reconstruction that says a different word.
The whole "S3 handles Indic" claim — and therefore this recommendation — rests on
half a measurement until `cer_delta` exists for ta/te. *Nothing below matters more
than this.*

**Run 2 — measure the metric our product actually turns on.**
Speaker similarity after round-trip is **not measured anywhere today**, and 2.1
says it is the first thing to break. The 160 pairs are already on disk at
`.cache/codec_roundtrip/<lang>/{id}.orig.wav` and `{id}.recon.wav`. Score them
with `microsoft/wavlm-base-plus-sv` per **D8** — a standalone pass, no re-encode.
Report per-language SECS as a ratio to English, the same way MCD is. If Indic SECS
tracks English, the recommendation holds with a second leg under it.

**Run 3 — remove the harness's own thumb on the scale.**
`roundtrip()` conditions s3gen on *the clip being reconstructed*
(`codec_roundtrip.py`, docstring: "it hands s3gen the speaker for free"). That is
the right test for "what do the tokens carry", and it is also why every
semantic codec scores well on SIM. Add a second arm conditioning on a **different
20–30 s clip of the same speaker** — FLEURS records each sentence by several
speakers, and **D10** already forbids drawing references from the scored set. The
gap between the two arms is the part the LM must actually supply. Run it before
quoting any SIM number externally.

**Run 4 — the latency check that decides whether the codec matters at all.**
Time to first audio at chunk sizes of 380 / 120 / 60 characters, codec unchanged
(`text.py` chunker; see **D4**). If 60 characters gets first audio under 2 s,
§2.3 is confirmed on our hardware and the codec is permanently off the latency
critical path. This is an hour of work and it retires the strongest argument for
switching.

**Run 5 — only if a switch goes back on the table: port TaDiCodec behind `S3Codec`.**
The harness takes any object with `encode_reference` / `reference_conditioning` /
`decode`. Two module-level constants are hardcoded to S3 and must become
per-codec, or the duration and token-rate columns silently lie:

- `TOKEN_RATE = 25` → feeds `row["token_seconds"]`
- `DECODE_PAD_TOKENS = 3` → feeds `duration_ratio` and the spectral trim

TaDiCodec also needs the transcript at decode time, which `clip["text"]` already
carries. Then:

```
python -m indic.bench.codec_roundtrip --codec tadicodec --langs hi,bn,ta,te,mr,en \
       --clips 20 --asr auto --out indic/data/codec_roundtrip_tadicodec.json
```

**The number that decides it:** per-language `MCD / en` at 6.25 Hz versus the
measured 0.70–0.82× at 25 Hz. At 6.25 Hz each token spans ~2.6 phonemes (the
arithmetic is in [arXiv 2606.16969](https://www.alphaxiv.org/abs/2606.16969))
against ~0.65 at 25 Hz, so the codec must have learned Indic phonotactics rather
than merely Indic phones. **If any Indic language's ratio rises above 1.0 while
English holds, 6.25 Hz does not carry Indic and the answer is final.** Listen to
the written wavs before acting either way — §3.6 says so for a reason.

---

## 6. The explicit answer: is switching away from S3 worth the disruption?

**No.** Four reasons, in descending order of force.

1. **It costs the pretrained audio vocabulary, which is the asset.** The 6561 S3
   rows in `embed_tokens` are what the 1.7B model knows about speech, and §3.5
   kept them bit-identical through the Sarvam swap for exactly that reason. A new
   codec makes them noise. Relearning them needs the audio-LM pretraining corpus
   we do not have, on a <1000 h budget that is already the project's tightest
   constraint. Every other item on this list is secondary to this one.

2. **There is almost nothing to win.** S3 is at 25 AR tok/s. Only TaDiCodec beats
   it, and only by moving to a non-streaming, text-conditioned diffusion decoder
   trained without a word of Indic. Every other "lower frame rate" candidate is
   *more* AR steps on our flat LM (§1).

3. **The problem the switch was meant to solve is elsewhere.** 17 s to first audio
   is a 21-second first chunk (§2.3). A perfect 2× codec leaves it at 8.5 s;
   chunking fixes it to ~1.6 s with the codec untouched.

4. **We would be trading a measurement for a hope.** We hold the only Indic codec
   round-trip number in this survey — Indic reconstructing *better* than English
   at 25 Hz (§3.6). No candidate publishes an Indic number, and most publish no
   Indic training data.

**What would change the answer.** Any *one* of: (a) the KittenTTS 2 license forces
a clean-room rebuild, so the audio vocabulary is being retrained regardless — then
reopen with TaDiCodec and the 12.5 Hz/19,683 recipe both on the table; (b) Run 1
comes back FAIL on ta/te with ASR, which turns the recommendation inside out, and
the harness's own verdict text already tells you to fine-tune s3gen before
changing codec; (c) Run 4 shows chunking cannot get first audio under ~2 s, making
token rate the binding constraint after all.

**If the audio vocabulary is being retrained anyway, do not pick TaDiCodec by
reflex.** Pick the 12.5 Hz **3⁹ = 19,683-code** supervised-FSQ tokenizer of §2.1.
It is a 2× sequence win with *better* WER and SIM than our 25 Hz today, it keeps
the semantic/acoustic factorisation that makes 20–30 s cloning work, it keeps the
2-step meanflow decoder, and it costs 13,122 extra embedding rows — 13,122 × 2048
× 2 B = **54 MB** at bf16, against the 342 MB the Sarvam swap already removed, and
well inside the 4,495 tokens/step ceiling fitted in §3.4. It is a build, not a
download — Qwen-Audio-3.0-TTS is API-only — but the build is "train an FSQ
bottleneck on an ASR encoder", which is the one piece of this stack that does not
need a billion-parameter budget.

---

## 7. Claims I could not verify

- **ZipCodec's codebook structure.** 0.80 kbps at 6.25 Hz is 128 bits/frame, which
  cannot be one index. The abstract says "scalar spherical quantizer" and nothing
  more ([arXiv 2609.11642](https://arxiv.org/abs/2609.11642)). Its tok/s is
  therefore unknown and it cannot be ranked.
- **Whether NanoCodec's 105 languages include any Indic language**, and in what
  quantity. The list is not published.
- **Licenses for** BigCodec, WavTokenizer, TAAE/Stable Codec, BiCodec/Spark-TTS,
  FocalCodec, FACodec, and the X-Codec2-25TPS-24k fork. None were read from a
  LICENSE file this session. All of these fail on tok/s anyway except the fork.
- **FACodec's specifications and quality.** Reported here from memory of the
  NaturalSpeech 3 design and not re-derived; at ~480 tok/s it is out regardless.
- **Mimi's released codebook count.** Sources disagree between 8 (Moshi's
  configuration) and 16/32 (the full RVQ stack). 8 is the better-supported number
  for the released setup; either way it is ≥100 tok/s.
- **Vocos and BigVGAN first-chunk latency** under matched chunking. No source
  compares them directly; our 2-step meanflow path makes the question moot for now.
- **Whether TaDiCodec's decoder can run without a transcript.** Its repo roadmap
  lists "add auto-ASR for text input" as unchecked, which implies it currently
  cannot ([repo](https://github.com/HeCheng0625/Diffusion-Speech-Tokenizer)).
- **Web search budget was exhausted** partway through this session, so anything
  published in the last few weeks of 2026-10 may be missing. The newest work found
  here is ZipCodec (2026-09) and the codec-probing study
  [arXiv 2610.20955](https://arxiv.org/html/2610.20955) (2026-10).

---

## 8. One thing this survey turned up that belongs to another stream

`vocoder.py` sets `DECODER_COND_SECONDS = 10` and the student flow
`STUDENT_REF_SECONDS = 4.0` / `STUDENT_MAX_PROMPT_TOKENS = 112`. The product
promises cloning from **20–30 s**, so the decoder sees at most 10 s of it — and
the quantised student path at most ~4.5 s. Meanwhile `prompt.py` puts the *entire*
reference through as audio tokens with no truncation: 30 s = **750 tokens** of
prefix, 82 MiB of KV cache at the measured 112 KiB/token, before a single output
token.

So the 20–30 s budget is being spent almost entirely on the LM prefix and barely
at all on the decoder that actually carries the voice. That is a conditioning and
prompt-budget question, not a codec question — but it is the clearest efficiency
win this survey ran into, and it is worth a measurement: sweep reference length
against SECS and see where the curve flattens.
