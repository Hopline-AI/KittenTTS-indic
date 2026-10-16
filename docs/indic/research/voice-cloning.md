# Speaker conditioning and zero-shot voice cloning

**Research note, 2026-10-16.** Companion to
`docs/indic/2026-10-16-findings-and-decisions.md`. Two kinds of claim appear
below and they are kept apart on purpose:

- **Measured here** — re-read out of `indic/data/teacher_forced.json` and out of
  the vendored `kittenml/kittentts2/` source on this machine. No new training, no
  new inference; the numbers were already on disk, we had just never arranged
  them this way.
- **From the literature** — every one carries a URL. Where a paper's number is
  not directly comparable to ours (different scorer, different corpus) it is said
  so inline rather than quietly averaged in.

Anything neither measured nor sourced is marked **UNVERIFIED**.

---

## 0. The four things to read if you read nothing else

1. **The open gate resolves, and it resolves in a third direction.** The reading
   "our reference layout is wrong" and the reading "teacher forcing cannot see
   identity" are *both* wrong as stated. Teacher-forced loss sees identity
   perfectly well — **0.273 nats/token, 58 nats over a target utterance** — but
   our two conditioning routes are **~90% redundant with each other**, so
   ablating either one alone shows almost nothing. The 2×2 that proves this was
   already in `teacher_forced.json`; §1.1.
2. **`role_swapped` was never capable of answering the layout question** and
   should not have been a gating control. It changes 4 marker tokens while
   leaving ~250 reference audio tokens and the speaker head untouched. §1.2.
3. **There is a third conditioning route we never measured, and the literature
   says it is the one that actually produces speaker similarity.** `s3gen` is
   conditioned on a CAM++ x-vector *and* the reference mel-spectrogram
   (`kittenml/kittentts2/s3gen.py:3921, :4032-4034`). CosyVoice 2 — the same
   architecture family — ablated the LM speaker embedding away entirely and
   speaker similarity did not move. §1.3.
4. **20-30 s is the wrong product spec for today's pipeline.** Our own decoder
   hard-truncates the reference to **10 s** (`vocoder.py:30`,
   `DECODER_COND_SECONDS = 10`) and the distilled decoder to **4 s**
   (`STUDENT_REF_SECONDS = 4.0`). Everything past 10 s reaches only the LM
   prefix, where it costs KV cache and buys little. **Ask for 8-15 s, and spend
   the product effort on prompt *quality* instead** — a noisy-reverberant prompt
   costs F5-TTS 0.273 SECS, which is larger than every length effect in the
   literature combined. §3.

---

## 1. The open gate: why the reference prefix showed almost nothing

### 1.1 The 2×2 was already in our data

Gate 2 reported two identity controls against two different baselines and never
put the four cells in one table. Here they are, from
`indic/data/teacher_forced.json` (8 English FLEURS examples, dev split, bf16,
reference cosine 0.65 — the run whose full control set survives on disk; §3.7 of
the findings doc quotes a separate 16-example run whose `speaker_zeros` row was
not kept):

| | reference prefix **on** | reference prefix **off** |
|---|---|---|
| **spk_proj on** | `correct` **4.32686** | `no_reference` 4.34566 |
| **spk_proj off** | `speaker_zeros` 4.36347 | `speaker_zeros_no_reference` **4.59945** |

Read as an interaction rather than as four separate rows:

```
joint identity budget (neither -> both)      0.27259 nats/token
  speaker head alone recovers                0.25379   (93.1% of it)
  reference prefix alone recovers            0.23598   (86.6% of it)
  sum of the two singles                     0.48978
  overlap                                    0.21719   (44.3% of the sum)
```

**And the split is statistically clean at n=8**, which matters because the gate
was failed on a control whose effect is indistinguishable from zero. Paired
per-example differences, same 8 English examples throughout:

| contrast | mean (nats/tok) | sd | t(7) | examples with the expected sign |
|---|---|---|---|---|
| neither − both (joint identity budget) | **+0.2726** | 0.1187 | **+6.49** | **8/8** |
| neither − spk only (value of the speaker head) | **+0.2538** | 0.0644 | **+11.15** | **8/8** |
| neither − ref only (value of the prefix) | **+0.2360** | 0.0855 | **+7.80** | **8/8** |
| spk only − both (prefix *on top of* the head) | +0.0188 | 0.1105 | +0.48 | 5/8 |
| ref only − both (head *on top of* the prefix) | +0.0366 | 0.1243 | +0.83 | 7/8 |
| `role_swapped` − correct | −0.0256 | 0.0710 | −1.02 | 3/8 |

Every *total* identity effect is unanimous across examples and 6-11 standard
errors from zero. Every *marginal* effect is within one standard error of zero.
The suite did compute one of the top rows — `speaker_zeros_no_reference` against
`no_reference`, +0.254, which it correctly baselined — but the **verdict** was
decided by `role_swapped`, the row that is 1.0 standard errors from zero and has
the wrong sign in 5 of 8 examples.

**Both readings in the findings doc are refuted by this.**

- *"Teacher-forced loss cannot see identity."* It can. Removing both identity
  routes costs **0.273 nats/token**, which over a 214-token English target is
  **58.3 nats ≈ 84 bits**. That is not a null result; it is comfortably larger
  than the 0.2-nat margin the gate itself set.
- *"Our reference layout is wrong."* A layout the checkpoint could not read would
  not recover **86.6%** of the identity budget on its own. `speaker_zeros` —
  reference prefix present, speaker head disabled — lands within 0.037 nats of
  `correct`. The prefix is being read.

What is actually true is the third thing: **the two routes carry nearly the same
information, so each is individually almost free to delete.** The gate measured
`no_reference` (+0.019) and `role_swapped` (−0.026) with the *other* route still
supplying identity, and correctly observed that nothing happened. The error was
in concluding that nothing was there.

> The findings doc's own commentary anticipated half of this — "the prefix carries
> two hundred audio tokens of the same voice, so the model has the speaker in
> context and does not need `spk_proj`" (`teacher_forced_loss.py`, the `BASELINE`
> comment). That reasoning was applied to the speaker head and a baseline was
> swapped for it. It was never applied in the other direction, to the prefix.

**This matches the literature exactly.** MiniMax-Speech ablates the same two
routes at *generation* time
([arXiv:2505.07916](https://arxiv.org/html/2505.07916v1), Table 4, SIM via the
Seed-TTS-eval WavLM-large verifier):

| conditioning | WER ↓ | SIM ↑ |
|---|---|---|
| learnable speaker encoder only | 1.252 | 0.730 |
| speaker encoder + audio prompt | 1.243 | **0.746** |
| audio prompt only | **1.207** | 0.726 |
| pretrained SV embedding only | 1.400 | 0.746 |
| pretrained SV embedding + prompt | 1.704 | 0.744 |

Either route alone reaches 0.726-0.746; both together reach 0.746. **The prompt
adds +0.016 SIM on top of an embedding, and the embedding adds +0.020 on top of a
prompt.** Our teacher-forced 2×2 and their generation-time SECS 2×2 say the same
thing: these routes are substitutes, not complements.

### 1.2 `role_swapped` could not have answered the layout question

`role_swapped` flips `uses_reference_role_tokens`, which swaps four marker ids
(`reference_text_start/end`, `reference_speech_start/end`) for the target
segment's own wrappers. Everything else is byte-identical: the same reference
transcript, the same ~250 reference audio tokens, the same `spk_proj` vector at
position 0.

So the control asks "can the model read the voice from a prefix whose brackets
changed?" — and the answer is obviously yes, because the voice is in the audio
tokens, not in the brackets. Measured −0.026 nats, i.e. the alternative layout is
*marginally better*, which is noise at n=8.

**A control that leaves the signal in place while removing the label cannot
distinguish a wrong label from an ignored one.** It belongs in
`DIAGNOSTIC_CONTROLS`, not in `IDENTITY_CONTROLS`, and the gate should not have
been failed on it. The honest statement of what Gate 2 established is:

> Target segment CONFIRMED (+4.69 nats). Identity conditioning CONFIRMED
> (+0.273 nats when both routes are removed). **Reference *layout* untested** —
> no control in the suite varies the layout while the reference is the only
> identity route.

### 1.3 The route nobody measured — and the literature says it is the important one

Read from the vendored source, not from documentation:

```
kittenml/kittentts2/s3gen.py:3921    self.speaker_encoder = CAMPPlus(...)
kittenml/kittentts2/s3gen.py:3747    CausalMaskedDiffWithXvec(spk_embed_dim=192)
kittenml/kittentts2/s3gen.py:4032-34 prompt_feat=ref_mels_24, embedding=ref_x_vector
kittenml/kittentts2/s3gen.py:3879    conds[:, :mel_len1] = prompt_feat
kittenml/kittentts2/model.py:355     "conditioning": self.codec.reference_conditioning(reference_wav)
kittenml/kittentts2/model.py:468     wav = self.codec.decode(reference["conditioning"], audio_ids)
```

So KittenTTS 2 conditions on the speaker **three** ways, not two:

| # | route | where | what it sees |
|---|---|---|---|
| 1 | pyannote 512-d x-vector → `spk_proj` → position 0 input embedding | LM | the whole clip |
| 2 | reference codec tokens in the prompt | LM | the whole clip |
| 3 | **CAM++ 192-d x-vector + the reference mel as `prompt_feat`** | **s3gen flow decoder** | **the first 10 s** |

Route 3 is in neither the gate nor the training loop. It is a frozen Chatterbox
`s3gen_meanflow` checkpoint (`vocoder.py:25-26`) and **no teacher-forced LM loss
can ever see it.**

This matters because routes 1-2 may be the *less* important ones. CosyVoice 2 —
same S3 supervised-semantic-token codec, same flow-matching decoder lineage —
deleted the LM speaker embedding on purpose
([arXiv:2412.10117](https://arxiv.org/html/2412.10117v1), Table 7):

| | test-zh CER / SS | test-en WER / SS | test-hard WER / SS |
|---|---|---|---|
| + LLM init. | 2.96 / 0.808 | 4.57 / 0.730 | 9.94 / 0.789 |
| **+ Drop Spk Emb.** | **2.56 / 0.804** | **3.81 / 0.740** | **9.66 / 0.778** |

Their reading, quoted: removal "resulted in a significant reduction in content
errors while maintaining speaker similarity, indicating that content information
is primarily modeled by the LM, and speaker information is mainly recovered by
the flow matching model." Their stated motive was that the utterance-level vector
"contains not only speaker identify but also language and paralanguage
information", which "hurts prosody naturalness and cross-lingual ability".

MiniMax-Speech's Table 4 above shows the same cost in a different form: the
*pretrained SV embedding* rows have the **worst WER** of the five (1.400 / 1.704
against 1.207-1.252), at no SIM advantage. A frozen verification embedding is the
one conditioning route that reliably buys speaker similarity with intelligibility.

**UNVERIFIED for our checkpoint:** we have not measured how much of KittenTTS 2's
cloning SECS comes from route 3. The claim "most of it" is an inference from
CosyVoice 2's ablation on a related architecture, not a measurement of ours. §1.5
says how to settle it for about an hour of inference.

### 1.4 Is there a published reason a prefix can be invisible under teacher forcing?

Yes — three distinct mechanisms, and they compound. None of them says "teacher
forcing cannot see identity"; all of them say "teacher forcing *dilutes* it".

**(a) Redundancy with the ground-truth prefix.** Under teacher forcing the model
is handed the target's own ground-truth audio tokens for every position before
the one it predicts. Codec-LM work reports those tokens carry identity: the
speaker-anonymisation line finds codec LMs "naturally disentangle linguistic
information into semantic tokens, while encoding speaker information and
recording conditions mostly in acoustic tokens"
([arXiv:2309.14129](https://arxiv.org/html/2309.14129v3)). At generation time
that prefix is the model's own output and carries only the identity the model has
already committed to — so an error is self-reinforcing in a way teacher forcing
structurally cannot show. This is plain exposure bias applied to identity
([arXiv:2509.17021](https://arxiv.org/html/2509.17021v1) formalises the
teacher-forcing / free-running gap for LM-TTS and recovers +0.04 SIM and −2.0 WER
on LibriSpeech by mixing free running into training).

**(b) Dilution over the target.** Identity is a *global* property. Whatever it is
worth is paid once and then amortised over every supervised token, while a
content corruption is paid at every token. That is exactly our measured ratio:
`shuffled_audio` +4.686 nats/token against the identity budget's 0.273.

**(c) The objective is not the metric.** The cleanest published statement is the
on-policy personalisation paper
([arXiv:2610.17838](https://arxiv.org/html/2610.17838), LLaSA-1B, Seed-TTS-Eval,
SIM via ECAPA-TDNN cosine):

| method | WER % ↓ | SIM ↑ |
|---|---|---|
| zero-shot (ICL) | 3.7 | 0.574 |
| **SFT (teacher-forced)** | 18.2 | **0.536** |
| SFT-ZSD (teacher-forced) | 8.3 | 0.586 |
| OPSD-TA (on-policy) | 3.7 | 0.609 |
| **GRPO (speaker-similarity reward)** | **3.0** | **0.647** |

Teacher-forced SFT *lowered* speaker similarity below the un-tuned zero-shot
baseline. Their words: diversified pseudo-pairs yield "little improvement in
speaker similarity and increases word error rate because the targets remain fixed
and teacher-forced". The general version of this is 11 years old — average
log-likelihood and sample quality "are largely independent" in high dimensions
([Theis, van den Oord & Bethge, arXiv:1511.01844](https://arxiv.org/abs/1511.01844)).

And at generation time the prefix is enormous, which is the asymmetry we were
looking for. VALL-E's ablation
([arXiv:2301.02111](https://ar5iv.labs.arxiv.org/html/2301.02111), Table 5,
WavLM-TDNN SPK): removing the acoustic prompt from the AR model takes SPK from
**0.585 → 0.236 with WER unchanged at 5.9**. Phoenix TTS
([arXiv:2608.11737](https://arxiv.org/html/2608.11737), Table 6): removing the
speaker prompt takes SIM **0.720 → 0.422** (en) and **0.778 → 0.560** (zh).

> **So the honest answer to the gate question is:** a reference prefix shows
> almost nothing under teacher forcing *when a second identity route is left
> intact*, and shows a small-but-real effect when it is the only route, and shows
> a 0.3-0.35 SECS effect at generation time. Teacher-forced loss is a **valid but
> ~20× attenuated** proxy for identity. It is fine as a wiring check. It is not
> fine as a design metric, and no amount of it will rank two reference layouts.

### 1.5 The cheapest experiment that settles the layout question

Three experiments, in increasing cost. **Run them in this order and stop as soon
as one answers.** Together they also close the "is the layout right" question
that `role_swapped` could not.

**E1 — Re-read the logits we already compute, per position (≈30 min of work, 0
extra GPU).** `teacher_forced_loss.py` already produces per-token log-probs and
throws away everything but the mean. Keep the per-position vector and plot the
`correct` − `speaker_zeros_no_reference` gap against target-token index.

The arithmetic gives this test an enormous dynamic range. The joint budget is
58.3 nats. The uniform ceiling over 6561 codec tokens is ln(6561) = 8.789
nats/token, so the budget **cannot** be confined to fewer than 6.6 tokens
(0.27 s). Therefore:

| if the identity signal lives in… | the per-token gap there would be |
|---|---|
| the first 25 tokens (1.0 s) | **2.33 nats** — unmissable |
| the first 50 tokens (2.0 s) | **1.17 nats** |
| all 214 tokens, flat | 0.273 nats — what the mean already says |

- **Decaying profile** → identity conditioning works and teacher forcing dilutes
  it. Switch to generation-time scoring and stop using loss for identity.
- **Flat profile** → the model uses the reference as generic acoustic context at
  every position, which is also fine but means per-position loss is a dead end.
- Either way the *shape* is new information and it costs one re-run of a harness
  that already exists.

**E2 — Re-run the layout controls with `speaker="zeros"` (≈1 GPU-hour, no new
code paths).** This is the control the suite is missing, and it is one line of
`VARIANT_SPEC`:

```
"role_swapped_spk_off":   {"reference": "plain_role", "speaker": "zeros"}
"merged_layout_spk_off":  {"reference": "merged",     "speaker": "zeros"}
```

all compared against `speaker_zeros` (correct layout, speaker off) rather than
against `correct`. With the speaker head disabled the reference prefix is the
**only** identity route, so a layout the checkpoint cannot read must give up most
of the 0.236 nats/token that the current layout recovers. Margin 0.2 nats is now
a meaningful threshold; at n=8 with the observed per-example spread it is
comfortably resolvable, and 16-32 examples per language makes it safe.

**One piece of external evidence now argues our layout is in fact the right one.**
The released `config.json` declares the reference-role ids itself —
`final_seg_id 158236`, `reference_speech_start_id 158237`,
`reference_speech_end_id 158238`, `reference_text_start_id 158239`,
`reference_text_end_id 158240`
([config.json](https://huggingface.co/KittenML/kitten-tts-2/raw/main/config.json)).
Five marker rows exist in the released vocabulary and the vendor's own config
names all five. A checkpoint that never saw reference-role tokens would not ship
ids for them. Combined with the 86.6% recovery in §1.1, the balance of evidence
is that `uses_reference_role_tokens = True` **is** the trained layout, and the
gate's FAIL was a property of the control, not of the layout.

`"merged"` is a **third layout we have never tried**, and the literature points
at it as the remaining alternative. CosyVoice's in-context format merges the reference transcript and
the target text into *one* text block and treats the reference speech tokens as
*already-generated* output — there is no `speech_end` between reference and
target at all. CosyVoice 2 writes the sequence as
`S, prompt_text, text, T, prompt_speech`
([arXiv:2412.10117](https://arxiv.org/html/2412.10117v1);
[arXiv:2407.05407](https://arxiv.org/html/2407.05407v2) states the merge
explicitly). Both layouts `prompt.py` currently builds close the reference with an
end marker and open a fresh text/speech block. Given that KittenTTS 2 inherits
CosyVoice's S3 codec and s3gen wholesale, the continuation layout is a live
hypothesis and costs one function to express.

**E3 — The generation-time 2×3, on 50 utterances (≈2 GPU-hours).** This is the
only experiment that measures the thing the product sells, and it is the one that
finally tells us how much route 3 is doing. Generate the same 50 sentences under
six conditions and score SECS with an independent verifier (§6):

| | s3gen conditioned on the real reference | s3gen conditioned on a **different speaker** |
|---|---|---|
| LM: prefix + spk_proj | baseline | **isolates route 3** |
| LM: neither (prefix dropped, speaker zeroed) | **isolates routes 1+2** | floor |
| LM: prefix only / spk_proj only | substitution check | — |

If the top-right cell tracks the *decoder's* speaker rather than the LM's, route
3 is doing the work and the entire LM-side identity design — including the
layout question — is a second-order concern. That would be the single most
budget-changing result available to this project for two GPU-hours.

> **Recommendation: run E1 and E2 before the next training run, and E3 before
> accepting any cloning quality claim.** None of them needs training data.

---

## 2. Conditioning mechanisms, compared

### 2.1 What the five mechanisms actually are

| | mechanism | representative systems |
|---|---|---|
| **(a)** | in-context audio prompt — reference codec tokens prefixed to the sequence | VALL-E, CosyVoice 2 (ICL), F5-TTS, KittenTTS 2 route 2 |
| **(b)** | speaker embedding through a projection into one input position | CosyVoice 1, YourTTS, KittenTTS 2 route 1, Continuous-Token Diffusion |
| **(c)** | both at once | MiniMax-Speech one-shot, KittenTTS 2 |
| **(d)** | cross-attention to reference features | NaturalSpeech 2, Phoenix TTS (Perceiver → 32 prefix latents) |
| **(e)** | prompt-tuning / adapter / per-speaker optimisation | SFT, OPSD, GRPO personalisation |

(e) is disqualified by the product spec — it is per-speaker fine-tuning — but it
is the right *upper bound* to quote, and the on-policy numbers in §1.4 are the
honest version of how much headroom is left above zero-shot.

### 2.2 The comparison table

SECS columns are **not mutually comparable**: scorer and test set differ per row
and are named. Inference overhead is given for *our* 1.7345 B checkpoint
(28 layers, 8 kv-heads × 128, bf16) so the column at least is internally
consistent; it is arithmetic from the measured architecture, not a timing.

| method | best published SECS / SIM | ref seconds needed | inference overhead (ours, arithmetic) | low-data behaviour | license of the representative impl. |
|---|---|---|---|---|---|
| **(a) in-context prompt** | 0.726 (MiniMax prompt-only, WavLM-large SV) · 0.585 → 0.236 when removed (VALL-E, WavLM-TDNN) · 0.720 → 0.422 when removed (Phoenix) | 3-12 s; capped at 10 s by our decoder, 12 s by F5-TTS | **25 tokens/s of reference**: 10 s = 250 tok = 27.3 MiB KV + ~0.87 TFLOP prefill; 30 s = 750 tok = 82.0 MiB + ~2.60 TFLOP | **Best.** No new parameters, no new head to fit; it reuses the LM's existing attention. The route most likely to survive <1000 h | F5-TTS MIT; CosyVoice 2 Apache-2.0; VALL-E unreleased |
| **(b) speaker embedding + projection** | 0.746 (MiniMax SpkEmbed zero-shot) · 0.730 (MiniMax learnable encoder) | whole clip; **UNVERIFIED** where a statistics-pooling x-vector saturates — we found no enrolment-duration curve for `pyannote/embedding` specifically | **2.1 MFLOP** (`spk_proj` = Linear(512,2048)+LN, 1,054,720 params) — **1.2 million× cheaper than a 30 s prefix** | Good, **if the encoder is frozen and pretrained.** A *learnable* encoder is another thing to fit on <1000 h; a frozen one is free supervision from VoxCeleb-scale data | depends on the encoder — §4 |
| **(c) both** | **0.746** (MiniMax one-shot) — only **+0.016** over prompt-only, **+0.020** over embedding-only | as (a) | (a) + (b) | Good, but see the redundancy result: you are paying twice for one signal | — |
| **(d) cross-attention to reference** | Phoenix TTS 0.720 en / 0.778 zh (WavLM-TDNN) with a 32-latent Perceiver prefix | not stated by either paper | New module, trained from scratch; attention over reference frames at every layer | **Worst in a low-data regime.** It is the only option that adds a *trained* conditioning pathway. NaturalSpeech 2 also warns that attending to the prompt directly "exposes too many details to the diffusion model and may harm the generation" | NaturalSpeech 2 unreleased; Phoenix TTS unreleased |
| **(e) per-speaker adaptation** | GRPO 0.647 vs 0.574 zero-shot (LLaSA-1B, ECAPA) | 1 utterance | per-speaker training — **disqualified** | n/a | — |

Sources: [MiniMax-Speech](https://arxiv.org/html/2505.07916v1) Table 4 ·
[VALL-E](https://ar5iv.labs.arxiv.org/html/2301.02111) Table 5 ·
[Phoenix TTS](https://arxiv.org/html/2608.11737) Table 6 ·
[NaturalSpeech 2](https://ar5iv.labs.arxiv.org/html/2304.09116) §3.3 ·
[on-policy personalisation](https://arxiv.org/html/2610.17838) Table 1 ·
[F5-TTS infer README](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/README.md).

### 2.3 Which gives the best SECS per unit of inference cost

**(b), by six orders of magnitude, and it is not close.** A frozen speaker
embedding through a projection costs 2.1 MFLOP and reaches within 0.016-0.020
SIM of the best combined system in MiniMax-Speech's own ablation. A 30 s
in-context prefix costs 2.60 TFLOP of prefill and 82 MiB of KV cache for the same
information.

But the per-FLOP ranking is the wrong ranking for us, for two reasons:

1. **The cost of the prefix is bounded and small in absolute terms.** 250 tokens
   of prefill is ~0.87 TFLOP; on any card that can run a 1.7 B model this is tens
   of milliseconds, paid once per reference and cacheable across every request
   for that speaker (`model.py:_reference_cache` already does this). It is not
   a per-token cost.
2. **The low-data column is the one that decides this project.** (a) adds zero
   parameters. (b) adds 1.05 M trainable parameters *and* inherits whatever the
   frozen encoder got wrong about Indian speakers (§4). (d) adds a whole trained
   module. With <1000 h, "adds no parameters" beats "is cheap at inference".

### 2.4 Which degrades least in a low-data regime

Ranked, with the reasoning:

1. **(a) in-context prompt.** No new parameters. The mechanism is the LM's own
   attention, which is already trained. It is also the only route whose quality
   improves automatically as the base LM improves.
2. **(b) with a *frozen, pretrained* encoder.** The encoder was trained on
   VoxCeleb-scale data you do not have; the only thing you fit is a 1 M-parameter
   projection. The caveat is distribution shift onto Indian speakers (§4) and
   the leakage cost CosyVoice 2 and MiniMax both measured — a frozen SV embedding
   carries language and channel, not only timbre.
3. **(b) with a *learnable* encoder.** MiniMax's learnable encoder beats the
   frozen one on WER (1.252 vs 1.400) at comparable SIM — but they trained on a
   "substantial subset" of a very large corpus. **UNVERIFIED** whether that
   advantage survives at <1000 h; we have found no ablation at our data scale.
4. **(d) cross-attention.** Most parameters, least data.
5. **(e)** disqualified.

---

## 3. How much reference audio is actually needed?

### 3.1 The direct answer

**8-15 seconds. Not 20-30.** Ask the user for 15 s so that trimming, leading
silence and a bad first take still leave 10 clean seconds. Specifying 30 s makes
the product worse: it is a longer ask at sign-up, it is more audio to upload, and
**the decoder throws away everything after the first 10 s anyway.**

### 3.2 Why — measured in our own code

| component | reference it actually consumes | source |
|---|---|---|
| s3gen flow decoder (default) | **first 10.0 s** | `vocoder.py:30` `DECODER_COND_SECONDS = 10` |
| s3gen flow decoder (distilled student) | **first 4.0 s**, ≤112 prompt tokens | `vocoder.py:41-42` |
| s3gen minimum | **6.0 s**, shorter clips are *tiled* (speech repeated, never silence-padded) | `vocoder.py:31`, `_load_decoder_reference` |
| CAM++ x-vector for s3gen | the same truncated 10 s | `s3gen.py:4016` — computed inside `embed_ref` |
| pyannote 512-d x-vector → `spk_proj` | **the whole clip**, untruncated | `speaker.py:__call__` |
| LM in-context prefix | **the whole clip**, untruncated, 25 tokens/s | `model.py:344`, no cap anywhere |

Upstream agrees on the same order of magnitude. F5-TTS: *"Use reference audio
<12s"*, and *"Long reference audio will be clip short to ~12s"*
([infer README](https://raw.githubusercontent.com/SWivid/F5-TTS/main/src/f5_tts/infer/README.md)).

So past 10 s the only consumers are the pyannote x-vector (statistics pooling —
diminishing returns, and 10 s is already past the knee for x-vectors) and the LM
prefix, which §1.1 shows is ~87% redundant with the embedding anyway. **Every
second past 10 costs 25 tokens of KV cache and buys the thinnest of the three
routes a little more of what it already had.**

### 3.3 Does 20-30 s help over 5-10 s? The published ablations

| system | scorer | 3 s | 5 s | 10 s | beyond |
|---|---|---|---|---|---|
| VALL-E, VCTK 108 spk | WavLM-TDNN | 0.382 | 0.423 | **0.484** | not measured |
| VALL-E, VCTK 11 unseen spk | WavLM-TDNN | 0.389 | 0.380 | 0.414 | not measured |
| **GroundTruth, same protocol** | WavLM-TDNN | **0.546** | **0.591** | **0.620** | — |
| Voicebox | WavLM-TDCNN SIM-r | default 3 s | — | "about the same ... up to 10 s" | — |
| NaturalSpeech 2 | human SMOS + prosody | 3 s | 5 s | 10 s, "longer prompt → higher similarity" but not uniformly | — |
| kNN-TTS (retrieval, not comparable) | SECS | — | intelligibility needs ~30 s | — | similarity plateaus ~**1 min**; baselines "benefit less ... beyond 10 to 30 seconds" |

Sources: [VALL-E](https://ar5iv.labs.arxiv.org/html/2301.02111) Table 6 ·
[Voicebox](https://ar5iv.labs.arxiv.org/html/2306.15687) §5.7 Fig. 6 ·
[NaturalSpeech 2](https://ar5iv.labs.arxiv.org/html/2304.09116) §5.5 Table 10 ·
[kNN-TTS](https://arxiv.org/html/2408.10771v2) Fig. 3.

**Read the GroundTruth row before reading the VALL-E rows.** Real recordings of
the same speaker also score 0.546 → 0.620 as the clip lengthens from 3 s to 10 s,
because *the scorer's own enrolment gets more reliable with more audio*. Of
VALL-E's +0.102, roughly **+0.074 is the metric, not the model**. The genuine
model-side gain from 3 s → 10 s is on the order of **+0.03 SECS**, and from 10 s
→ 30 s no one has published a number at all.

> **Pitfall to carry into our own eval:** if we A/B 10 s against 30 s references,
> we must hold the *scorer's* enrolment audio fixed (a separate, fixed-length
> clip) or we will measure the verifier warming up and call it cloning.

### 3.4 Does prompt quality matter more than prompt length? Yes, by 3×

[DAIEN-TTS / arXiv:2608.03011](https://arxiv.org/html/2608.03011v1), Table I,
Seed-TTS test-en, F5-TTS as the in-context baseline:

| prompt | WER % | **SECS** | MOS | SSMOS |
|---|---|---|---|---|
| clean | 2.30 | **0.590** | 3.86 | 3.46 |
| noisy + reverberant | 2.80 | **0.317** | 3.75 | 3.19 |

**−0.273 SECS from room noise and reverberation.** That is larger than the entire
3 s → 10 s length effect (+0.102, of which +0.03 is real) by roughly 9×, and
larger than every length effect in the table combined. The paper also states the
mechanism plainly: as an in-context system, F5-TTS *"tends to preserve the
reverberation from the speaker prompt"* — in-context methods *"tend to preserve
the environmental characteristics of the audio prompt."* On real-world recordings
(TITW) F5-TTS reaches only 0.481 SECS at 12.42% WER.

For an Indian consumer product where the reference is a phone recording in a
room with a ceiling fan, **this is the dominant term.** Product consequence:
record in-app with a level meter and a noise check, reject and re-prompt on a
bad take, and ask for 15 s of *clean* audio rather than 30 s of anything.

### 3.5 Do multiple reference clips help?

Weakly, and the evidence is thin.

- The standard baseline treatment is to **average the speaker embedding across
  all reference utterances for a speaker** ([kNN-TTS](https://arxiv.org/html/2408.10771v2)
  §IV). Averaging is cheap, is route (b) only, and reduces per-clip channel noise
  — it cannot help route (a) or route 3, which take one contiguous clip.
- kNN-TTS, which *is* built to exploit a large reference pool, needs ~30 s for
  intelligibility and plateaus at ~1 min — but it is retrieval, not conditioning,
  and is not comparable to our architecture.
- **UNVERIFIED:** we found no controlled ablation of "one 15 s clip vs three 5 s
  clips" for an in-context codec-LM TTS. If we want this, it is a cheap arm of
  E3 in §1.5: average the pyannote and CAM++ embeddings over N clips, pick the
  longest single clip for the prefix and for `prompt_feat`, and compare.

### 3.6 The recommended spec, with the reasoning stated

```
ask for      15 s          (so 10 s survives trimming and a bad first second)
accept       8 - 20 s
reject       < 6 s         s3gen tiles below this and the timbre estimate degrades
cap          20 s          past this, truncate for the LM prefix too: it costs
                           KV cache and reaches only the already-redundant route
quality gate louder than a measured noise floor, no obvious reverb,
             one speaker, no music
```

**If the product has already shipped a 20-30 s ask, do not change the ask — change
what we do with the audio.** Keep the full clip for the pyannote embedding (free,
and more audio genuinely stabilises a statistics-pooling x-vector), select the
best *contiguous 10-12 s window* for the s3gen `prompt_feat` and the LM prefix,
and select it by SNR rather than by taking the first 10 s as the code does today.
That is a strictly better use of 30 s than `wav24[:10*24000]`.

