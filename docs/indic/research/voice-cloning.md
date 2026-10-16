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

## 0. What to read if you read nothing else

1. **The open gate resolves, and it resolves in a third direction.** "Our
   reference layout is wrong" and "teacher forcing cannot see identity" are
   *both* wrong as stated. Teacher-forced loss sees identity perfectly well —
   **0.273 nats/token, 58 nats over a target utterance, 8/8 examples,
   t(7) = +6.49** — but our two conditioning routes are **~90% redundant with
   each other**, so ablating either one alone shows nothing. The 2×2 that proves
   it was already in `teacher_forced.json`; we had never arranged it this way.
   §1.1.
2. **`role_swapped` was never capable of answering the layout question** and
   should not have been a gating control: it changes four marker tokens while
   leaving ~250 reference audio tokens and the speaker head untouched. The
   decisive version costs one line of `VARIANT_SPEC` — rerun the layout controls
   with the speaker head *off*. §1.2, §1.5.
3. **There is a third conditioning route we never measured, and the literature
   says it is the one that produces speaker similarity.** `s3gen` is conditioned
   on a CAM++ x-vector *and* the reference mel (`s3gen.py:3921`, `:4032-4034`).
   CosyVoice 2 — same codec, same decoder lineage — deleted its LM speaker
   embedding and speaker similarity did not move. §1.3.
4. **20-30 s is the wrong product spec.** Our decoder hard-truncates the
   reference to **10 s** (`vocoder.py:30`) and the distilled decoder to **4 s**;
   F5-TTS independently caps at 12 s. Published length curves flatten by 10 s,
   and a benchmark that swept 30 s → 30 minutes found extra reference buys
   **prosody, not timbre**. **Ask for 15 s and spend the effort on prompt
   *quality*** — a noisy-reverberant prompt costs F5-TTS 0.273 SECS. §3, §7.1.
5. **Our evaluation has a bigger bug than the one D8 was written to prevent.**
   Whether the prompt is still attached to the scored audio is worth **0.151
   absolute SIM**; scorer choice is worth up to **0.095** with rank inversions;
   the circularity we built D8 around has been measured twice at about **0.008**.
   D8's decision survives — for entirely different reasons. §6.2, §6.6.
6. **Our conditioning encoder is the weakest model in its field** — pyannote at
   2.8% VoxCeleb1-O against 0.23-0.43% for the current options, and gated. But
   swapping it is a **retrain**, not an upgrade, because `spk_proj` reproduces
   its output to max abs difference 0. §4.
7. **The cheapest win in the whole note is not a conditioning change.** F5-TTS
   counts text length in UTF-8 bytes, which inflates Indic scripts
   catastrophically; a character-level count took a bilingual model's Bengali
   from **108% to ~24% WER with no retraining.** Audit every duration and length
   estimate we have. §5.6, R8.

## 1. The open gate: why the reference prefix showed almost nothing

### 1.1 The 2×2 was already in our data

Gate 2 reported two identity controls against two different baselines and never
put the four cells in one table. Here they are, from
`indic/data/teacher_forced.json` (8 English FLEURS examples, dev split, bf16,
reference cosine 0.65 — the run whose full control set survives on disk;
findings-doc §3.7 quotes a separate 16-example run whose `speaker_zeros` row was
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

One more row supports reading this as *identity* rather than as a generic input
perturbation. `speaker_noise` feeds a random **unit-norm** vector where
`speaker_zeros` feeds a zero vector (`teacher_forced_loss.py:speaker_vector`), and
an actively wrong speaker costs **more** than no speaker at all: +0.065 against
+0.037 nats over `correct`. A model that merely disliked an unusual input at
position 0 would be hurt most by the zero vector, which is the one furthest from
the unit-norm manifold the encoder actually produces. It is hurt least by it.

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

> **So the honest answer to the gate question is: yes, and the reason is
> redundancy, not blindness.** A reference prefix shows nothing measurable under
> teacher forcing *when a second identity route is left intact* (our +0.019 nats,
> t = 0.48); it shows a large, unanimous effect when it is the only route (+0.236
> nats, t = 7.80); and removing it at generation time is the largest single
> effect in the zero-shot TTS ablation literature (−0.349 SPK in VALL-E, −0.298
> SIM in Phoenix TTS, with WER flat or barely moved).
>
> Teacher-forced loss is therefore a **valid but heavily attenuated** proxy for
> identity, and the attenuation is *conditional* — it collapses to nothing
> exactly when another route covers for it, which is the configuration our gate
> ran in. It is fine as a wiring check. It is not fine as a design metric, and no
> amount of it will rank two reference layouts unless the other route is off.

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
at it as the remaining alternative. CosyVoice's in-context format merges the
reference transcript and the target text into *one* text block and treats the
reference speech tokens as *already-generated* output — there is no `speech_end` between reference and
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
>
> `indic/bench/reference_budget.py` — which already sweeps
> `--prefix-seconds × --decoder-seconds` and monkeypatches `DECODER_COND_SECONDS`
> to reach past 10 s — has **three** versions of this problem today, and all
> three are cheap to fix before it is first run:
> 1. it scores against the *reference clip itself*, whose length is the swept
>    variable → score against a fixed held-out clip of the same speaker instead;
> 2. it groups FLEURS clips into "speakers" with `WavLMSpeakerEmbedder` and then
>    **scores with the same model**, so the reference is selected to be easy for
>    the scorer → cluster with a different encoder than the one that scores;
> 3. it **concatenates** clips to reach 30 s, and its own comment notes "a seam
>    inside a reference is a discontinuity the speaker encoder reads as a second
>    voice" — so the long-reference arms carry more seams than the short ones,
>    confounding exactly the variable under test.

### 3.4 Does prompt quality matter more than prompt length? Yes, and not narrowly

[DAIEN-TTS / arXiv:2608.03011](https://arxiv.org/html/2608.03011v1), Table I,
Seed-TTS test-en, F5-TTS as the in-context baseline, with the condition's own
ground-truth ceiling alongside:

| prompt | WER % | **SECS** | that condition's GT SECS | deficit vs its own GT | FAD |
|---|---|---|---|---|---|
| clean | 2.30 | **0.590** | 0.734 | 0.144 | — |
| noise only | 2.58 | 0.506 | 0.699 | **0.193** | 3.83 |
| reverb only | 2.68 | 0.391 | 0.605 | **0.214** | 7.11 |
| noise + reverb | 2.80 | **0.317** | 0.734 | 0.417 | 8.56 |

**−0.273 SECS raw from room noise and reverberation**, which is ~9× the real
3 s → 10 s length effect and larger than every length effect in §3.3 combined.

Two honest qualifications on that number, both of which matter for how we build
the product:

- **Reverb is the harder condition, but not by the margin the raw column
  suggests.** Against each condition's own ground-truth ceiling the deficits are
  0.193 (noise) and 0.214 (reverb) — reverb worse by ~0.02, not by the ~2.4×
  implied by comparing raw SECS to the clean baseline. **The paper does not make
  this comparison; both readings are derived.** The ordering survives, the
  magnitude does not.
- **WER barely moves** (2.30 → 2.80) while SECS collapses. **Do not gate prompt
  quality on intelligibility** — an unusable prompt will transcribe fine.

The mechanism is stated plainly: as an in-context system, F5-TTS *"tends to
preserve the reverberation from the speaker prompt"* — in-context methods *"tend
to preserve the environmental characteristics of the audio prompt."* FireRedTTS
says the same of its utterance-level encoder: it captures "speaker identity,
speaking style, **and the acoustic environment**". On real in-the-wild recordings
(TITW) F5-TTS reaches only 0.481 SECS at 12.42% WER.

**And denoising is not a free fix — it has a measurable crossover.** FireRedTTS
Table 6, 40 unseen clean prompts with noise added at DNS-Challenge SNRs:

| prompt SNR | SMOS without → with enhancement | SIM % without → with |
|---|---|---|
| 20 dB | 3.97 → **3.89** | 50.66 → **49.20 (−1.46)** |
| 10 dB | 3.37 → 3.53 | 41.81 → 43.60 (+1.79) |
| 0 dB | 2.62 → **3.01** | 28.81 → **35.15 (+6.34)** |

Their conclusion: enhancement "should be selectively applied only to low-SNR
audio prompts." **The crossover sits between 10 and 20 dB — gate denoising on an
estimated prompt SNR, never run it unconditionally.** Against that, leaving a
0 dB prompt alone costs ~0.19 SECS
([arXiv:2505.13830](https://arxiv.org/html/2505.13830): clean 0.827 → noisy
0.642 → token-level denoising 0.827, full recovery, at 1.10 GFLOPs against
MP-SENet's 38.93).

**The better fix is model-side.** [arXiv:2406.05699](https://arxiv.org/html/2406.05699v1)
(Microsoft, Interspeech 2024) compared noise-augmented training + DNSMOS data
filtering against enhancement-on-the-prompt: the augmented model reaches noisy
SIM-o **0.65** / DNSMOS 3.32 *and* clean SIM-o **0.75**, against the baseline's
0.60 / 3.01 / 0.71 and SE-on-prompt's 0.60 / 3.28. **Strictly better on every
axis.** Data filtering alone is worth +0.054 noisy SIM-o, and the pretrain noise
probability optimum is 0.50. For us that is a data-augmentation recipe and a
late-stage decoder fine-tune, not a new architecture — see R7 in §8.1.

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


---

## 4. Speaker encoders

### 4.1 The incumbent is the weakest model on the list

`kittenml/kittentts2/speaker_embedding.py` is a vendored reimplementation of
[`pyannote/embedding`](https://huggingface.co/pyannote/embedding) — SincNet
frontend, TDNN stack, statistics pooling, 512-d output, MIT. Measured against the
field it is **2.8% EER on VoxCeleb1-O**, where the serious 2025-2026 options are
0.23-0.43%. That is a **6-12× gap**, and it is also **gated on HuggingFace**
(`gated: "auto"` — auto-approval, but it still needs an accepted terms page and a
token in CI, which the RUNBOOK's gated-repo trap already warns about).

### 4.2 The table

Scoring configuration differs per row and is stated, because AS-Norm/QMF numbers
are not reachable from a plain cosine scorer. "O"/"H" are VoxCeleb1-O-clean and
VoxCeleb1-H-clean.

| model | Vox1-O EER | Vox1-H | CN-Celeb EER | params | dim | license | gated | URL |
|---|---|---|---|---|---|---|---|---|
| **pyannote/embedding** *(our conditioner)* | **2.8** | — | — | ~4.5M (**UNVERIFIED**) | **512** | MIT | **yes** | [hf](https://huggingface.co/pyannote/embedding) |
| pyannote/wespeaker-resnet34-LM | 0.797 | 1.695 | — | 6.63M | 256 | CC-BY-4.0 | no | [hf](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) |
| **WavLM-Large + ECAPA** (UniSpeech `wavlm_large_finetune.pth`) | **0.431** | 1.154 | — | ~320M | 256 | **CC-BY-SA-3.0** ⚠ | no (mirror) | [github](https://github.com/microsoft/UniSpeech/tree/main/downstreams/speaker_verification) |
| `microsoft/wavlm-base-plus-sv` *(our current scorer)* | **not published** | — | — | ~94M | not stated | **no license field on HF**; card defers to UniSpeech | no | [hf](https://huggingface.co/microsoft/wavlm-base-plus-sv) |
| **ReDimNet2-B0** | 1.04 | 1.97 | — | 1.1M (0.33 GMACs) | 192 † | **MIT** | no | [github](https://github.com/PalabraAI/redimnet2) |
| **ReDimNet2-B2** | **0.57** | 1.41 | — | 3.6M (0.95 GMACs) | 192 † | **MIT** | no | ″ |
| **ReDimNet2-B3** | 0.42 | 1.22 | — | 4.1M (2.70 GMACs) | 192 † | MIT | no | ″ |
| **ReDimNet2-B3 (+cnc2)** | 0.51 | 1.14 | **4.78** | 4.1M | 192 † | MIT | no | ″ |
| **ReDimNet2-B6 (vb2+vox2)** | **0.23** | 0.67 | **7.49** | 12.3M (13.05 GMACs) | 192 † | MIT | no | ″ |
| **ReDimNet2-B6 (+cnc2)** | 0.26 | 0.79 | **3.59** | 12.3M | 192 † | MIT | no | ″ |
| CAM++ (3D-Speaker) *(what s3gen uses)* | 0.65 | 1.58 | 6.30 | 7.18M (1.15 GMACs) | **192** | Apache-2.0 | no | [github](https://github.com/modelscope/3D-Speaker) |
| SpeechBrain ECAPA-TDNN | 0.80 | — | 7.67 | ~20.7M (**UNVERIFIED**) | 192 | Apache-2.0 | no | [hf](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) |
| WeSpeaker ResNet293-LM | 0.532 | 1.311 | — | 28.62M (28.10 GMACs) | 256 | CC-BY-4.0 | no | [hf](https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM) |
| NeMo TitaNet-Large | 0.66 | — | — | ~23M | 192 (**UNVERIFIED**) | CC-BY-4.0 | no | [hf](https://huggingface.co/nvidia/speakerverification_en_titanet_large) |
| SimAM-ResNet100 (VoxBlink2) | **0.229** | 0.868 | — | 50.2M | 256 | CC-BY-4.0 per policy | **HF repo is empty** | [wespeaker](https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md) |
| ECAPA2 | — | — | — | — | — | **CC-BY-NC-4.0 — DISQUALIFIED** | no | [hf](https://huggingface.co/Jenthe/ECAPA2) |
| w2v-BERT 2.0 + MFA (2026 claim) | **0.12** | 0.55 | — | ~600M | — | **NOASSERTION — all rights reserved** | — | [arXiv](https://arxiv.org/abs/2510.04213) |
| Resemblyzer / GE2E | **none published** | — | — | — | 256 | Apache-2.0 | no | [github](https://github.com/resemble-ai/Resemblyzer) |

Repo licenses were read from the LICENSE files, not from HF tags. Two of the
load-bearing ones were re-verified directly for this note: `microsoft/UniSpeech`'s
[LICENSE](https://raw.githubusercontent.com/microsoft/UniSpeech/main/LICENSE)
opens *"Attribution-ShareAlike 3.0 Unported"*, and
[PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2) is MIT with the
EER table above reproduced from its own model zoo. † **ReDimNet2's embedding
dimension is UNVERIFIED** — the repo publishes only `(batch, emb_dim)`; 192 is
carried over from ReDimNet v1 and must be read off the checkpoint before
`spk_proj` is resized.

### 4.3 Three licence landmines

1. **The field-standard scorer is CC-BY-SA-3.0, not MIT.** `microsoft/UniSpeech`'s
   LICENSE file opens "Attribution-ShareAlike 3.0 Unported". That is the
   WavLM-Large-SV checkpoint Seed-TTS-eval, F5-TTS, CosyVoice and ZipVoice all
   report SIM against. Using it in an **offline eval harness** distributes
   nothing and almost certainly never triggers ShareAlike; **shipping it inside
   the product** plausibly does. The `k2-fsa/TTS_eval_models` mirror is tagged
   `apache-2.0`, which is the mirror's tag and does not override Microsoft's.
   **Get counsel on this before it reaches the product repo.**
2. **ECAPA2 is CC-BY-NC-4.0.** It appears as a strong baseline in the ReDimNet2
   paper, so somebody will suggest it. Hard no.
3. **VoxCeleb's own terms could not be retrieved.** The Oxford VGG page served no
   licence statement. Every downstream CC-BY-4.0 claim (pyannote, WeSpeaker,
   NVIDIA) rests on an assertion about the VoxCeleb licence that we could not
   independently verify. **UNVERIFIED, and our whole commercial filter leans on
   it.** MIT-licensed ReDimNet2 is the one family that does not depend on this
   chain.

### 4.4 Do VoxCeleb-trained encoders generalise to Indian speakers?

**The honest answer is that nobody has published the measurement, and the
indirect evidence says "worse than you would hope, and the ranking changes".**

**(a) Indic unseen-speaker verification is an order of magnitude harder.**
IndicSUPERB / Kathbath ([ar5iv:2208.11761](https://ar5iv.labs.arxiv.org/html/2208.11761),
AAAI 2023), ASV EER %, SUPERB protocol (frozen SSL + small head — *not*
comparable to the §4.2 table, but internally consistent):

| | FBANK | IndicWav2Vec | XLS-R |
|---|---|---|---|
| test, **known** speakers (12-lang avg) | 3.0 | 2.1 | 2.1 |
| test, **unknown** speakers (12-lang avg) | **11.2** | **15.3** | **12.1** |
| Hindi, unknown | 9.0 | 8.8 | 6.3 |
| Bengali, unknown | 13.3 | 16.6 | 10.7 |
| Marathi, unknown | 8.8 | 12.8 | 10.8 |
| **Tamil, unknown** | **14.4** | **15.3** | **15.7** |
| Telugu, unknown | 11.6 | 15.4 | 11.9 |

Unknown-speaker is exactly our zero-shot cloning regime. **Tamil is the worst
language here, as it was in our own tokenizer fertility and teacher-forced
perplexity measurements** — the third independent measurement to say so.

Note also that the *bigger* SSL model is *worse* on unseen speakers (IndicW2V
15.3 vs FBANK 11.2), while XLS-R — trained on far more speakers — generalises
better than IndicW2V. **Speaker diversity of the encoder's training data beats
architecture**, which is the single most decision-relevant signal in this section.

**(b) Language + sensor mismatch costs ~4× EER on Indian speech.**
I-MSV 2022 ([arXiv:2302.13209](https://arxiv.org/html/2302.13209v1)), 13 Indian
languages, 100 speakers, 5 sensors:

| baseline | matched sensor & language | overall |
|---|---|---|
| i-vector | 4.61 | 13.72 |
| x-vector | **2.40** | **9.32** |

**3.9× degradation.** Language and sensor are confounded there, so it is an upper
bound on the language term alone. Fine-tuning fixes it well — NPU-HC reached
0.289-0.862% after fine-tuning on Indian data from VoxCeleb pretraining
([arXiv:2211.16694](https://arxiv.org/html/2211.16694v2)) — but **they do not
report the pre-fine-tuning zero-shot EER**, which is the number we actually need.

**(c) The cleanest quantified mismatch evidence is Chinese, not Indic, and it
points at a cheap fix.** Same ReDimNet2 checkpoint family, two test sets
([PalabraAI/redimnet2](https://github.com/PalabraAI/redimnet2)):

| checkpoint | Vox1-O | CN-Celeb | ratio |
|---|---|---|---|
| B6, vb2+vox2 | 0.23 | **7.49** | 32.6× |
| B6, vb2+vox2 **+cnc2** | 0.26 | **3.59** | 13.8× |
| B3, vb2+vox2 **+cnc2** | 0.51 | 4.78 | 9.4× |

**Adding in-language data to the encoder's training set halves the mismatched-
language EER (7.49 → 3.59) at a cost of 0.03 points in-domain.** No Indic-trained
equivalent exists publicly. That is a gap somebody could close in a week with
IndicVoices and a WeSpeaker recipe, and it would be worth more than any
architecture change.

**(d) VoxCeleb ranking does not survive the language shift.** 3D-Speaker
([arXiv:2403.19971](https://arxiv.org/html/2403.19971v2)), same models across
domains:

| model | Vox1-O | CN-Celeb | 3D-Speaker multi-dialect |
|---|---|---|---|
| CAM++ | 0.65 | 6.30 | 11.78 |
| ERes2NetV2 | 0.61 | 6.04 | 11.30 |
| **ResNet34** | **1.05** | **6.92** | 12.81 |
| **ECAPA-TDNN** | **0.86** | **7.67** | 12.24 |

ECAPA beats ResNet34 on VoxCeleb and **loses to it on CN-Celeb**. Picking by
Vox1-O alone is picking on a benchmark that re-orders under exactly the shift we
care about.

**(e) What does not help.** DISPLACE 2023/2024
([arXiv:2406.09494](https://arxiv.org/html/2406.09494v1)) has **no speaker
verification track and reports no EER** — diarization only (baseline DER 25.45%
dev / 29.96% eval; best team 21.27%). Indic DiarBench
([arXiv:2607.23808](https://arxiv.org/html/2607.23808v1), 22 languages, 2026)
benchmarks **commercial ASR APIs only** and explicitly excludes pyannote; useful
as evidence that Indic speaker handling is broadly poor, useless for choosing an
encoder. VoxBlink2 ([arXiv:2407.11510](https://arxiv.org/html/2407.11510v1)) has
111k speakers across "18 languages" but **never names any Indian language and
publishes no per-language breakdown — UNVERIFIED coverage.** Resemblyzer's own
README says it "currently works best on English language only".

> **There is no public 2024-2026 benchmark of modern speaker encoders on
> Indian-language ASV reporting EER.** Four separate arXiv metadata sweeps found
> only I-MSV 2022 and IndicSUPERB 2022, both predating every model in §4.2. **No
> further literature search will answer this; we have to measure it.**

### 4.5 The conditioning encoder and the scorer should not be the same model

Our D8 said so already. **The decision is confirmed, but the stated reason is not
citable — and three better reasons are.**

We searched for a paper asserting that scoring with the conditioning encoder is
circular and **found none**. Treat the circularity argument as our team's
position, soundly reasoned: a trained `spk_proj` is explicitly optimised to make
the conditioning encoder's embedding of the output match its embedding of the
reference, so scoring with that encoder measures how well `spk_proj` did its job.
That reasoning stands; it just has no citation behind it. The three things that
*do* have citations are stronger anyway:

1. **Scorer choice changes the conclusion.** TTSDS2
   ([arXiv:2506.19441](https://arxiv.org/html/2506.19441v2), Table 3) runs three
   speaker encoders over the same systems and the same human ratings; Spearman ρ
   against human SMOS ranges **0.47 to 0.82** depending on encoder and domain.
   The authors also note X-Vector shows clustering that "could indicate
   overfitting to specific systems" but cannot diagnose it because the encoders
   are "essentially black boxes". **That is the closest published statement to our
   concern, and it is a hedge, not a finding.**
2. **ASV embeddings have confounds a TTS pipeline systematically introduces.**
   [arXiv:2507.02176](https://arxiv.org/html/2507.02176v1) probes encoders where a
   same-speaker control should give EER ≈ 0.50:

   | probe | EER (should be 0.50) |
   |---|---|
   | same-speaker control | ~0.50 |
   | **short vs long utterance, same speaker** | **0.30-0.39** |
   | SNR 0 dB | 0.01-0.10 |
   | emphasis / de-emphasis filter | 0.07 / 0.01 |
   | after re-equalisation | back to ~0.50 |

   **Duration alone makes the same speaker look like a different person**, and so
   does any spectral tilt — which a vocoder always imposes. Two concrete rules
   for our harness follow in §6.
3. **The lowest-EER encoder is not the best conditioner.**
   [arXiv:2506.20190](https://arxiv.org/abs/2506.20190) (TSD 2025) swaps three
   encoders into the same YourTTS framework: the original H/ASP encoder beat both
   ECAPA-TDNN and x-vectors on a listening test *and* on objective cosine
   distance, with ECAPA beating x-vectors. Their conclusion: ECAPA's dominance in
   speaker recognition "doesn't guarantee better speaker similarity in zero-shot
   TTS". **So §4.2's EER column is a weak prior for the conditioning choice and a
   strong one for the scoring choice.**

### 4.6 Recommendations

**For SCORING — report two, not one.**

1. **WavLM-Large + ECAPA (UniSpeech `wavlm_large_finetune.pth`), 256-d, 0.431.**
   Not because of its EER but because of **comparability**: every competitive
   zero-shot TTS paper since Seed-TTS reports SIM against this exact checkpoint,
   and a number scored with anything else cannot be read against theirs. Maximally
   independent of pyannote (different pretraining, objective and family).
   **Conditions:** offline eval only — see the CC-BY-SA-3.0 landmine above; pull
   the ungated mirror from `k2-fsa/TTS_eval_models`; budget GPU, it is ~320M.
2. **ReDimNet2-B6 (vb2+vox2+cnc2), MIT, 192-d, 0.26 / CN-Celeb 3.59.** The best
   score we can use commercially with **zero conditions**, 25× smaller, and the
   only checkpoint in the table with direct evidence of having been hardened
   against language mismatch. **If we run one scorer and care more about Indic
   truth than paper-comparability, run this one.**

Given TTSDS2's 0.47-0.82 ρ spread, a single scorer is a single point of failure,
and **the two disagreeing is itself the signal**.

> **Action on our current default.** `indic/eval/metrics.py` uses
> `microsoft/wavlm-base-plus-sv`, whose docstring claims it is "the x-vector head
> the speaker-verification literature (and the SECS numbers in the XTTS/F5
> papers) is reported against". **That claim is wrong.** F5-TTS and the Seed-TTS
> line report against WavLM-**large** fine-tuned, from UniSpeech — a different
> model. `wavlm-base-plus-sv` also publishes **no VoxCeleb EER and has no licence
> field on HuggingFace at all.** It is not a bad diagnostic, but it is not the
> convention and our numbers will not be comparable to published SIM.

**For CONDITIONING — rank by training-data speaker diversity, not by EER.**

1. **ReDimNet2-B2 or B3 (vox2, LM), MIT, 192-d, 0.57 / 0.42, 0.95 / 2.70 GMACs.**
   Best cost-quality point in the table, MIT with no conditions, compact enough to
   sit inside the training loop. Changes `spk_proj`'s input dim **512 → 192**.
2. **ReDimNet2-B3 (+cnc2), MIT, 192-d, 0.51 / CN-Celeb 4.78.** Trades 0.09 points
   of in-domain EER for materially better language-mismatch behaviour. On the
   evidence in §4.4(a) and (c) — training-data coverage beats architecture — this
   is arguably the right default; promote it above #1 if an Indic trial list
   confirms the CN-Celeb pattern transfers.
3. **CAM++ (WeSpeaker/3D-Speaker), Apache-2.0, 192-d, 1.15 GMACs.** The safest
   incremental step, with official ONNX — **and note that s3gen already runs
   exactly this model internally** (`s3gen.py:3921`), so adopting it for route 1
   would mean both LM and decoder read identity through the same encoder. That is
   a *reduction* in independence and an argument against it here.
4. **Keep pyannote/embedding as the control arm.** `spk_proj` was trained against
   its 512-d output and reproduces it to max abs difference 0 (findings doc §3.8).
   **Swapping the encoder invalidates the trained head** — this is not a free
   upgrade, it is a retrain. Do not retire pyannote until a replacement wins on a
   *listening test*, which is precisely the substitution
   [arXiv:2506.20190](https://arxiv.org/abs/2506.20190) showed can backfire. If we
   only want out of the gating dependency, `pyannote/wespeaker-voxceleb-resnet34-LM`
   is ungated, CC-BY-4.0, 256-d, 0.797.
5. **Not WavLM-Large-SV as conditioner** — 320M in the training loop,
   CC-BY-SA-3.0 contaminating a shipped artifact, and it destroys scoring
   independence. **Not Resemblyzer** — English-only by its own README, no
   published EER.

**The hard rule:** the conditioning encoder and the primary scorer must not be
the same model *or the same family*. If we adopt ReDimNet2 for conditioning, the
primary scorer must be WavLM-Large-SV, not a different ReDimNet2 size.

### 4.7 The one experiment worth more than this whole section

**Build an Indic ASV trial list and measure it ourselves.** Kathbath/IndicSUPERB
([github](https://github.com/AI4Bharat/IndicSUPERB)) already ships known/unknown
speaker ASV protocols per language. Run pyannote/embedding, ReDimNet2-B2,
ReDimNet2-B3-cnc2, CAM++ and WavLM-Large-SV over Hindi, Bengali, Tamil, Telugu
and Marathi, unknown-speaker protocol, and report EER per language. **That
produces the five numbers this section could not find anywhere in the literature,
costs about a day, and will probably reorder the rankings above.** Everything in
§4.6 is inference from English and Chinese benchmarks.

---

## 5. Cross-lingual cloning — English reference, Indic output

This is the headline product use case and it is the **least benchmarked thing in
this entire note.** No Indic TTS system publishes a quantified English-prompt →
Hindi/Tamil cloning number. IndicF5's "Polyglot" setting is Indic → Indic and
reports no polyglot speaker similarity at all. **We would be creating this
benchmark, not consuming one.**

### 5.1 What degrades, with numbers

**The cleanest same-vs-cross comparison** is XTTS at Interspeech 2024
([arXiv:2406.04904](https://arxiv.org/html/2406.04904v1), Tables 2 & 4). All
references are English DAPS speakers, so the English row is same-language and
everything else is cross-lingual:

| | CER | SECS |
|---|---|---|
| English (same-language) | 0.543 | **0.642** |
| 15-language cross-lingual average | 2.065 | **0.505** |

**−0.138 SECS (−21% relative) and 3.8× the CER.** Note XTTS's paper covers 16
languages, **none Indic**; Hindi arrived later in the v2 weights.

CosyVoice 3 ([arXiv:2505.17589](https://arxiv.org/html/2505.17589v2)) says the
same at a different scale: en→zh WER 8.48 / SS 67.4 and zh→en 4.99 / 67.8,
against intra-lingual zh CER 3.89 and en WER 5.24 — **cross-lingual roughly
doubles the error rate.** VALL-E X
([arXiv:2303.03926](https://arxiv.org/html/2303.03926v1)) is asymmetric: En TTS
with Zh prompts reaches ASV 0.36 / WER 4.07, while Zh TTS with En prompts falls
to ASV **0.29** / WER 8.52.

### 5.2 Hindi is the worst cell in the one study that tested it

Google's zero-shot cross-lingual voice transfer
([arXiv:2409.13910](https://arxiv.org/html/2409.13910v1), Table I;
[demo table](https://google.github.io/tacotron/publications/zero_shot_voice_transfer/index.html)),
English reference → 9 target languages, human-rated same-speaker agreement %:

| language | VAE | SharedGST | MultiGST | SegmentGST |
|---|---|---|---|---|
| reference ceiling | 85 ±6 | 85 | 85 | 85 |
| English | 68 ±4 | 48 | 58 | 64 |
| Arabic | 90 ±7 | 90 | 74 | 90 |
| French | 88 ±5 | 86 | 75 | 87 |
| Japanese | 77 ±5 | 68 | 70 | 73 |
| **Hindi** | **57 ±6** | **38 ±5** | **42 ±6** | **47 ±6** |
| mean | 76% | 67% | 66% | 73% |

**Hindi is the lowest cell in the table under every method**, and the paper never
explains it. Naturalness MOS for Hindi is *fine* (3.9-4.1, at or above the 3.3
reference) — **the damage is entirely identity, not quality.** No other Indic
language was tested, and this is a GST/VAE-era system, not a codec LM, so
**UNVERIFIED whether the deficit transfers to our architecture.** It is still the
only published signal we have, and it points at our largest market.

### 5.3 In-context prompting is the wrong route cross-lingually

This is the most decisive architectural result in this section, and it cuts
against §2.4's ranking. MiniMax-Speech
([arXiv:2505.07916](https://arxiv.org/html/2505.07916v1), Table 3), Chinese voice
prompt → other languages; "zero-shot" = speaker encoder only, "one-shot" = plus
an in-context text+audio prompt:

| target | WER ZS | WER OS | SIM ZS | SIM OS |
|---|---|---|---|---|
| Finnish | 4.527 | 8.112 | 0.554 | 0.655 |
| Czech | 2.823 | 5.096 | 0.605 | 0.648 |
| Romanian | 3.081 | 5.353 | 0.625 | 0.690 |
| French | 4.497 | 5.489 | 0.586 | 0.645 |
| Thai | 2.826 | 4.107 | 0.729 | 0.748 |
| Arabic | 1.446 | 2.649 | 0.619 | 0.632 |
| Vietnamese | 0.659 | 1.788 | 0.692 | 0.725 |

**In-context prompting buys ~+0.033 SIM and costs ~+64% relative WER.** The
authors state flatly that text-speech prompt methods "struggle when the reference
language differs from the target language." CosyVoice's own `cross_lingual` mode
agrees by construction: it **omits the prompt text and prompt tokens entirely**
so the prompt's prosody does not carry into the target language
([arXiv:2407.05407](https://arxiv.org/html/2407.05407v2)).

**For us this is cheap**, because `prompt.py` already supports it:
`build_generation_prompt(reference_prefix=None)` is exactly the cross-lingual
configuration, with `spk_proj` and s3gen's CAM++/`prompt_feat` left carrying
identity. It is one branch, not a redesign — and it is the main reason R3's
independent dropout is not optional.

### 5.4 Accent leakage: cause, and fixes ranked by measured effect

**Cause, and it is consensus:** speaker embeddings implicitly encode accent and
language. Accent-VITS argues a single gradient-reversal classifier cannot fully
separate them because accent lives in prosody and pronunciation as well as
timbre. RADTTS adds the condition that makes it ours: accent and language are
"easily learned from non-parallel data" **when each speaker has only one accent
in training** — which is exactly what a user enrolment is. CosyVoice 2's reason
for deleting its speaker embedding (§1.3) is the same observation.

| # | fix | measured effect | cost | source |
|---|---|---|---|---|
| 1 | **Conditional LayerNorm on an accent embedding** | accent sim W-CS **0.196 → 0.490** | CER 0.154 → 0.253 (+64% rel) | Joycent T7 |
| 2 | **Dual-level language-ID injection (time *and* text)** | cross-lingual WER en→zh **6.03 → 1.87** (−69%), it→zh 7.37→2.93, zh→it 6.65→3.44 | intra-lingual *also* improves 5.46→4.94 | X-Voice T8 |
| 3 | **Language-ID conditioning (VALL-E X)** | accent MOS **2.35 → 4.03** (En→Zh); a *wrong* LID is worse than none | ASV **0.49 → 0.48** — identity pays | [2303.03926](https://arxiv.org/html/2303.03926v1) T6 |
| 4 | **Self-distillation speech factorization** | SIM **0.491 → 0.753** (EN), 0.636 → 0.791 (ZH) | WER *improves* 2.366 → 2.121 | Seed-TTS T6 |
| 5 | **GRL on a language/accent classifier** | AccLeak **0.284 → 0.203** (−29% rel) vs XTTS-v2 | SpkSim ~unchanged (0.832 → 0.842) | CrossAccent-TTS T1 |
| 6 | **VQ information bottleneck** | timbre leakage S-SIM(src) **0.306 → 0.109** as K goes 16384 → 8 | WER **6.81 → 21.81** — brutal | Vevo T2 |
| 7 | **Accent Analogy Guidance** (training-free sampler term) | ΔSIM **+0.11 to +0.27** over the CFG hull at equal accent | none at inference | [2609.29123](https://arxiv.org/html/2609.29123v1) T2 |

**The one to implement first is #2, and the ablation is the reason.** X-Voice
shows **text-level FiLM alone does essentially nothing** cross-lingually
(en→zh 6.03 → 6.05); the whole −69% comes from injecting language ID at the
**time / diffusion-step level**. Conditioning on language in the text stream,
which is the obvious thing to do, is the thing that does not work.

### 5.5 Identity and nativeness are a Pareto frontier — pick a point deliberately

This is the finding that should go in the product spec, not just the model spec.
AAG on held-out Korean dubbing ([arXiv:2609.29123](https://arxiv.org/html/2609.29123v1),
Table 2), with closed-set speaker ID over 41 speakers (chance 2.4%):

| setting | accent (1-5) | SIM | WER % | UTMOS | closed-set spk ID |
|---|---|---|---|---|---|
| default | 2.54 | **0.651** | 5.7 | 3.25 | — |
| tuned | 3.51 | 0.429 | 4.1 | 3.73 | — |
| **equal-accent CFG reweighting** | 4.22 | **0.034** | 4.9 | 4.08 | **4%** |
| **AAG** | **4.28** | **0.294** | 5.1 | 3.72 | **69%** |

**Accent leakage and speaker identity are the same signal.** Pushing nativeness
from 2.54 to 4.28 costs SIM 0.651 → 0.294 *with the best known method*, and naive
CFG reweighting at the same accent level **destroys identity outright** — 4%
speaker ID against a 2.4% chance floor. There is no setting where we get a native
Hindi accent and full English-speaker identity. **We have to choose, and the
choice should be a product decision, not an accident of a hyperparameter.**

Two architecture warnings from the same paper, both load-bearing for us:

- **F5-TTS cannot reach a native accent at all** — its hull tops out at accent
  **1.8/5**, and pushing past that costs WER 10.4 → 26.4 and UTMOS 3.11 → 2.50.
  **IndicF5 is built on F5-TTS**, and so is the only published Indic
  cross-lingual benchmark (§5.6).
- **X-Voice scores premise 0.00** — its accent "barely depends on the reference's
  language", so AAG gains nothing on it. Its architectural language conditioning
  had already removed the leakage. **The strongest evidence available that
  conditioning beats post-hoc correction.**

### 5.6 Phoneme coverage: the failure is duration, not phonetics

**Cloning from English does not break Indic phonemes at the ASR-measurable
level.** The benchmark is IIT Madras, October 2026
([arXiv:2610.14826](https://arxiv.org/html/2610.14826)) — **English reference
audio → Indic generation**, F5-TTS Base multilingual, 3,248 h:

| language | GT WER / CER | no duration fix: WER / SIM / CER | + duration heuristic: WER / SIM / CER |
|---|---|---|---|
| Hindi | 14.40 / 5.21 | 26.70 / 0.666 / 17.05 | **11.66 / 0.680 / 3.42** |
| Bengali | 24.78 / 6.48 | 46.78 / 0.668 / 21.57 | **26.60 / 0.688 / 7.07** |
| Tamil | 25.39 / 4.87 | 50.95 / 0.600 / 24.02 | **20.39 / 0.646 / 3.52** |
| Telugu | 21.06 / 3.07 | 37.66 / 0.635 / 18.02 | **16.40 / 0.662 / 1.93** |
| Marathi | 10.56 / 2.08 | 30.76 / 0.637 / 15.44 | **10.74 / 0.661 / 2.00** |

Three things to take from it:

1. **An inference-time duration fix is the highest-leverage, lowest-cost
   intervention in this whole note.** F5-TTS counts text length in **UTF-8
   bytes**, which massively inflates Indic scripts; switching to character-level
   counting plus per-language duration-per-character statistics takes the
   bilingual small model's Bengali from **108.12% WER to ~24%**, with **no
   retraining at all.** Check whether our own chunker or any duration estimate
   counts bytes — this is a one-line class of bug with an enormous effect.
2. **After the fix, English-prompted Indic output lands at or below the
   ground-truth ASR CER floor** for Hindi (3.42 vs 5.21), Tamil (3.52 vs 4.87),
   Telugu (1.93 vs 3.07) and Marathi (2.00 vs 2.08). Only Bengali is worse. This
   is the same floor-relative discipline as our D9.
3. **The cost is identity: SIM 0.600-0.699.** ⚠ **The paper has no
   same-language-reference baseline**, so this cannot be turned into a
   cross-lingual delta. Building that baseline is ours to do.

The **inverse direction** — Indic reference → English output — is *easier*: CER
0.92-1.89 against a 1.33 GT floor, three of five *below* the floor, and **SIM is
higher (0.639-0.725)**. Accent leakage is audible but costs intelligibility
nothing.

**Per-phoneme evidence exists and is weak.** The PSP probe
([arXiv:2604.25476](https://arxiv.org/html/2604.25476)) reports Hindi retroflex
and aspiration collapse at **0.0-4.5%** across ElevenLabs v3, Cartesia Sonic-3,
Sarvam Bulbul, Indic Parler-TTS and Chatterbox — effectively solved. Telugu
retroflex 26.7-50.0% looks alarming until you see **the native-human noise floor
on that probe is 43%**, which makes the absolute number meaningless. Tamil
retroflex 64.3-70.5% against a ~47% floor is a real ~20 pp signal, and **Tamil
ழ /ɻ/ collapses at 85.7% for ElevenLabs v3, Cartesia Sonic-3 and Sarvam Bulbul
alike** (best is Indic Parler-TTS at 61.5%). ⚠ **Provenance warning: PSP and its
siblings are single-author, mutually self-citing preprints with n=10 pilot sets
and an explicit admission that "differences of 5 percentage points are not
statistically separable." Hypothesis-generating only.** If the Tamil zha number
survives replication it is a differentiator nobody has claimed; validate the
probe's noise floor on native human audio first.

**The mechanism matters more than the numbers:** two independent ablations in
that line show a 10× text-head data scale-up and a Hindi LoRA moved WER by 5× and
13× respectively while **per-phoneme retroflex collapse did not move at all**.
Corollary: **the speaker prompt does not determine phonetic coverage; the
training inventory does.** Our phoneme risk is a data question, not a
conditioning question.

**Code-switching dumps far more error than any phoneme.** Raw English spans
inside Indic text are silently dropped — intent preservation 0-10%, output 35-45%
shorter than the input predicts — and a deterministic transliteration
preprocessor recovers 64-82% relative WER. **It is detectable from the audio
length statistic alone**, which makes it a cheap guard to add. (Same
single-author provenance caveat.)

Hindi schwa deletion is **not** our bottleneck: 98.00% per-schwa / 97.78%
word-level with XGBoost ([ACL 2020, arXiv:2004.10353](https://arxiv.org/html/2004.10353v2)),
with residual errors concentrated in multi-morpheme compounds and loanwords —
i.e. brand names.

### 5.7 The most on-point paper in existence, and it is almost our project

**CrossAccent-TTS** ([arXiv:2606.25403](https://arxiv.org/html/2606.25403v1),
Sony Research India, June 2026). **986 h of training data — 636 h in-house Indic
plus 350 h Emilia-YODAS — on Hindi, Telugu, Tamil, Bengali, Marathi and English.
That is our exact language set at almost exactly our data budget.**

| model | UTMOS ↑ | AccLeak ↓ | AccSim ↑ | SpkSim ↑ |
|---|---|---|---|---|
| IndicF5 | 2.817 | 0.312 | 0.312 | 0.843 |
| XTTS-v2 | 3.168 | 0.284 | 0.284 | 0.832 |
| **proposed** | **3.181** | **0.203** | **0.371** | 0.842 |

Architecture, which is strikingly close to a design we could build: Neucodec FSQ
at 50 tok/s → Perceiver Resampler (32 slots, d=768) → **Qwen2.5-0.5B** decoder
over IPA text, a learnable language-embedding table added to every latent slot,
and GRL on an accent/language classifier at λ = 0.1. Accent intensity is a
continuous dial, `λ·e_lang1 + (1−λ)·e_lang2`.

⚠ **Caveats that keep it from being a recipe:** no ablation (so the GRL's
isolated contribution is unmeasured), no WER or CER at all, no code or weights
released, and **the paper never states the reference language per evaluation** —
so whether this is English → Indic is **UNVERIFIED**. IndicF5 and XTTS-v2 also
share identical AccLeak/AccSim values, which looks like a table error.

### 5.8 Indic systems and licences — the hard filter

**The independent quality result first.** AI4Bharat's SpeechArena
([arXiv:2604.21481](https://arxiv.org/html/2604.21481v2)) — **120K+ pairwise
comparisons, 1,900+ native raters, 10 Indic languages** — Bradley-Terry:

| rank | system | win rate |
|---|---|---|
| 1 | Gemini 2.5 Pro TTS | **70%** |
| 2 | ElevenLabs v3 | 57% |
| 2 | Cartesia Sonic 3 | 56% |
| 4 | Sarvam Bulbul v3 Beta | 52% |
| 5 | MiniMax Speech 2.8 HD | 47% |
| 6 | GPT-4o-mini TTS | 40% |
| 7 | **IndicF5 — the only open model** | **19%** |

SHAP over the ratings says **expressiveness and intelligibility are the strongest
predictors of preference**, with noise and hallucinations contributing least.
**This flatly contradicts IndicF5's self-reported numbers** (MUSHRA 80.5 against
a human 91.8; S-SIM 97.3 *exceeding* the human reference's 97.0 — a saturated
metric). **Treat IndicF5's paper as a recipe and the arena as the quality claim.**

| system | licence | commercial? | Indic | cloning |
|---|---|---|---|---|
| **IndicF5 / IN-F5** | MIT (gated; card bars unauthorised cloning) | **yes** | 11 langs, 1,417 h | one-shot, needs transcript |
| **SPRINGLab Indic-Mio** | Apache-2.0 | **yes** | 22 + English, 44.1 kHz, RTF<0.1 | zero-shot |
| **ARTPARK DhVaani-0.5** | Apache-2.0 | **yes** | 27 langs, 123M ZipVoice | zero-shot |
| Indic Parler-TTS | Apache-2.0 | yes | 21 langs, 1,806 h | **fixed speakers only** |
| kenpath/svara-tts-v1 | Apache-2.0 but **Llama-3.2 upstream** | ⚠ check | 19 langs | unclear |
| Sarvam Bulbul v3/v4 | API only, **ToS bars training on output** | ⚠ hostile | 11 langs | enterprise |
| **XTTS-v2** | **CPML — non-commercial only** | **no** | **Hindi (17 langs)** | yes |
| F5-TTS / MaskGCT / Llasa | CC-BY-NC-4.0 | no | no | yes |
| Spark-TTS | CC-BY-NC-SA-4.0 (**downgraded from Apache over training data**) | no | no | yes |
| **CosyVoice 3 (Fun-CosyVoice3-0.5B)** | **Apache-2.0** | **yes** | **no Indic** | yes |

Datasets: IndicVoices-R **CC BY 4.0** (1,704 h / 22 langs / 10,496 speakers),
IndicVoices CC BY 4.0, SYSPIN CC BY 4.0 (920 h, but reported audio distortion and
text-audio mismatches — budget filtering), **Emilia-YODAS CC BY 4.0** (the
commercially usable Emilia split, and it has no Indic).

> **Two licence risks to resolve before training, both flagged as UNVERIFIED.**
> **(1) Rasa.** The HF tag, the paper and AI4Bharat's own Parler table all say
> **CC BY 4.0**; the government AIKosh catalogue says **CC BY-ND 4.0**, which
> would forbid releasing a model trained on it. Three-to-one in our favour and
> probably a cataloguing error — but Rasa feeds IndicF5, Indic-Mio, DhVaani and
> svara-TTS. Get it in writing.
> **(2) The IIT Madras IndicTTS licence PDF** (`iitm.ac.in/donlab/indictts/downloads/license.pdf`)
> **timed out on four fetch attempts.** It is the root licence under four open
> Indic models. Someone has to read it.
> **(3) Sarvam's ToS** bars using Bulbul output "to develop, train, test,
> fine-tune, or otherwise improve any machine learning or AI system". **Keep
> Bulbul output air-gapped from our pipeline, including as a scored baseline that
> could feed back** — the same rule the findings doc's §7 already imposes on
> KittenTTS output under the Stellon licence.

**IndicF5's training recipe is the most actionable table for a <1000 h budget**
("Phir Hera Fairy", [arXiv:2505.20693](https://arxiv.org/html/2505.20693v1)):

| fine-tuning strategy | overall MUSHRA |
|---|---|
| **EN→IN (English checkpoint, Indic data only)** | **73.4** |
| EN→EN+IN (mix English back in) | 66.2 |
| from scratch | 43.2 |

**Fine-tuning an English checkpoint on Indic data *alone* beat mixing English
back in.** That is a direct challenge to findings-doc §3.1's corollary that
"English recordings must stay in every training phase" — which was argued from
tokenizer fertility, a different axis. **Worth treating as a live question rather
than settled in either direction.** Their data-scaling table adds: **S-SIM is
flat from 1 h to 100 h per language (92.7 → 93.1)** while MUSHRA moves 33.7 →
64.3 and WER 59.4% → 32.6%. **Spend hours on naturalness and WER; speaker
similarity is not where data buys you anything.** And zero-shot speaker
generalisation needs the right corpus: fine-tuning on IndicTTS gave S-SIM 78.93,
on IndicVoices-R **88.18**.

### 5.9 One result that argues our users are the easy case

LASE ([arXiv:2605.00777](https://arxiv.org/abs/2605.00777)) measures identity
loss for the same voice across scripts:

| reference accent | WavLM-base-plus-sv | ECAPA-TDNN |
|---|---|---|
| Western-accented (n=1043) | 0.082 | 0.105 |
| **Indian-accented (n=1369)** | **0.006** | 0.044 |

**An Indian-accented English reference loses ~14× less identity than a
Western-accented one under WavLM** — and our headline use case is precisely an
Indian user recording in English. It also, incidentally, defends
`wavlm-base-plus-sv` on this one axis (0.006 against ECAPA's 0.044). ⚠ **Single-
author preprint, synthetic pairs from 8 commercial voices, ~1,100 training pairs.
Cheap to replicate; do not plan on it unreplicated.**

---

## 6. Measuring it honestly

### 6.1 The convention, pinned exactly

The Seed-TTS-eval SIM checkpoint is the Google Drive id
`1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP`
([seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval)), which is
**byte-identical to the entry in microsoft/UniSpeech's speaker-verification table
for "WavLM large, Fix pre-train: No"** — the best row there, Vox1-O / E / H =
0.431 / 0.538 / 1.154. Our D8 guessed right; the link is now hard rather than
inferred.

**It is not a WavLM model.** It is an ECAPA-TDNN head over WavLM-large features,
and both `UniSpeech/.../verification.py` and F5-TTS's `utils_eval.py` load it the
same way:

```python
model = ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large", config_path=None)
model.load_state_dict(torch.load(ckpt)["model"], strict=False)   # <- strict=False
sim = F.cosine_similarity(emb1, emb2)                             # 16 kHz mono
```

> **`strict=False` is a silent-failure trap of exactly the kind the findings doc
> catalogues.** A key-name mismatch loads a randomly initialised head and returns
> plausible-looking numbers. **Assert on a known same-speaker / different-speaker
> pair at startup**, the same way `selfcheck.py` asserts the speaker injection.

Ungated non-Drive mirror: [`bezzam/wavlm_large_finetune_seed_tts_eval`](https://huggingface.co/bezzam/wavlm_large_finetune_seed_tts_eval/tree/main)
(1.3 GB, cc-by-sa-3.0, commit message records the upstream sha256). **See §4.3 on
the CC-BY-SA-3.0 landmine: offline eval only.**

**SIM-o vs SIM-r refers to the *reference* side, not the generated side.**
Voicebox, which coined them: *"we advocate for computing similarity against the
original audio context, which we call SIM-orig (SIM-o)"*; SIM-r compares against
the prompt after a vocoder round trip, and *"SIM-resyn is not comparable across
models using different vocoders."* **Report SIM-o only.**

### 6.2 The protocol detail worth 0.151 — fix this before anything else

**The reference is the *prompt*, not the ground-truth recording.** Verified in two
independent codebases: seed-tts-eval's `get_wav_res_ref_text.py` parses the
ground-truth column and never uses it (the line that would swap it in is
commented out), and F5-TTS's `run_sim` is called with `(gen_wav, prompt_wav)`.

And whether the prompt is still *attached* to the scored audio changes the number
enormously. [arXiv:2510.06927](https://arxiv.org/abs/2510.06927) §4.1 measured
the two live conventions on LibriSpeech test-clean continuation:

| protocol | SIM-o |
|---|---|
| VALL-E: prompt **excluded** from the scored audio | **0.754** |
| VALL-E 2: full synthesis **including** the prompt | **0.905** |

**0.151 absolute, same model, nothing changed but what reaches the scorer.** For
scale, the entire circularity effect we built D8 around is ~0.008 (§6.5) and the
entire scorer-choice effect is ~0.095 (§6.3). **We were guarding the smallest
door.** Audit our harness for prompt inclusion first, and state the protocol —
reference = prompt audio, prompt excluded, reference duration, sample rate — in
every table we publish.

### 6.3 A 0.6 from one scorer is not a 0.6 from another, and ranks invert

**Floors differ enormously.** UniSpeech's own README runs the same two VoxCeleb
pairs through two models:

| | different-speaker pair | same-speaker pair |
|---|---|---|
| unispeech_sat | **0.0317** | 0.5389 |
| ecapa_tdnn | **0.2053** | 0.5302 |

Same ceiling, floors 0.17 apart — completely different usable dynamic range.
**Resemblyzer-era SECS numbers (~0.82-0.86) and WavLM-SV SIM numbers (~0.65-0.80)
are not on the same scale and must never be compared.**

**And the two scorers disagree on ordering, not just on scale.** CosyVoice 3
([arXiv:2505.17589](https://arxiv.org/html/2505.17589v2), Table 4) reports WavLM
SIM with ERes2Net in parentheses:

| system | test-en WavLM | test-en ERes2Net | Δ | test-zh WavLM | test-zh ERes2Net | Δ |
|---|---|---|---|---|---|---|
| Human | 0.734 | 0.742 | +0.008 | 0.755 | 0.775 | +0.020 |
| F5-TTS | 0.647 | 0.742 | **+0.095** | 0.741 | 0.794 | +0.053 |
| CosyVoice 2 | 0.652 | 0.736 | +0.084 | **0.748** | **0.806** | +0.058 |
| CosyVoice 3-1.5B | 0.720 | 0.789 | +0.069 | 0.781 | 0.837 | +0.056 |
| **MaskGCT** | 0.714 | 0.730 | +0.016 | **0.774** | **0.752** | **−0.022** |
| FireRedTTS | 0.460 | 0.526 | +0.066 | 0.635 | 0.653 | +0.018 |

The offset ranges from −0.022 to +0.095. On test-zh, WavLM ranks MaskGCT (0.774)
**above** CosyVoice 2 (0.748); ERes2Net **reverses it** (0.752 vs 0.806).
CosyVoice 2's own paper: *"the results of speaker similarity are not consistent
on different SV models."* And within one scorer, F5-TTS reports its own test-en
SIM-o as 0.67 while CosyVoice 3's rerun gives 0.647 — **treat ±0.02 as noise even
with the scorer and protocol fixed.**

### 6.4 What counts as a good score

| anchor | WavLM-large SV SIM | note |
|---|---|---|
| random different speakers | ~0.03-0.05 **UNVERIFIED for this checkpoint** | 0.0317 measured for unispeech_sat; the floor is scorer-specific |
| failing clone | < 0.50 | FireRedTTS 0.460 — audibly the wrong speaker |
| mediocre | 0.55-0.65 | Spark-TTS 0.584, Llasa-8B 0.581 |
| competitive open source | 0.65-0.72 | F5-TTS 0.67, MaskGCT 0.714, IndexTTS2 0.709 |
| **ground-truth ceiling, test-en** | **0.730-0.734** | a *real recording* of the same speaker, different session |
| **ground-truth ceiling, test-zh** | **0.750-0.755** | |
| above the ceiling | 0.76-0.81 | Seed-TTS, MiniMax, CosyVoice 3 — **saturated** |

Two things to internalise.

**(a) The ceiling is a cross-session score and our model is not playing the same
game.** "Human 0.730" means a genuine recording of that speaker *from a different
session* against the prompt. A TTS system is scored generated-against-prompt —
same session, same channel, same noise floor. **That is mechanically why systems
beat the "human" row**, and it is not evidence they sound more like the speaker.

**(b) Above ~0.73 the metric has stopped measuring identity.** Seed-TTS scores
SIM 0.762 against human 0.730 while its **CMOS is −0.07** — humans rate it
slightly *worse* than the recording it beats. F5-TTS on test-zh ties ground truth
at 0.76 with SMOS 3.83 against 3.72, while E2-TTS scores *higher* SIM (0.730) and
*lower* SMOS (3.44) than F5-TTS at 16 NFE. The ordering disagrees in both
directions. I2D ([arXiv:2603.24430](https://arxiv.org/abs/2603.24430)) measures
the spread across 11 SOTA systems at **sd = 3.83 points, falling to 1.51 once one
weak outlier is removed.**

> **Target: reach 0.65-0.70 and then stop optimising SIM and start listening.**
> Anything above ~0.73 is a flag to investigate — prompt leakage? channel
> copying? — not a win.

### 6.5 The pitfall is real, and better evidenced than expected

**(a) The SIM-human correlation vanishes on hard pairs.** Liu, Babel & Zhu,
Interspeech 2024 ([ISCA](https://www.isca-archive.org/interspeech_2024/liu24q_interspeech.pdf)),
530 listeners, 7 WeSpeaker models. With a by-talker-pair random intercept,
verification score against similarity rating is **β = 0.00**, and against an AX
same/different decision **β = 0.00 at 51.8% — chance.** Their words: *"when
controlled for talker pairs, the correlation between verification similarities
and perceptual similarities disappears"*; it *"only exists at a gross-phonetic
level."* **The correlation everyone cites comes from easy pairs, not the regime
that matters.**

**(b) Utterance-level SIM is near-useless; system-level is fine.**

| study | SIM vs human, utterance level | system level |
|---|---|---|
| I2D 2026 | SRCC **0.32** | 0.68 |
| CodecMOS-Accent | Pearson **0.39** | 0.86 |
| SVSNet (purpose-built) | LCC 0.574 max | 0.965 |

**Never make a per-clip judgement from SIM.** Report system-level means with
confidence intervals.

**(c) The metric can be maxed by audibly broken audio.** ASVspoof 5
([arXiv:2502.08857](https://arxiv.org/abs/2502.08857)): bona-fide ASV EER 5.22%,
pooled over attacks 32.11%, and **attack A19 drives it to 49.80% — literal
chance** — while the organisers note *"notable concatenation artefacts"* in A12
and A19, and A12's predicted-MOS EER is 1.82% (trivially distinguishable from
real speech). **The attacks that most completely satisfy the verifier are the
ones that sound most broken.**

**(d) The direct metric-gaming datum.** DMOSpeech's own ablation
([arXiv:2410.11097](https://arxiv.org/abs/2410.11097), Table 4): the
**SV-loss-only** variant scores **SIM 0.70, higher than the full model**, at
**WER 6.62 against 1.94**. Authors: *"while SV loss can enhance speaker
similarity, it negatively impacts intelligibility and naturalness."*

**(e) The cleanest illustration is YourTTS.** All seven VCTK configurations beat
ground-truth SECS (0.843-0.864 against GT 0.824) while all seven score *below*
ground truth on human Sim-MOS. The authors' own explanation is a **recording
artefact**: *"significant breathing sounds in most audios. The speaker encoder
may not be able to handle these features, hereby lowering the SECS of the ground
truth."* (Their Sim-MOS confidence intervals overlap, so the human ranking is not
itself significant.)

**And yes, it rewards the channel.** §4.5's duration/SNR/EQ probe is the
mechanism, and since the reference *is the prompt* (§6.2), copying the prompt's
room and noise floor is rewarded. ⚠ **UNVERIFIED: no published study isolates
"TTS copies prompt noise → SIM rises."** The mechanism is strongly implied by
§6.2 plus §4.5 plus §3.4, but that specific claim is an experiment to run, not a
citation to cite.

### 6.6 Our no-circular-scorer decision: **CONFIRM the decision, CHALLENGE the reasoning**

**Keep the separate scorer. The justification in D8 does not survive contact with
the evidence.**

**Evidence *for* circularity is thinner than we assumed.** Exactly one paper
states it, informally and without measurement —
[arXiv:2408.15916](https://arxiv.org/abs/2408.15916) §3.3: *"We use a different
separate speaker encoder for evaluation because we have noticed that speaker
encoders tend to give higher scores when the acoustic model is trained on the
same speaker encoder."* The dedicated 2025 survey on TTS evaluation validity
([arXiv:2510.06927](https://arxiv.org/abs/2510.06927)) **does not mention the
issue at all.**

**Evidence *against* is two controlled experiments, and both say the effect is
small:**

1. [arXiv:2409.12403](https://arxiv.org/abs/2409.12403) runs DPO against a RawNet
   speaker reward and scores on both the seen and an unseen encoder: **+0.032 on
   RawNet (seen) against +0.024 on ECAPA-TDNN (unseen)** — ~75% transfers, so
   matched-encoder inflation is about **0.008 absolute**.
2. TLA-SA ([arXiv:2511.09995](https://arxiv.org/abs/2511.09995), Table 4) trains
   a speaker-alignment loss against three supervisors and scores with two.
   Training against WavLM-SV produced the **worst** Sim-WavLM score of the three
   (0.538 against 0.544 for both ERes2Net and CAM++ supervision). **The matched
   cell is the one that should have been inflated and it was not.**

**And the field's apparent consensus is an architectural artefact, not a
methodological choice.** Of 14 flagship zero-shot systems, **nine have no speaker
encoder in the conditioning path at all** (Voicebox, NaturalSpeech 2, Seed-TTS,
F5-TTS, E2-TTS, MaskGCT, VALL-E 2, Llasa, CosyVoice 2's LM) — they condition
in-context on raw audio, so there is nothing to reuse. YourTTS's stated reason
for Resemblyzer is **comparability**, not circularity: *"Following previous
works, we compute SECS using the speaker encoder of the Resemblyzer package,
allowing for comparison with those studies."* MiniMax-Speech's reason is also
pure convention.

**Scale check: 0.008 (circularity) against 0.095 (encoder choice) against 0.151
(prompt inclusion).**

**The real hazard is somewhere else, and it is live in the field.** Circularity
closes through **rewards, verifiers and rerankers**, not through conditioning.
Seed-TTS uses *"the SIM and WER objective metrics as rewards"* with SIM =
WavLM-large SV, evaluates on the same metrics, and admits *"we observed reward
hacking."* Llasa does best-of-N where *"the similarity verifier is a
WavLM-finetuned speaker verification model"* and then reports SIM from
*"WavLM-large fine-tuned on the speaker verification task"* — an explicit argmax
over the reported metric, unacknowledged. The grounding principle is Kynkäänniemi
et al. on FID ([arXiv:2203.06026](https://arxiv.org/abs/2203.06026)): *"FID can be
manipulated to a great extent... Optimizing FID directly in this way is a form of
adversarial attack."* That is about optimising a distance **in the metric's
feature space** — which rewards and rerankers do and plain conditioning does not.

> **Verdict: D8 stands, for four better reasons.**
> 1. **Comparability.** WavLM-large SV is the field's lingua franca; scoring with
>    our pyannote x-vector makes our numbers incomparable with everything
>    published. This is by far the strongest reason.
> 2. **Encoder variance is real and large** — up to 0.095, with rank inversions.
>    A single in-house encoder gives no robustness read.
> 3. **Our projection head is *trained*,** which puts us closer to the TLA-SA
>    setup (training *against* an encoder) than to pure in-context conditioning —
>    which is where the measured inflation, small as it is, actually lives.
> 4. **Option value.** Separation costs nothing and preserves the ability to add
>    a reranker later.
>
> **And write down the rule that matters more than the conditioning/evaluation
> split: if anything in our pipeline ever optimises, reranks or selects against a
> speaker-embedding score, that encoder is permanently disqualified as an
> evaluator.** That is the rule the field keeps breaking.
>
> Keep reporting the pyannote similarity — as a clearly labelled training
> diagnostic, exactly as `KittenSpeakerEmbedder`'s docstring already says, and
> never as the headline.

### 6.7 The spec we should run

**Speaker similarity**
- Primary: `wavlm_large_finetune.pth` via the `bezzam` mirror, loaded as
  `ECAPA_TDNN_SMALL(feat_dim=1024, feat_type="wavlm_large")`, 16 kHz mono,
  `F.cosine_similarity`. **Assert a known pair at startup** (`strict=False`).
- Second scorer, mandatory: **ReDimNet2-B6+cnc2 (MIT, §4.6)** or a VoxCeleb-
  trained ECAPA. ⚠ Note the `iic/speech_eres2net*` and `iic/speech_campplus*`
  checkpoints are **Mandarin-trained (zh-cn, 200k speakers)** — questionable for
  Indic.
- **SIM-o only.** State the protocol in every table: reference = prompt audio,
  **prompt excluded from the scored audio**, reference duration, sample rate.
- **System-level mean + 95% CI. Never a per-clip verdict.**
- Apply the §4.5 controls — **match utterance durations**, run SNR assessment on
  both genuine and synthesised audio, and **re-equalise** to the genuine spectral
  balance ([code](https://github.com/ubisoft/ubisoft-laforge-spkrid)). **Each of
  these is worth more than the circularity effect.**
- **Always include our own ground-truth-against-prompt row in the same table.**
  Without that ceiling the SIM number means nothing — and the ceiling is
  dataset-dependent (0.69 against 0.754 on nominally the same LibriSpeech).

**Intelligibility** — this part our D9 already got right.
- English / Indian English: `openai/whisper-large-v3`. **Not `large-v3-turbo`**:
  the decoder is pruned 32 → 4 layers and the card warns accuracy "is uneven
  across languages… lower for low-resource languages."
- Indic: `ai4bharat/indic-conformer-600m-multilingual`, chosen for comparability
  with the October 2026 IIT-M benchmark (§5.6).
- **CER as the headline with the floor printed beside it**, which is D9. Measured
  floors on real speech ([A2TTS, arXiv:2507.15272](https://arxiv.org/abs/2507.15272)):
  **Hindi WER 8.27, Bengali 9.35, Tamil 20.69, Malayalam 29.22.** A raw "WER 22%"
  is meaningless against a 12% floor.

**Naturalness — the weakest metrics in the stack, handle accordingly.** TTSDS2
measured 16 metrics against 11,000+ human ratings; Spearman ρ against MOS across
Clean / Noisy / Wild / Kids: **UTMOS 0.51 / 0.47 / −0.12 / −0.02**, UTMOSv2
0.39 / 0.34 / 0.16 / 0.05, and UTMOSv2 against **SMOS 0.09 / 0.19 / −0.04 /
−0.02.** WER against MOS is **negative throughout** (−0.19 to −0.45). So: carry
UTMOS22-strong for paper comparability, **SCOREQ** (`data_domain='synthetic'`) as
the better ranker, DNSMOS P.835 OVRL purely as an artifact tripwire (<3.0 =
investigate), and **report all three as Δ against our own ground-truth recordings
in the same language and recording chain** — Kannada gold speech scores UTMOS
3.461 against Indian-English gold 4.248 *from the same speakers*, a ~0.8 MOS
penalty for the language alone. **TTSDS2** itself (`pip install ttsds`,
`multilingual=True`) is the only metric with ρ > 0.5 against humans in every
domain (avg 0.67), and its Speaker / Prosody / Intelligibility split says *what*
broke. ⚠ **UNVERIFIED whether TTSDS2's 14 languages include any Indian language —
assume not until checked.**

**Human evaluation — the only thing that settles it.**
- One language, several systems: **MUSHRA-DG-NMR**
  ([arXiv:2411.12719](https://arxiv.org/abs/2411.12719), TMLR, 492 listeners,
  Hindi + Tamil). **30 listeners × 30 utterances** for ρ > 0.95; **more listeners
  beats more utterances**; degraded-TTS anchor, not a lowpass anchor.
- Many systems, many languages: pairwise + Bradley-Terry, as SpeechArena (§5.8).
- **Before claiming human parity, run the Human Fooling Rate**
  ([arXiv:2508.04179](https://arxiv.org/abs/2508.04179), Interspeech 2025).
  Binary human-vs-TTS forced choice; the human reference itself tops out at
  **74.11%**. The killer datum: **XTTS scored MUSHRA 76.58 on Expresso — above
  the human reference's 74.78 — and was identified as machine 58% of the time.**

### 6.8 There is no Indic TTS evaluation convention. This is the honest answer.

No IndicMOS, no Indic TTS leaderboard, no Indic arena. **TTS Arena V2 is
English-only** — the only Indian entries (Veena, Maya 1) sit near the bottom and
are judged by English listeners on English prompts. Three mutually incompatible
automatic stacks are in use:

| stack | used by | note |
|---|---|---|
| **UTMOS + WavLM-large SIM + IndicConformer CER/WER** | [arXiv:2610.14826](https://arxiv.org/html/2610.14826) (IIT-M, Oct 2026, 9 languages) | **Closest to the global convention — adopt this one.** Test set and recipes released |
| NORESQA-MOS + in-house wav2vec2 SIM | AI4Bharat IndicVoices-R | Two unaddressed circularity problems: the non-matching-reference pool is **English (LibriTTS)**, and the speaker encoder is **fine-tuned on the very data being evaluated**. The Noresqa repo was archived 2025-07-17 |
| UnitSpeech Sim-O + Conformer CER | A2TTS | Sim-O 0.705-0.734 across 7 languages against real speech 0.725-0.772 |

**Nobody has validated WavLM-SV or UTMOS against Indian-language human ratings.**
The corpus that could do it exists: **`ai4bharat/MANGO`** (CC BY 4.0, ~255,150
ratings from 492 listeners — Hindi 127,500, Tamil 118,500), built for MUSHRA
methodology, with no MOS-predictor correlations reported.

> **Correlating UTMOS, SCOREQ and our SIM against MANGO is about a day of work,
> nobody has done it, and it is the only way to honestly defend any automatic
> number we publish in Hindi or Tamil.** Alongside §4.7's Indic ASV trial list,
> it is the highest-value evaluation work available to this project.

One warning that lands directly on us: LASE (§5.9) measured an **0.082-0.105
cross-*script* identity gap** for WavLM and ECAPA on Indic data — **comparable in
magnitude to the entire encoder-choice effect.** Report our cross-script delta
explicitly or we will mistake encoder bias for model error.

---

## 7. Prosody and style transfer versus identity

### 7.1 Does cloning carry speaking rate and emotion as well as timbre?

**Emotion and recording channel: yes, strongly. Speaking rate: only partially,
and no system closes the gap.**

LP-TTS-Bench ([arXiv:2610.19774](https://arxiv.org/html/2610.19774), Oct 2026)
is the first benchmark to separate these axes properly. It defines
**SRD = |log(rate_syn / rate_gt)|** alongside Pause Distance, Prominence Contour
Correlation, Intonation Correlation, Pitch Range Distance and a subjective P-MOS:

| system | SIM-O (zh/en) | **SRD** (zh/en) | PRD (zh/en) | P-MOS (zh/en) |
|---|---|---|---|---|
| Qwen3TTS-Base | 0.56 / 0.77 | 0.155 / 0.124 | 5.017 / 3.668 | 2.94 / 3.13 |
| LPTTS | 0.57 / 0.78 | 0.137 / 0.112 | 4.532 / 3.279 | 3.53 / 3.53 |

**SRD of 0.112-0.155 is a 12-17% speaking-rate mismatch that no system
eliminates**, and the authors note it "stays above zero for every system". So
timbre clones well and rate clones badly — they are not one signal.

**The result that matters most for our product spec** comes from the same paper:
sweeping prompt length from **30 s to 30 minutes**, prosody metrics improve and
saturate around **10 minutes, while SIM-O stays roughly stable throughout.**

> **Beyond roughly 10-30 s, extra reference audio buys prosody, not timbre.**

That is the clean statement of §3 from the other direction, and it is consistent
with VALL-E's 3/5/10 s curve, Voicebox's flat-to-10 s, and RVCBench's MCD
dropping sharply to ~8-12 s and then flattening. **UNVERIFIED in detail:** the
LP-TTS-Bench and RVCBench curves are published as figures only, so the saturation
points are read off plots, not tables.

It also tells us what a longer reference would be *for*. If we ever want to clone
a speaker's *habits* — their characteristic rate, their pause structure — that is
a minutes-scale ask, not a 30-second one, and it is a different product.

### 7.2 Timbre and prosody can be separated, and here is what it costs

| method | what it buys | what it costs | source |
|---|---|---|---|
| **Seed-TTS self-distillation** | SIM **0.491 → 0.753** (EN), 0.636 → 0.791 (ZH) | WER *improves* 2.366 → 2.121 | [2406.02430](https://arxiv.org/html/2406.02430v1) T6 |
| **Vevo VQ bottleneck** (K 16384 → 8) | timbre leakage S-SIM(src) 0.306 → **0.109** | WER **6.81 → 21.81** | [2502.07243](https://arxiv.org/html/2502.07243v1) T2 |
| **IndexTTS2** (GRL + separate Emo/Spk Perceivers) | SS 0.870 / WER 3.115 on LibriSpeech; duration control to ≤0.067% token error at 0.75-1.25× | — | [2506.21619](https://arxiv.org/html/2506.21619v1) |
| **MegaTTS 3** multi-condition CFG | accent intensity as a continuous dial (α_spk vs α_txt) | no CFG at all: SIM-O 0.71 → 0.43, WER 1.82 → 6.85 | MegaTTS 3 |

**Seed-TTS's +0.262 SIM at no WER cost is the largest clean disentanglement gain
in the literature** and is the one worth studying if we ever have the budget.

⚠ Two caveats on the disentanglement papers, both relevant because they are the
ones people will cite at us: **IndexTTS2 has no ablation of its GRL or its
emotion conditioner at all, and publishes no speaker-similarity number for the
case where the emotion prompt comes from a different speaker** — the exact
question. Its own ablation shows removing the GPT latent *raises* SS 0.870 →
0.887 while worsening WER, which is the identity/intelligibility tradeoff again.
**DisCo-Speech** ([2512.13251](https://arxiv.org/html/2512.13251v2)) likewise has
no ablation table and no same-vs-cross-speaker breakdown. Vevo is **English-only
with no cross-lingual results**, and its paper is CC BY-NC-SA.

### 7.3 Is carrying prosody desirable? The evidence leans "no, make it a dial"

**For disentangling — and the strongest statements come from systems that tried
the other way:**

- **Seed-TTS states the downside outright.** Evaluators noted output "maintained
  consistent prosody defined by the reference", which "improves similarity but
  **slightly reduces prosodic variation in long-form speech**". For speakers with
  strong accents or exaggerated styles, **fine-tuned traditional TTS beat
  zero-shot in-context cloning**, which they attribute to a 15 s prompt that "may
  omit the speaker's representative prosody". Zero-shot in-context emotion
  accuracy is poor and uneven: **Angry 0.46, Happy 0.44, Sad 0.53, Surprise
  0.13.**
- **MiniMax-Speech found emotionally congruent references actively harmful** —
  output emotion became "overly reliant on the reference's emotion", and
  **neutral or random-emotion references worked better.** They frame their
  zero-shot encoder's advantage as not being "constrained by the prosody of a
  specific text-audio prompt".
- **The cross-lingual case settles it for us.** §5.3: prompt-based conditioning
  "struggles when the reference language differs from the target language", and
  cross-language prompts drag prosody toward the reference's language. When the
  output language differs from the reference's, prosody-carrying is *actively
  wrong*.

**For carrying it:** in-context prompting does genuinely win on similarity —
MiniMax +0.033 SIM, Confucius4-TTS +0.015 English SIM (0.700 → 0.715) — but at
+64% and +13% relative WER respectively. LP-TTS-Bench's position is that short
prompts give "only a local realization of a speaker's rate and pauses, not
recurring habits", which is an argument that short-prompt prosody transfer is
*noise* rather than a feature.

**Direct user-preference evidence on prosody-matching specifically: we found
none.** The nearest proxy is AI4Bharat's SHAP result that **expressiveness is the
top predictor of overall preference** (§5.8) — which argues for *good* prosody,
not for *copied* prosody. **Mark this as a genuine gap in the literature.**

> **Our position: make it a dial, not a commitment.** The machinery is cheap and
> already exists in the field (IndexTTS2 ≤0.067% token error across 0.75-1.25×;
> Vevo2 97-98% duration-control accuracy), and §5.5 has already established that
> the identity/nativeness tradeoff must be a deliberate product choice. Adding
> rate and expressiveness to the same control surface costs little and avoids
> hard-wiring the wrong default for a user whose enrolment happened to be read
> slowly off a script.

### 7.4 The recording channel is the part of "style" we definitely do not want

§3.4 has the numbers; the point to carry here is that **in-context conditioning
transfers the room as readily as the voice**, and that this is a *conditioning*
property rather than a data-quality accident. F5-TTS's output FAD against the
true noisy-reverberant target is **8.56** (ground truth 0.10) — it neither
removes the environment nor reproduces it faithfully, emitting a degraded hybrid.
VoXtream2 states the mechanism: *"As the model more closely mimics the target
voice, the synthesized speech quality converges to that of the acoustic prompt."*
Seed-TTS reports the opposite tendency — it "tends to reduce the background noise
during the generation process" — so this is architecture-dependent, not
universal.

**Our route 3 is the exposed one.** `s3gen` takes the reference *mel* as
`prompt_feat` and prepends it to the generated mel (`s3gen.py:3879`). A mel
carries the room. The LM's codec prefix carries it too, but less directly. **This
is one more reason the SNR-selected window in R5 is worth building: it is the
only place in our pipeline where prompt cleanliness can be enforced once and
benefit all three routes.**

---

## 8. The recommended conditioning design

### 8.1 The design

**Keep both LM routes but stop letting them cover for each other; give the
decoder the best ten seconds rather than the first ten; switch routes by
language; and add nothing new.**

```
  user records 15-30 s
         |
         +-- full clip -----------> speaker x-vector --> spk_proj --> LM position 0
         |                          (route 1; 2.1 MFLOP, always on)
         |
         +-- best contiguous -----> S3 codec tokens ---> LM reference prefix
         |   10-12 s window,        (route 2; 25 tok/s, ~30 MiB KV, cached per
         |   selected by SNR         speaker, DROPPED when ref lang != target lang)
         |
         +-- same window ---------> CAM++ 192-d + reference mel --> s3gen prompt_feat
                                    (route 3; where the timbre is actually rendered,
                                     hard-capped at 10 s by vocoder.py:30)
```

**R1 — Route 2 (in-context prefix) is the primary identity route *within* a
language.** It adds zero parameters, which is the property that matters most at
<1000 h (§2.4), and removing it at generation time is the largest effect in the
ablation literature: −0.349 SPK (VALL-E), −0.298 SIM (Phoenix TTS).

**R2 — Keep route 1 (`spk_proj`), and give it a *learned null vector* rather than
zeros.** Position 0's input embedding is *replaced*, not added to
(`speaker.py:_inject_speaker`), and the encoder always emits a unit-norm vector,
so a zero vector is out of distribution by construction. One trainable 512-d null
embedding, used whenever the speaker route is dropped, keeps every training and
inference sequence in distribution. (Today `speaker_zeros` costs 0.037 nats over
`correct`, part of which is the OOD input rather than the missing identity.)

**R3 — Drop the two LM routes *independently* during training.** This is the
direct fix for the failure that made our gate unreadable, and it is what makes
R4 legal at inference:

```
p = 0.30   reference prefix dropped, speaker vector kept     <- the cross-lingual mode
p = 0.30   prefix kept, speaker vector replaced by the null embedding
p = 0.10   both dropped
p = 0.30   both present                                      <- the same-language mode
```

Without it the two routes stay ~90% redundant (§1.1) and neither is individually
reliable. **UNVERIFIED:** these probabilities are a starting point, not a measured
optimum.

**R4 — When the reference language differs from the target language, drop the
reference prefix and run on route 1 + route 3 alone.** This is the single
clearest architectural result in §5.3: in-context prompting buys ~+0.033 SIM
cross-lingually and costs **~+64% relative WER** (MiniMax-Speech Table 3), and
the authors say prompt-based methods "struggle when the reference language
differs from the target language". CosyVoice's own `cross_lingual` mode omits the
prompt text and tokens for exactly this reason. **`prompt.py` already supports
it** — `build_generation_prompt(reference_prefix=None)` *is* this mode — so it is
a branch on a language comparison, not a redesign. It is also the reason R3
cannot be skipped: the model has to have been trained to work without the prefix.

**R5 — Spend the 20-30 s on *selection*, not on length.** Choose the best
contiguous 10-12 s window by SNR and give *that* window to routes 2 and 3,
instead of `wav24[:10*24000]`. Keep the full clip for route 1, which is free and
does benefit from more audio. §3.4: prompt quality is worth several times what
prompt length is worth, and §7.4: route 3 takes the reference *mel*, so it takes
the room with it.

**R6 — Cap the LM prefix at 12 s.** Past that it is KV cache spent on the route
§1.1 shows is 87% redundant with a 2.1 MFLOP projection, and F5-TTS (`<12s`) and
our own decoder (10 s) independently landed on the same window.

**R7 — Denoise conditionally, never unconditionally.** FireRedTTS's crossover
sits between 10 and 20 dB prompt SNR; above it, enhancement *costs* similarity
(−1.46 pp SIM at 20 dB) and below it, enhancement *buys* a lot (+6.34 pp at
0 dB). Gate on an estimated SNR. Better still, follow §3.4's model-side result —
noise/reverb augmentation plus DNSMOS data filtering beat enhancement-on-the-
prompt on *every* axis at once, and that is a data recipe rather than a new
architecture.

**R8 — Audit every duration and length estimate for byte-vs-character counting
before anything else.** §5.6: F5-TTS counts text length in UTF-8 bytes, which
inflates Indic scripts catastrophically, and fixing it took a bilingual model's
Bengali from 108% to ~24% WER with **no retraining**. We already know our Qwen3
tokenizer charges 1.46 tokens per *character* in Telugu (findings doc §3.1), so
any duration heuristic keyed on token or byte count is wrong by a language-
dependent factor. **This is the cheapest possible win in the project and it is
not a conditioning change at all.**

**R9 — Choose a point on the identity/nativeness frontier deliberately, and make
it a product decision.** §5.5: accent leakage and speaker identity are the same
signal; the best known method still pays SIM 0.651 → 0.294 to move accent 2.54 →
4.28, and naive CFG reweighting at the same accent level drops closed-set speaker
ID to **4%** against a 2.4% chance floor. There is no setting that gives a native
Hindi accent *and* full English-speaker identity. Decide which the product sells.

**R10 — Add nothing.** No cross-attention, no Perceiver resampler, no learnable
speaker encoder, no per-speaker adapters. Each adds trained parameters on the
axis we are poorest on, and §2.2 shows the ceiling they buy is about +0.02 SIM.

**R11 — Never validate this design on teacher-forced loss.** §1.4. Use it as a
wiring check — assert the ~0.27-nat drop when both routes are disabled — and rank
nothing by it.

### 8.2 What we are explicitly *not* recommending, and why

| rejected | why |
|---|---|
| Dropping `spk_proj` entirely, as CosyVoice 2 did | Tempting: it is the route that leaks language and costs WER (§1.3, §5.4). But CosyVoice 2 could drop it because its flow decoder was *trained* alongside that choice; ours is a frozen third-party checkpoint. Removing route 1 from a checkpoint trained with it is a different experiment from training without it. Test it in E3 first. **It also removes the only route that works cross-lingually (R4).** |
| Swapping the conditioning encoder as a free upgrade | `spk_proj` was trained against pyannote's 512-d output and reproduces it to max abs difference 0. A different encoder is a **retrain**, not an upgrade — §4.6 #4. |
| Adopting CAM++ for route 1 | `s3gen` already runs CAM++ internally (`s3gen.py:3921`). Using it for route 1 too would make the LM and the decoder read identity through the same model, reducing independence for no measured gain. |
| Time-level language-ID conditioning (§5.4 fix #2, the −69% WER one) | It is the best-evidenced accent fix in the literature, and we **cannot apply it**: it lives in the diffusion/flow trajectory, and our flow decoder is a frozen Chatterbox checkpoint. Record it as the strongest argument for eventually owning the decoder. |
| Asking for 20-30 s | §3. |
| Scoring with our own conditioning embedder | §4.5, §6. |

---

## 9. Decisions proposed

Numbered to continue the findings doc's D1-D12.

**D13 — `role_swapped` moves from `IDENTITY_CONTROLS` to `DIAGNOSTIC_CONTROLS`,
and Gate 2's identity verdict is restated.** The control leaves ~250 reference
audio tokens and the speaker head in place and changes four markers, so it cannot
distinguish a wrong layout from a correctly-read one. Identity conditioning
itself is **confirmed** at +0.273 nats/token, 8/8 examples, t(7) = +6.49. What
remains unconfirmed is the *layout*, and the control that tests it is E2 (§1.5).

**D14 — Teacher-forced loss is a wiring check for identity, never a design
metric.** It collapses to zero exactly when a second identity route covers for it
(§1.1) — the configuration our gate ran in — and in one published case points the
wrong way entirely (§1.4c).

**D15 — Train with *independent* dropout on the two LM identity routes** (R3).
The redundancy is what made the gate unreadable and what would make either route
individually unreliable at inference; independent dropout is the only thing that
forces each to stand alone, and it is what makes D16 legal.

**D16 — Cross-lingual generation drops the reference prefix** (R4). Evidence in
§5.3; the code path already exists.

**D17 — The product asks for 15 s, accepts 8-20 s, rejects below 6 s** — and any
20-30 s already asked for is re-spent on *selecting* the best 10-12 s window
rather than on feeding a decoder that truncates at 10 (§3.6).

**D18 — Prompt quality is a first-class product requirement, ahead of prompt
length, and denoising is conditional on SNR** (R7). §3.4.

**D19 — D8's *decision* stands; D8's *reasoning* is replaced.** Keep a separate
scorer, but not because circularity is large — it has been measured twice at
about **0.008 absolute**, and one controlled study found the matched cell was the
*worst* one (§6.6). The reasons that hold are: **comparability** (WavLM-large SV
is the field's lingua franca), **encoder variance** (up to 0.095, with rank
inversions), the fact that **our projection head is trained**, which is where the
small measured inflation actually lives, and option value. Two corrections
follow: `microsoft/wavlm-base-plus-sv` is **not** the convention, publishes no
VoxCeleb EER and carries no licence field — move the headline to the
Seed-TTS-eval WavLM-Large checkpoint (`ECAPA_TDNN_SMALL(feat_dim=1024,
feat_type="wavlm_large")`) and add ReDimNet2-B6+cnc2 (MIT) as a commercially
clean second opinion.

**D19b — The rule that matters more than the conditioning/evaluation split: if
anything in our pipeline ever optimises, reranks or selects against a
speaker-embedding score, that encoder is permanently disqualified as an
evaluator.** This is the rule the field keeps breaking — Seed-TTS uses SIM as an
RL reward and reports SIM, and admits reward hacking; Llasa reranks best-of-N
with a WavLM verifier and reports WavLM SIM. It costs us nothing today and it is
the only form of circularity with measured teeth (§6.6).

**D19c — Audit the eval harness for prompt inclusion before publishing any
number.** Whether the prompt is still attached to the scored audio is worth
**0.151 absolute SIM** — twenty times the circularity effect and larger than
everything else in §6 combined (§6.2). Also fix the three confounds in
`reference_budget.py` listed in §3.3 before it is first run.

**D20 — Do not swap the conditioning encoder without retraining `spk_proj`, and
not to CAM++ in any case.** §8.2.

**D21 — Before choosing an encoder, build an Indic ASV trial list** (§4.7). There
is no published Indic encoder benchmark, the VoxCeleb ranking demonstrably
re-orders under language shift, and a day of work produces the five numbers the
literature does not have.

**D22 — Audit duration and length estimates for byte-vs-character counting
immediately** (R8). This is not a conditioning decision; it is here because it is
the largest measured effect per engineer-hour anywhere in this note.

**D23 — Treat "English must stay in every training phase" as reopened.**
Findings-doc §3.1 argued it from tokenizer fertility. IndicF5's ablation argues
the opposite from quality: fine-tuning an English checkpoint on **Indic data
alone** scored MUSHRA 73.4 against 66.2 for mixing English back in and 43.2 from
scratch (§5.8). Different axes, genuinely unresolved, and cheap to settle with a
two-arm run once there is data.

**D24 — Keep Sarvam Bulbul output out of the pipeline entirely**, including as a
scored baseline that could feed back, under its ToS (§5.8) — the same rule the
findings doc already imposes on KittenTTS output under the Stellon licence.

---

## 10. Open questions and unverified claims

**Ours to measure (nothing here blocks on data):**
- How much of KittenTTS 2's cloning SECS comes from route 3 versus routes 1-2.
  Inferred from CosyVoice 2's ablation on a related architecture; **not measured
  on our checkpoint.** E3, §1.5.
- Whether the reference-role layout is the trained one. The released config
  declaring all five marker ids is strong circumstantial evidence; E2 is the
  measurement.
- **The English→Indic cloning benchmark does not exist.** Neither the IIT Madras
  cross-lingual work nor IndicF5 publishes a same-language-reference baseline, so
  there is no cross-lingual SECS delta for any Indic language anywhere. We would
  be creating it.
- Whether a 30 s reference is actively *harmful* to the flow decoder rather than
  merely wasted: the code truncates at 10 s, but `reference_budget.py`
  monkeypatches that cap and nobody has listened to the result.
- Whether one 15 s clip beats three 5 s clips. No published ablation for
  in-context codec-LM TTS.
- Where a statistics-pooling pyannote x-vector saturates with enrolment duration.
- The R3 dropout probabilities.

**Unverified in the literature:**
- `pyannote/embedding`'s parameter count (~4.5M is an estimate from its TDNN
  config, not published).
- **ReDimNet2's embedding dimension** — the repo publishes only `(batch,
  emb_dim)`. 192 is carried from v1 and must be read off the checkpoint before
  `spk_proj` is resized.
- `microsoft/wavlm-base-plus-sv`'s VoxCeleb EER and embedding dimension: neither
  is published. NeMo TitaNet-Large's dimension likewise.
- **VoxCeleb1/2's own licence text could not be retrieved** from the Oxford VGG
  page. Every downstream CC-BY-4.0 claim rests on it. MIT-licensed ReDimNet2 is
  the one family that does not depend on this chain.
- **Rasa's licence: CC BY 4.0 (HF tag, paper, AI4Bharat's own table) vs CC BY-ND
  4.0 (AIKosh catalogue).** Rasa feeds IndicF5, Indic-Mio, DhVaani and svara-TTS.
  Get it in writing.
- **The IIT Madras IndicTTS licence PDF timed out on four fetch attempts.** It is
  the root licence under four open Indic models.
- VoxBlink2's per-language coverage — "18 languages", no Indian language named,
  no breakdown.
- The IndoNLP 2025 "Indic SV EER 15% → 5%" result: PDF would not extract,
  baseline unknown.
- The 2026 SOTA 0.12% Vox1-O claim (w2v-BERT 2.0 + MFA): single preprint, **no
  LICENSE file**, unusable regardless.
- **No paper was found stating that scoring with the conditioning encoder is
  circular.** §4.5 — that argument is **our position**, not a citation.
- CrossAccent-TTS never states the reference language per evaluation, so whether
  its results are English→Indic is unknown; and it has no ablation, so the GRL's
  isolated contribution is unmeasured.
- The Google cross-lingual Hindi deficit (57% vs English 68%) is from a GST/VAE
  system, not a codec LM. Whether it transfers to our architecture is unknown.
- **PSP / Praxy / LASE** per-phoneme and English-prompt results are single-author,
  mutually self-citing preprints with n≈10 and an admission that 5 pp differences
  are not separable. Hypothesis-generating only — including the Tamil ழ /ɻ/ 85.7%
  collapse and the LASE "Indian-accented references lose 14× less identity"
  result, both of which would be significant if they replicate.
- LP-TTS-Bench's and RVCBench's prompt-length curves are figures only; the
  saturation points are read off plots.
- The DAIEN-TTS noise-vs-reverb normalisation in §3.4 is **derived** — the paper
  itself never compares the two conditions against their own ceilings.
- No measured CPU latency or RTF for any speaker encoder in §4.2; GMACs only.
- The different-speaker SIM floor for the *exact* Seed-TTS-eval checkpoint.
  UniSpeech publishes examples only for `unispeech_sat` (0.0317) and
  `ecapa_tdnn` (0.2053). **Measure it on 1,000 random cross-speaker pairs — 20
  minutes, and it calibrates every number we report.**
- **No published study isolates "TTS copies prompt noise → SIM rises."** The
  mechanism is strongly implied by §6.2 + §4.5 + §3.4 but nobody has shown it.
  This is the weakest link in the §6.5 argument.
- No paper demonstrates reward hacking of a *speaker-similarity* reward
  specifically. The closest analogues are Seed-TTS's general admission,
  DMOSpeech's SV-loss ablation, and reward hacking of UTMOS/DNSMOS/SCOREQ. **Do
  not overclaim this.**
- No study correlates the *exact* WavLM-large SV checkpoint with human judgement;
  all the correlation evidence in §6.5 is for the ASV-cosine metric *class*.
- Whether TTSDS2's 14 languages include any Indian language — the list is in a
  figure that would not extract. **Assume Hindi is not covered until checked.**
- Seed-TTS-eval's exact aggregation (`average.py` 404s); a plain mean over
  per-utterance scores is the obvious reading, unconfirmed from source.

**Gaps in the field, not in our search:**
- No 2024-2026 benchmark of modern speaker encoders on Indian-language speaker
  verification reporting EER.
- No controlled reference-length ablation beyond 10 s for an in-context codec-LM
  TTS, and none at all between 30 s and 10 minutes except LP-TTS-Bench's figure.
- No direct user-preference evidence on whether prosody-matching is wanted. The
  nearest proxy says users reward *expressiveness*, not *copying*.
- **Seed-TTS-eval has no Indic test set, and there is no Indic TTS leaderboard,
  arena or MOS convention at all** (§6.8). TTS Arena V2 is English-only. Nobody
  has validated WavLM-SV or UTMOS against Indian-language human ratings, although
  `ai4bharat/MANGO` (CC BY 4.0, ~255k ratings, Hindi + Tamil) makes it a day's
  work. The Indic half of our evaluation is ours to define — and worth
  publishing.
