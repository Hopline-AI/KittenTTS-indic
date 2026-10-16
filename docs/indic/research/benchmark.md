# The benchmark: the one yardstick every architecture, codec and data choice is scored on

**Research note, 2026-10-16.** This specifies the measurement instrument, not a
result. Nothing here has been run. Where a number is arithmetic from something we
already measured it says so; where it comes from a paper it carries the URL;
where it is an assumption waiting on a pilot it is labelled an assumption.

Read `docs/indic/2026-10-16-findings-and-decisions.md` first. Decisions **D8–D11**
there already fixed the evaluation methodology — the speaker embedder, the ASR
floor, where cloning references come from, and that Set C reports a gap. This
document does not revisit them; it builds the rest of the instrument on top and
numbers its own decisions **B1–B16** so the two do not collide.

---

## 1. State of play, in one paragraph

We have half an instrument. `indic/eval/` can already synthesize a test set, run
it through the right recognizer per language, subtract the ASR floor, and score
speaker similarity — resumably and shardably. What it cannot do is any of the
things this project's actual claims rest on: it cannot measure inference cost, it
cannot say whether a difference between two arms is real, it cannot produce a
quality-versus-training-hours curve, and its Set A is **too small to decide
anything** — 118 parallel FLEURS sentences on the dev split (measured, from
`indic/data/fertility.json`), which resolves a paired CER difference no smaller
than **2.3 points** and, once corrected for comparing six arms, **2.9 points**.
Every candidate architecture we would plausibly consider differs by less than
that. The single highest-value change in this document is not a new metric; it is
**n**.

---

## 2. THE SCORECARD

One page. Every architecture, codec and data candidate gets exactly these rows.
An arm that has not filled all of Tier 1 has not been evaluated.

### Tier 0 — gates. Binary. Nothing below is reportable until all four pass.

| Gate | Definition | Threshold | Computed by |
|---|---|---|---|
| **synthesis** | fraction of Set A records that produced a non-empty wav without raising | ≥ 0.99 | `run_eval.py` already records `error`; `score.py` counts it |
| **runaway** | fraction of generations longer than 2× the ground-truth recording | < 0.01 | new, `score.py` (`audio_seconds` vs `truth_seconds`, both already on the row) |
| **script** | fraction of ASR hypotheses whose majority codepoint block is the target language's `LANGUAGES[lang].block` | ≥ 0.98 | new, `metrics.py` |
| **floor coverage** | fraction of scored rows that carry a `floor_cer` | = 1.00 on Set A | `score.py` already has it; D9 requires the headline be restricted to this subset |

The script gate exists because of a failure mode `asr.py` already documents:
Whisper left to identify the language hears a badly synthesized Hindi clip as
Urdu and writes Arabic script. With the language pinned it cannot do that, but
the *model* can still emit the wrong script, and then CER measures orthography
rather than pronunciation. The runaway gate exists because RTF has no defence
against a model that truncates — see §5.1.

### Tier 1 — the four numbers that decide an arm

| # | Metric | Exactly how | Set | n |
|---|---|---|---|---|
| **1** | `cer_above_floor` | corpus CER (total errors / total reference chars, `metrics.corpus_rate`) minus the same quantity on the human recording of the same sentences. ASR: `ai4bharat/indic-conformer-600m-multilingual`, **RNNT** decoding, for hi/bn/ta/te/mr; `openai/whisper-large-v3`, greedy, language pinned, for en. Normalization: `metrics.normalize_for_scoring` (NFC → Indic digits → drop P/S/C → casefold) | A | 400 / lang / arm |
| **2a** | `secs_wavlm` | cosine between `microsoft/wavlm-base-plus-sv` x-vectors of the output and the *reference clip the model was given*, reported with its same-speaker ceiling and different-speaker floor from FLEURS on the same clips, and as `secs_norm = (s − floor)/(ceiling − floor)` | A | 400 / lang / arm, ≥ 32 reference clips |
| **2b** | `sim_seedtts` | the same cosine computed with the Seed-TTS-eval speaker verifier (WavLM-large + ECAPA-TDNN head, `wavlm_large_finetune.pth`, mirrored at `bezzam/wavlm_large_finetune_seed_tts_eval`) | A | same rows |
| **3** | `rtf_stream`, `ttfa_p50`, `ttfa_p95` | §5. Single stream, end-to-end, medium duration bucket, clocks locked, 5 repeats | E | 60 / lang, 5 repeats |
| **4** | `streams_per_gpu@SLO` | largest concurrency C at which p95 TTFA ≤ 300 ms **and** p95 per-stream RTF ≤ 1.0. §5 | E | sweep |

**B1 — report two speaker-similarity numbers, not one.** D8 chose
`wavlm-base-plus-sv` as the default and the reasoning stands: it is independent of
the conditioning path, ungated, and CPU-cheap. But no published number is
reported against it. Seed-TTS-eval, the F5-TTS line, the Open TTS Leaderboard and
the 2026 Indic cross-lingual benchmark all use a WavLM-large fine-tuned for
speaker verification
([seed-tts-eval](https://github.com/BytedanceSpeech/seed-tts-eval),
[F5-TTS](https://arxiv.org/html/2410.06885v3),
[Open TTS Leaderboard](https://huggingface.co/blog/open-tts-leaderboard),
[arXiv 2610.14826](https://arxiv.org/html/2610.14826)). Adding it costs one more
forward pass per clip and is the single cheapest thing that makes our similarity
numbers comparable to everyone else's. `secs_wavlm` decides arms; `sim_seedtts`
is what goes in a paper or a README.

**B2 — the reference pool must grow from 4 to at least 32.** `testsets.py` sets
`REFERENCE_POOL = 4` and assigns clips round-robin. With 400 utterances that is
100 utterances per reference clip, and SECS is strongly clustered by the
reference speaker. The design effect is `1 + (m − 1)·ICC`:

| pool | utt/clip | ICC 0.05 | ICC 0.10 | ICC 0.20 |
|---|---|---|---|---|
| 4 | 100 | n_eff 67 | **n_eff 37** | n_eff 19 |
| 8 | 50 | 116 | 68 | 37 |
| 16 | 25 | 182 | 118 | 69 |
| **32** | 12.5 | 254 | **186** | 121 |

At pool 4, 400 utterances buy the statistical power of 37. Worse than the lost
power: with four reference clips the *arm × speaker interaction* is estimated from
four points, so an arm that happens to clone one of those four voices well wins
the comparison on one clip. The clustering largely cancels in a paired
arm-vs-arm difference, because both arms get the same clips — it does **not**
cancel for the absolute SECS we would quote, and it does not cancel for the
interaction. Pool 32.

### Tier 2 — always reported, never decides alone

| Metric | Why it is here rather than Tier 1 |
|---|---|
| `wer_above_floor` | D's own module docstring is right that CER leads on Indic; WER catches the case where CER stayed flat because the model substituted a plausible wrong word |
| per-language `cer_above_floor` for bn/te/mr | Tier 1 decides on hi and ta (§8); the other three are reported and watched, not powered |
| Set C digit gap (`cer(native) − cer(ascii)` within a `pair_id`) | D11: a gap, never a level |
| `vram_reserved_gb` and `vram_nvml_gb` at the SLO concurrency | §5.1 |
| `wh_per_audio_hour`, `usd_per_audio_hour` | §5.1 |
| `rtfx_offline` (best batch, aggregate) | a different and much larger number than `1/rtf_stream`; never conflate the two |
| `kv_bytes_per_token` | derived *and* measured; the check is that they agree (§5.4) |
| `audio_bits_per_second` from the teacher-forced gate | §10, item 8 — this replaces perplexity whenever codecs differ |
| `tokens_per_word` from `tokenizer_fertility.py` | already measured; explains most of the prefill half of TTFA |

### Tier 3 — milestone only, not per-arm

Run these at a release candidate, never in the inner loop.

| Metric | Cost | When |
|---|---|---|
| **TTSDS2** multilingual score | ~9.4 CPU-minutes per system score ([TTSDS2 §5](https://arxiv.org/html/2506.19441v1)) | once per milestone, en + hi if covered |
| **Seed-TTS-eval test-en** WER + SIM | ~1,088 synths | before any English claim |
| **CMOS**, native raters | 400 ratings / language / pair (§7) | twice in the project's life (§7) |

### Tripwire — fires an alert, never enters a table

`utmosv2` on the English arm only, compared to *the same arm's previous
checkpoint*. A drop of more than 0.5 means the vocoder or codec detokenizer broke;
a rise means nothing. See §3.3 for why it cannot be anything more than this.

---

## 3. Why these and not the others

### 3.1 The recognizers

The field has converged, and it has converged on what `asr.py` already does.

- **Seed-TTS-eval**, the most-cited zero-shot TTS yardstick, uses
  `openai/whisper-large-v3` for English and `funasr/paraformer-zh` for Mandarin
  ([repo](https://github.com/BytedanceSpeech/seed-tts-eval),
  [Seed-TTS paper](https://arxiv.org/pdf/2406.02430)).
- The **2026 Indic cross-lingual TTS benchmark** uses exactly our split:
  Whisper-large-v3 for English, **IndicConformer-600M for the Indian languages**
  ([arXiv 2610.14826](https://arxiv.org/html/2610.14826)).
- The **Open TTS Leaderboard** (published 2026-09-30) has moved to
  `Qwen/Qwen3-ASR-1.7B-hf` on the grounds that it tops the Open ASR Leaderboard
  ([blog](https://huggingface.co/blog/open-tts-leaderboard),
  [repo](https://github.com/huggingface/open-tts-leaderboard)).

**B3 — keep IndicConformer-600M RNNT as the headline scorer and do not chase
Qwen3-ASR.** `asr.py` already states the rule: changing the scorer invalidates
every number taken before the switch. IndicConformer is what the Indic TTS
literature reports against, which is the comparability we actually need. Qwen3-ASR
is worth adding as a *third* backend behind `--asr qwen3` precisely so a future
reader can see whether an IndicConformer regression is the recognizer having an
opinion — the same role Whisper plays today. It is not the headline.

**B4 — the floor is not optional and the field now agrees.** The 2026 Indic
benchmark publishes ground-truth rows alongside every system, and they are
enormous: GT WER 25.39 on Tamil, 21.06 on Telugu, 10.56 on Marathi, 14.40 on Hindi,
against GT CER 4.87 / 3.07 / 2.08 / 5.21 ([arXiv
2610.14826](https://arxiv.org/html/2610.14826)). A Tamil TTS WER of 30 against a
GT of 25.39 is a different claim from a Tamil WER of 30 against a GT of 3. D9 is
vindicated; this document only adds that **the floor must be re-measured on every
language for our own recognizer configuration**, because theirs is not ours
(decoder, normalizer and clip set all differ).

### 3.2 Speaker similarity

Three checkpoints are in circulation and they are not interchangeable:

| Checkpoint | Used by | Note |
|---|---|---|
| WavLM-large fine-tuned for SV (`wavlm_large_finetune.pth`) | Seed-TTS-eval, F5-TTS, Open TTS Leaderboard, Indic cross-lingual benchmark | the de facto standard; **this is `sim_seedtts`** |
| `microsoft/wavlm-base-plus-sv` | our default per D8 | ungated, CPU, independent of the TTS stack |
| ECAPA-TDNN | the 2026 Hindi distillation paper ([arXiv 2607.18662](https://arxiv.org/pdf/2607.18662)) | reports IndicF5 at 0.784 and a 190M student at 0.750 on 50 Hindi FLEURS sentences |
| wav2vec2 fine-tuned on IV-R | the IndicVoices-R benchmark's S-SIM ([arXiv 2409.05356](https://arxiv.org/pdf/2409.05356)) | scored 0–100, trained on the eval corpus; **not comparable to any of the above** |

**B5 — never quote an S-SIM against a SIM.** IV-R reports 78.93 for an
IndicTTS-finetuned model and 88.18 for an IV-R-finetuned one; those are not
0.789 and 0.882 on the WavLM scale and putting them in the same column as a
Seed-TTS SIM would be a fabrication. If we report on the IV-R benchmark, we
report its number in its own row with its own embedder named.

TTSDS2's correlation study is a useful warning in the other direction: speaker
similarity correlates *well* with human judgement in noisy and in-the-wild
domains (ρ up to 0.85 for RawNet3 on Wild MOS) and *poorly* on clean read speech
(ρ 0.36 on Clean MOS) ([arXiv
2506.19441](https://arxiv.org/html/2506.19441v1), Table 2). Set A is clean read
speech. So SECS on Set A is a *conditioning-worked* check, not a quality metric,
and it should be read that way: against its floor and ceiling, as a fraction of
the headroom that exists.

### 3.3 MOS predictors — the case is now closed, and it is worse than we thought

D8-era reasoning in `metrics.py` says automatic MOS is relative-only on Indic
because it was trained on English listening tests. The 2025–2026 literature says
something stronger: **on modern, clean TTS these predictors are near chance even
in English.**

TTSDS2's Spearman correlations with listening tests, by domain (MOS / CMOS /
SMOS) ([arXiv 2506.19441](https://arxiv.org/html/2506.19441v1), Table 2):

| metric | Clean | Noisy | Wild | Kids |
|---|---|---|---|---|
| **TTSDS2** | 0.75 / 0.69 / 0.73 | 0.59 / 0.54 / 0.71 | 0.75 / 0.71 / 0.75 | 0.61 / 0.50 / 0.70 |
| UTMOS | 0.51 / 0.30 / 0.31 | 0.47 / 0.29 / 0.00 | **−0.12** / −0.12 / −0.26 | −0.02 / −0.18 / −0.04 |
| UTMOSv2 | 0.39 / 0.25 / 0.09 | 0.34 / 0.36 / 0.19 | 0.16 / 0.14 / −0.04 | 0.05 / 0.03 / −0.02 |
| DNSMOS | 0.41 / 0.37 / 0.22 | 0.57 / 0.36 / 0.22 | 0.35 / 0.28 / 0.03 | 0.31 / 0.10 / 0.28 |
| NISQA | 0.05 / 0.00 / 0.06 | 0.05 / −0.21 / −0.53 | −0.32 / −0.33 / −0.64 | −0.29 / −0.27 / −0.46 |
| **WER** | **−0.19** / −0.18 / −0.17 | −0.11 / −0.30 / −0.13 | −0.28 / −0.17 / −0.22 | −0.45 / −0.26 / −0.39 |

A 2026 study of 13 reference-free predictor families (33 score variants) on six
human-rated corpora makes the saturation explicit ([arXiv
2609.13150](https://arxiv.org/html/2609.13150)). Pairwise accuracy, chance = 0.5:

| predictor | BVCC (artifact-rich) | TTS-Arena | TTS-HP (clean commercial) |
|---|---|---|---|
| human ceiling | 0.887 | — | **0.764** |
| SCOREQ | 0.909 | 0.585 | 0.502 |
| UTMOS | 0.892 | 0.577 | 0.510 |
| UTMOSv2 | 0.899 | 0.540 | **0.528** |
| Distill-MOS | 0.868 | 0.553 | 0.517 |
| DNSMOS | 0.678 | 0.507 | 0.516 |
| *clip duration* | 0.517 | 0.523 | *0.524* |

On clean modern TTS the best predictor (0.528) is statistically
indistinguishable from **clip duration** (0.524), with bootstrap 95% intervals of
about ±0.02. And optimising one directly collapses: using SCOREQ as a reward for
300 GSPO steps drove the reward from −4.71 to −0.19 while held-out UTMOS-v1 fell
from 4.51 to 1.23 and human Elo dropped to 1675±115 against a 2098±101 baseline —
a KL penalty at β = 0.1 did not prevent it.

**B6 — no MOS predictor appears in the scorecard, ever, in any tier.** Not as a
headline, not as a secondary, not averaged into a composite, and absolutely not
as a training reward. The one surviving use is the tripwire in §2: UTMOSv2 on the
English arm against that arm's own previous checkpoint, where a large drop means
the vocoder broke. That is a regression detector, not a quality metric, and it is
printed outside the table so nobody puts it in a slide.

Supporting evidence that this is not an Indic-specific worry: DNSMOS's own 2020
paper reports its primary model "does not generalize well for categories other
than English", SRCC 0.65 ([arXiv 2010.15258](https://arxiv.org/abs/2010.15258));
UTMOSv2 scores *original recordings* of Walloon below human ratings, attributed
to English-only optimisation ([arXiv 2508.09767](https://arxiv.org/pdf/2508.09767));
and a 2026 cross-domain comparison has UTMOSv2 at ρ ≈ 0.5 on out-of-domain data
([ProxyMOS, arXiv 2610.00419](https://arxiv.org/html/2610.00419)).

**B7 — WER/CER is an intelligibility floor, not a quality score.** The TTSDS2
table above shows WER *negatively* correlated with MOS in every domain. That is
not a reason to stop measuring it — a TTS that cannot be transcribed is broken,
and on Indic that is the live risk, not naturalness. It is a reason to stop
treating "lower CER" as "better model" once CER is near the floor. The scorecard
encodes this: `cer_above_floor` decides an arm **while it is above the floor**;
once an arm reaches floor + 2 points, further CER differences are reported but no
longer break a tie, and the tie breaks on efficiency.

### 3.4 Has anything better appeared? Yes: TTSDS2

TTSDS2 (ICLR 2026, [paper](https://arxiv.org/abs/2506.19441),
[code](https://github.com/ttsds/ttsds)) scores a system by how close its
*distributions* of perceptual-factor features are to real reference data, rather
than predicting a MOS for each clip. Factors and features:

| factor | features |
|---|---|
| Generic | WavLM, HuBERT-base, wav2vec2-base activations |
| Speaker | d-Vector, WeSpeaker |
| Prosody | PyWORLD F0, HuBERT speaking rate, Allosaurus speaking rate, prosody embeddings |
| Intelligibility | wav2vec2-ASR and Whisper-small ASR **activations** (not WER) |

Multilingually it substitutes mHuBERT-147 for HuBERT and XLSR-53 for wav2vec2. It
is the only metric among 16 compared that exceeds ρ 0.50 with listening tests in
every domain and for every subjective score. Its multilingual arm regenerates a
YouTube-derived test set across 14 languages — 250 videos scraped per language,
≤16 utterances each, 50 speaker-matched pairs kept, 50 utterances synthesized per
system per language — specifically to avoid test-set leakage.

**B8 — TTSDS2 is a Tier 3 milestone check, not a per-arm metric.** Three reasons,
in order. It needs a *reference distribution* in the target language that we would
have to build; its 50-utterance-per-language design is an order of magnitude below
the n we need to separate arms (§8); and it costs ~9.4 CPU-minutes per system
score. It is the right instrument for "is our Hindi in the same distribution as
real Hindi speech", which is a milestone question, and the wrong one for "is arm B
better than arm A", which is a weekly question. Confirm the 14-language list
covers hi/bn/ta/te/mr before planning on it — the paper does not enumerate them.

---

## 4. Standard benchmarks: getting on a shared yardstick

Our own Set A is unavoidable — no public benchmark carries an ASR floor in our
five languages with our recognizer. But a number nobody can compare to anything is
a weak claim. Four external sets, in priority order.

| Benchmark | What it is | Our use | Cost |
|---|---|---|---|
| **Seed-TTS-eval `test-en`** | 1,088 English samples from Common Voice; WER by Whisper-large-v3, SIM by WavLM-large-SV. [repo](https://github.com/BytedanceSpeech/seed-tts-eval), [EvalScope](https://evalscope.readthedocs.io/en/latest/benchmarks/seed_tts_eval.html) | **Run it.** It is the single most-cited zero-shot TTS number; reported WERs now sit near 1.3–3.0% ([Qwen3.5-Omni 1.26](https://arxiv.org/html/2604.15804v2), [dots.tts 2.92](https://arxiv.org/pdf/2606.07080), [WavTTS 1.50](https://arxiv.org/html/2606.03455v1)). Our English arm must not be embarrassing here, and D1 keeps English in every training phase anyway | ~1,100 synths |
| **Cross-Lingual Indic Test Set** | 9 Indian languages + English, ~2 h per language pair, IndicConformer-600M + Whisper-large-v3 + WavLM-large + UTMOS, with GT rows. [repo](https://github.com/Cross-Lingual-TTS-for-Indian-Languages/Cross-Lingual-TTS-Indic-Test-Set), [paper](https://arxiv.org/html/2610.14826) | **Run the Indic→Indic direction** on hi/bn/ta/te/mr. This is the only published Indic set with baselines and ground-truth floors in our exact five languages and with our exact recognizer. It is the comparability claim | 759–1,179 utt/lang; sample to 400 |
| **LibriSpeech-PC test-clean (F5 subset)** | 1,127 samples, 39 speakers, 2 h. [F5-TTS](https://arxiv.org/html/2410.06885v3) reports 2.42% WER / 0.66 SIM-o at 32 NFE | Only if we claim English parity with the flow-matching line. Otherwise skip | ~1,100 synths |
| **CV3-Eval `zero_shot/en`, 50 prompts** | the Open TTS Leaderboard's TTFA protocol. [repo](https://github.com/QwenAudio/CV3-Eval) | **Run the 50-prompt TTFA slice only.** It makes our time-to-first-audio directly comparable to every model on that board, for almost nothing | 50 synths |

**B9 — do not run Seed-TTS-eval `test-zh` or `test-hard`.** 2,020 Mandarin samples
plus hard cases, scored by Paraformer-zh, in a language we do not target and whose
result would change no decision. The cost is days of synthesis and a second ASR
stack.

**B10 — do not enter TTS Arena or Artificial Analysis during development.** Arena
Elo is the right instrument for a release and the wrong one for a weekly decision:
the Open TTS Leaderboard authors give the reasons themselves — arenas cannot keep
up with release pace, voter criteria drift over time, and open-weight models are
structurally underrepresented because the operator has to host them (16 of 92
models on Artificial Analysis are open-weight)
([blog](https://huggingface.co/blog/open-tts-leaderboard)). A vendor's own claim
of an arena top spot is a vendor claim ([MiniMax-Speech,
arXiv 2505.07916](https://arxiv.org/pdf/2505.07916)).

**B11 — the IndicVoices-R benchmark is a data source, not a yardstick.** IV-R is
1,704 h / 10,496 speakers / 22 languages and its benchmark targets zero/few/many-shot
speaker generalization ([arXiv 2409.05356](https://arxiv.org/pdf/2409.05356)). But
its S-SIM uses a wav2vec2 fine-tuned *on IV-R itself*, and IndicF5 — the obvious
Indic comparator — was trained on 1,417 h from Rasa, IndicTTS, LIMMITS and IV-R
([model card](https://huggingface.co/ai4bharat/IndicF5)). Scoring IndicF5 on IV-R
is scoring it on its own training distribution. Use IV-R for training data; report
against the cross-lingual Indic test set.

---

## 5. Efficiency: the measurement protocol

This is the differentiator, so this is the section that has to be reproducible on
somebody else's cluster. The field has definitions but no protocol: RTF is
consistently "generation wall-clock / synthesized audio duration"
([Chatterbox-Flash](https://arxiv.org/pdf/2605.30748)), TTFB is consistently
"request to first audio" ([Deepgram](https://developers.deepgram.com/docs/text-to-speech-latency)),
and essentially nobody states clocks, exclusivity, warmup, repeats or variance.
The result is numbers that cannot be compared — Chatterbox-Flash's own authors say
their cross-system table "is not a controlled, like-for-like comparison", and
Inworld's sub-100 ms p99 TTFB figure excludes network latency
([Inworld](https://inworld.ai/resources/tts-latency-ttfb-p99)).

### 5.1 Definitions

**B12 — RTF is per *output* audio second, end to end, and it is void without the
quality row beside it.**

```
rtf   = wall_clock_seconds(generate) / audio_seconds(output)
rtfx  = 1 / rtf
```

The timer starts at the public API entry point with a text string in hand and
stops when the last PCM sample is available to the caller. It **includes** the
text frontend, tokenization, prefill, decode, codec detokenization and vocoder. It
**excludes** model construction, weight load and the first `--warmup` generations.

The denominator is the *output* duration, not the reference duration, and that is
the hole: a model that truncates shrinks numerator and denominator together and
its RTF does not move. Hence Tier 0's runaway gate, hence the hard cap at 2×
ground truth, and hence the rule that **every efficiency row carries the CER of
the audio it timed**. An arm whose Set E CER exceeds its own Set A CER by more
than 2 points has degraded under the efficiency configuration and its RTF is not
reportable.

**TTFA** — wall-clock from the generate call to the first PCM sample available to
the caller. For a non-streaming arm this equals total generation time and **must
be reported that way, not as N/A**. KittenTTS 2's measured 17 s to first audio is
the honest number and it is the headline weakness; hiding it behind "non-streaming"
would hide the problem this project exists to solve. Report p50 / p90 / p95 over
≥ 50 utterances with the first 3 discarded — the Open TTS Leaderboard's convention,
which keeps our number comparable
([repo](https://github.com/huggingface/open-tts-leaderboard)).

**Throughput.** Two numbers that are routinely conflated:

```
streams_per_gpu@SLO = max C such that p95(ttfa) <= 300 ms AND p95(rtf_per_stream) <= 1.0
rtfx_offline        = sum(audio_seconds) / wall_clock, at the arm's best batch size
```

The first is what a product can serve; the second is what a batch job achieves and
is much larger. The Open TTS Leaderboard's RTFx is the second kind (batched
offline, H200, each backend choosing its own batch size), and its TTFA is a
separate batch-1 measurement on an A100 — they are deliberately not the same run.
Concurrency degrades both severely and non-linearly: a 2026 report measures a
0.3B streaming TTS going from 259 ms TTFC / 0.197 RTF at concurrency 1 to 9,848 ms
/ 1.718 at concurrency 160 ([arXiv 2601.10770](https://arxiv.org/pdf/2601.10770)).
A single-stream RTF is therefore not a serving claim.

**VRAM.** Report both `torch.cuda.max_memory_reserved()` — per **D3**, because
that is what OOMs — and the NVML per-process used bytes, because a serving stack
(vLLM, ONNX Runtime, a separate vocoder session) allocates outside the torch
allocator and the gap is where a deployment dies.

**Energy.** NVML's cumulative energy counter (`nvmlDeviceGetTotalEnergyConsumption`,
millijoules) differenced across the timed region. State the boundary: **whole
board** — compute die, memory and losses — not wall outlet. Read the counter, not
instantaneous power sampling; tools disagree substantially and `nvidia-smi`
polling and direct NVML give different answers
([arXiv 2407.16893](https://arxiv.org/pdf/2407.16893),
[NeuralWatt methodology](https://docs.neuralwatt.com/energy-methodology.md)).
Report:

```
wh_per_audio_hour = (joules_total / audio_seconds) * 3600 / 3600 = joules / audio_second  -> Wh/h = W
```

i.e. **average board watts per real-time stream equivalent**, which is the figure
that composes.

**Cost.** Not a measurement; a measurement times an input.

```
rtf_aggregate      = rtf_at_C / C                  # GPU-seconds per audio-second
usd_per_audio_hour = gpu_price_per_hour * rtf_aggregate
```

The GPU price goes in the JSON as an input and is never averaged into anything.

### 5.2 The protocol

1. **Reference hardware.** `NVIDIA A100-SXM4-80GB`, one card, nothing else on it.
   The dev `RTX 5070 Ti Laptop` is a permitted second row and is **never** the
   comparison basis: a laptop card throttles and shares with a display.
2. **Lock the clocks.** `nvidia-smi -pm 1`, `nvidia-smi -lgc <base,base>`,
   `nvidia-smi -lmc <mem>`. Record the applied clock and power limits from
   `nvidia-smi -q` into the output JSON. Unlocked, the same A100 in a hot rack and
   a cold rack differ by more than most of the effects we are trying to detect.
   If clocks cannot be locked, the run is marked `clocks_locked: false` and its
   numbers are advisory only.
3. **Assert exclusivity.** NVML compute-process count must be 1 before the first
   timed region, and is re-asserted after. A second process invalidates the run.
4. **Warm up.** 3 untimed full generations at the medium bucket, discarded. This
   also pays for kernel autotune, CUDA-graph capture and any lazy import.
5. **Repeat 5 times, each in a fresh process.** Allocator state and compiled-kernel
   caches carry within a process and would make repeat 5 look faster than repeat 1.
   The HF and page caches stay warm — we are not measuring disk.
6. **Shuffle within a repeat**, with a per-repeat seed, so a thermal ramp does not
   land on the same utterances every time.
7. **Statistic.** Per repeat, take the median over utterances. Across repeats,
   report the **median of medians**, the min and the max, and
   `spread = (max − min) / median`. **If `spread > 0.05` the number is not
   reportable** — go back to steps 2 and 3. Report the within-repeat IQR
   separately; it is utterance variation, which is real, not instrument noise.
8. **Pin the decode.** temperature, top-p, repetition penalty, max new tokens and
   the 2× duration cap are fixed and recorded. Sampling changes output length and
   therefore both halves of RTF.
9. **Record the environment** exactly as `run_eval.py` already does — git commit,
   dirty flag, `TRACKED_PACKAGES` versions, model revision — plus device name,
   driver version, CUDA version, dtype (from `hpc.capability.select_dtype`),
   attention implementation (from `select_attn_implementation`), and whether
   `torch.compile`, CUDA graphs or vLLM were in the path.

### 5.3 Set E — the efficiency test set

Timing on Set A would confound arms, because the arms have different tokenizers
and Set A's sentences have different lengths in each language. Set E fixes
duration instead.

- Drawn from the **same 6-way parallel FLEURS ids as Set A**, so content is fixed
  across both language and arm.
- Three buckets by the *ground-truth recording's* duration: **short 2–4 s**,
  **medium 6–10 s**, **long 15–25 s**. 20 utterances per bucket per language = 60.
- FLEURS sentences rarely reach 15–25 s, so the long bucket concatenates 2–3
  consecutive parallel sentences joined by the language's own terminator
  (`LANGUAGES[lang].danda` decides `।` vs `.`). This is also the realistic serving
  case: paragraph input, where TTFA matters most.
- **One fixed reference clip per language**, 20 s, identical for every arm, so the
  reference-prefill cost is held constant and the only prefill difference between
  arms is the tokenizer — which is exactly the thing we want TTFA to expose.
- Deterministic: `run_eval.seed_for(record_id, base_seed)` is reused, so a Set E
  record synthesizes identically on a resume.

### 5.4 What the measured baseline already implies

Arithmetic from the figures in the brief and §2 of the findings doc, not new
measurements — but it decides where to look.

**KV cache is not the constraint.** The derived formula for KittenTTS 2's shape
is `2 (K,V) × 28 layers × 8 kv_heads × 128 head_dim × 2 bytes = 114,688 B =
112.0 KiB/token`, which matches the measured 112 KiB/token exactly. A 10 s
utterance under the Sarvam layout is ≈ 32 text + 150 reference audio + 250 target
audio + 10 markers = **442 tokens → 48.3 MiB of KV per stream**. On an 80 GB A100
with 3.47 GB of bf16 weights and ~2 GB of runtime, that is **~1,580 concurrent
streams before KV runs out**. Concurrency will be compute-bound long before it is
memory-bound, so the efficiency story has to be told in RTF and TTFA, not in KV
footprint. (This also means `kv_bytes_per_token` belongs in Tier 2 as a *sanity
check that the derived and measured values agree*, not as a differentiator.)

**Where the money is.** With `rtf_aggregate = rtf_at_C / C`:

| rtf_aggregate | Wh/audio-hour @ 400 W board | $/audio-hour @ $2/GPU-h |
|---|---|---|
| 0.78 (today, batch 1) | 312 | 1.56 |
| 0.20 | 80 | 0.40 |
| 0.05 | 20 | 0.10 |
| 0.025 | 10 | 0.05 |

A 30× improvement in `rtf_aggregate` is a 30× improvement in both energy and cost.
That is the differentiator, stated as a number an arm can be held to.

### 5.5 The efficiency table every arm fills

```
arm  lang  bucket  rtf  rtfx  ttfa_p50_ms  ttfa_p95_ms  spread  cer_gate
arm  C     ttfa_p95_ms  rtf_p95  vram_reserved_gb  vram_nvml_gb  W  streams@SLO
```

---

## 6. Data efficiency: the quality-vs-hours curve

This is how "trainable on a small corpus" stops being a slogan. The field does
not have a clean curve to copy — WavTTS reports SIM-o in the 0.3 range at ~585 h
of English versus sub-2% WER on 100K h ([arXiv 2606.03455](https://arxiv.org/html/2606.03455v1)),
BASE TTS is a 100K-h result ([arXiv 2402.08093](https://arxiv.org/pdf/2402.08093)),
and Zero-Voice claims generalization from 5–8 h of transcribed data
([TechRxiv](https://www.techrxiv.org/users/882196/articles/1260796-zero-voice-a-low-resource-approach-for-zero-shot-text-to-speech)).
Those are three different claims measured three different ways. We measure ours.

**B13 — fixed optimizer steps, not fixed epochs.** At fixed epochs the 1/32 arm
takes 1/32 of the gradient updates and the curve measures compute and data
confounded. At fixed steps the small fractions simply repeat more, which is the
question a practitioner actually asks: *if I only have 30 hours, how good can I
get for the same compute?*

**Fractions.** `1/32, 1/16, 1/8, 1/4, 1/2, 1` — log-spaced, because a power law
needs log spacing and a linear grid spends most of its points where the curve is
flat. On ~900 h that is **28 / 56 / 113 / 225 / 450 / 900 hours**.

**Nested and stratified.** Assign each utterance a stable key
`u = crc32(utterance_id) / 2^32` — reusing the one hash the project already has,
in `run_eval.seed_for`. Within each `(lang, speaker)` bucket sort by `u` and take
the first `⌈f·n⌉`. Nesting matters: without it, 1/8 and 1/4 differ in *which* data
as well as how much, and the curve acquires a sampling wiggle indistinguishable
from a real plateau. Stratifying matters because an unstratified 1/32 draw can
silently lose a language or collapse to a handful of speakers, and then the curve
measures speaker diversity rather than hours.

**Held fixed across every point:** total optimizer steps, LR schedule including
warmup and decay horizon, global batch in tokens, sequence length, initialisation
(`mean_rescaled` per **D2**), seed, eval set, and the checkpoint-selection rule.
That last one is a trap: "best on dev" is a selection over noise, and it biases
the noisier small-data arms upward. **Take the last checkpoint.**

**Seeds.** 3 seeds at the two anchor fractions (1/8 and 1/1), 1 seed elsewhere.
Run-to-run SD is approximately constant in `f`, so estimate it where it is
cheapest and apply the pooled SD as the band at every point. Reporting a 6-point
curve with no seed band is reporting noise as structure.

**A second, 2-point family at fixed *epochs*** (1/8 and 1/1 only) separates
"repeating the same hours hurts" from "distinct hours help". Two extra runs, and
without them the fixed-step curve has two readings.

**Budget.** 6 + 4 (extra seeds) + 2 (fixed-epoch family) = **12 runs**, each the
same cost because steps are fixed. Run the curve at **1/4 of the production step
budget**: the absolute level shifts but the exponent barely does, and the chosen
fraction then gets one full-budget confirmation run. State the shift honestly —
the curve's CER is not the production CER.

**Scope.** Curve on **hi and ta only** (parity case and worst case, per §8). The
other three ride along in the confirmation run. This halves the eval cost of the
whole exercise.

**Fit and headline.**

```
CER(h) = C_inf + A * h^(-alpha)
```

Three free parameters on six points, so report `alpha` with a profile-likelihood
interval and **refuse to extrapolate more than 2× beyond the largest measured h**.
The headline number is not the fit, it is a measurement:

> **h\* = the smallest fraction whose CER 95% CI overlaps the full-data CER's.**

That is "needs little data" as something we measured rather than something we
extrapolated.

---

## 7. Human evaluation: three moments, and not a fourth

**B14 — human evaluation happens exactly three times in this project's life.**

1. **Once, early: to license the cheap metrics.** Take the top three arms in one
   language and check that the scorecard's ordering matches a native CMOS
   ordering. If it does, every later decision can be made on the automatic
   scorecard. If it does not, the scorecard is wrong and everything downstream of
   it is too. This is the highest-value listening test the project will run, and
   it is the one most likely to be skipped.
2. **Final selection,** when the top two arms' `cer_above_floor` CIs overlap and
   efficiency does not break the tie.
3. **Before any public naturalness claim.**

Never for routine checkpoint ranking. It is too slow and its result arrives after
the decision.

**CMOS design.** 7-point, −3…+3, paired, same sentence, A/B order randomized per
trial, our arm against the named comparator. Native speakers of the language only.
Screening: a headphone/hearing check, two trapping pairs (a natural recording
against a 2 kHz-lowpassed copy of itself — a rater who does not prefer the natural
one is dropped), and one repeated pair for intra-rater consistency. Report N,
exclusions, platform, instructions and 95% CIs, always.

**Rater count.** Per-rating SD on a 7-point CMOS is ≈ 1.0 (assumption; measure it
in the first test and recompute). Then:

| effect | ratings, ideal | ratings, with rater+item random effects (deff ≈ 2) | practical design |
|---|---|---|---|
| CMOS 0.5 | 31 | 63 | 8 raters × 8 sentences |
| CMOS 0.3 | 87 | 174 | 14 × 14 |
| **CMOS 0.2** | 196 | **392** | **20 native raters × 20 sentences** |
| CMOS 0.1 | 785 | 1,570 | 40 × 40 — not worth it |

**The instrument has a ceiling.** Human pairwise agreement on clean modern TTS is
only **0.764** ([arXiv 2609.13150](https://arxiv.org/html/2609.13150)). A CMOS
below ~0.15 is below the resolution of listening tests, not merely below our n.
Spending on 1,570 ratings to chase CMOS 0.1 buys a number humans do not reliably
have.

**Absolute MOS (ITU-T P.808 ACR)** needs ≥ 8 ratings per stimulus as the standard
minimum; URGENT 2025 used 8 and the DNS Challenge used 10
([P.808 open implementation, arXiv 2005.08138](https://arxiv.org/pdf/2005.08138),
[URGENT 2025, arXiv 2507.11306](https://arxiv.org/pdf/2507.11306)). Release claim
only.

**MUSHRA (ITU-R BS.1534-3)** needs ≥ 15 listeners *after* post-screening, a hidden
reference and a 3.5 kHz anchor ([overview](https://www.emergentmind.com/topics/mushra-listening-test)).
Use it only when ≥ 3 systems need absolute placement on one interval scale. Not
during development.

**What can be claimed with no human eval at all:** intelligibility (CER above
floor, floor shown), speaker similarity (SECS against its floor and ceiling, plus
the published-comparable `sim_seedtts`), robustness (the Set C gap), every
efficiency number, and the data-efficiency curve. **What cannot:** "natural",
"indistinguishable from human", "better quality than X", or any MOS.

---

## 8. Statistical rigour: the required n

### 8.1 How a CER confidence interval is computed

CER is a **corpus ratio**, total errors over total reference characters
(`metrics.corpus_rate`), so the bootstrap must carry `(errors_i, length_i)` per
utterance and **recompute the ratio on each resample**. Averaging per-utterance
rates is a different quantity and is not comparable with published corpus CERs —
`metrics.py` already says this; the bootstrap has to obey it too.

- **Resample utterances**, 10,000 replicates, percentile (or BCa) interval. The
  matched-pairs bootstrap is the standard instrument here, and errors in connected
  speech are not independent events, which is exactly why ([Bisani & Ney, ICASSP
  2004](http://www-i6.informatik.rwth-aachen.de/PostScript/InterneArbeiten/Bisani_BootstrapEstimatesForConfidenceIntervalsInASRPerformanceEvaluation_ICASSP_2004.pdf)).
- **Paired across arms.** Resample utterance ids once, recompute *both* arms'
  corpus rates on that resample, take the difference. Bisani & Ney's key point is
  that the difference between two systems has a much smaller standard error than
  either absolute rate — which is why the same test set that gives a ±1.8 point
  absolute CI can still resolve a 1.25 point difference.
- **Block by speaker** where there is one. For SECS, resample *reference clips*
  and then utterances within them; utterances sharing a reference are correlated
  ([blockwise bootstrap](https://www.researchgate.net/publication/338115886_Statistical_Testing_on_ASR_Performance_via_Blockwise_Bootstrap)).
- **Cross-check with MAPSSWE** (NIST SCTK, shipped with Kaldi) on WER. It pools
  error counts across segments so long utterances weigh more; a Wilcoxon
  signed-rank over per-utterance differences weighs every utterance equally. Where
  the two disagree, one long utterance is driving the result, and that is worth
  knowing ([MAPSSWE](https://1cademy.com/node/the-matched-pair-sentence-segment-word-error-mapsswe-test/XcGT42kzAqCeSmFBxTXL)).

### 8.2 The numbers

Assumptions, to be replaced by the first pilot's measured values: per-utterance
CER SD **σ = 0.10** at a corpus CER around 5% (the distribution is heavy-tailed —
most utterances near zero, a few near 1.0 from truncations), and arm-to-arm
correlation **ρ = 0.6**, giving `σ_d = σ√(2(1−ρ)) = 0.0894`. Power 80%, α = 0.05
two-sided.

| n per lang per arm | absolute CER 95% CI | MDE, no correction | MDE, Holm over 6 arms × 2 langs |
|---|---|---|---|
| **118** (today, FLEURS dev parallel) | ±1.80 pts | **2.31 pts** | **3.05 pts** |
| 200 | ±1.39 | 1.77 | 2.20 |
| **400** (recommended) | **±0.98** | **1.25** | **1.66** |
| 600 | ±0.80 | 1.02 | 1.35 |
| 1000 | ±0.62 | 0.79 | 0.98 |

Required n for a target effect:

| effect to detect | n, no correction | n, Holm over 12 tests |
|---|---|---|
| 0.5 pt CER | 2,512 | 3,875 |
| 1.0 pt | 628 | 969 |
| **1.5 pt** | 279 | **489** |
| 2.0 pt | 157 | 242 |
| 3.0 pt | 70 | 108 |

SECS, with σ = 0.08 and ρ = 0.5 (`σ_d = 0.080`):

| n | absolute 95% CI | MDE, no correction | MDE, Holm ×12 |
|---|---|---|---|
| 118 | ±0.0144 | 0.0206 | 0.0256 |
| **400** | **±0.0078** | **0.0112** | **0.0139** |
| 600 | ±0.0064 | 0.0091 | 0.0114 |

— and these are **before** the pool-4 design effect, which at ICC 0.1 reduces
n = 400 to an effective 37. See **B2**.

**B15 — n = 400 per language per arm on Set A.** It resolves a 1.25-point paired
CER difference uncorrected and 1.66 points under Holm, and gives ±1.0 point on the
absolute. That is the right resolution for this project: the Tamil gap we are
trying to close is measured in whole points of perplexity and tens of points of
CER, not fractions. Going to 1,000 to chase 0.8 points costs 2.5× the synthesis
for a resolution nothing in the decision needs.

**Set A cannot supply 400 today.** The 6-way parallel FLEURS intersection on dev
is 118. Three ways out, in order of preference:

1. **Build Set A from `dev + test`.** FLEURS test is larger than dev; the union
   intersection should land in the 400–600 range. Costs nothing but a `--split
   dev+test` flag, and keeps the strict 6-way parallelism that §3 of the findings
   doc depends on.
2. **Keep the parallel core, relax the tail.** Define `A-par` (the strict 6-way
   intersection, used for cross-language comparisons) and `A-full` (per-language,
   up to 400, used for the powered within-language arm comparison). Cross-language
   claims use `A-par`; arm-vs-arm claims use `A-full`. Both carry floors.
3. Add the Cross-Lingual Indic Test Set's monolingual direction, which has
   759–1,179 utterances per language — but its clips are not parallel across
   languages and it brings its own speakers, so it supplements rather than
   replaces.

### 8.3 Multiple comparisons

With 6 arms × 6 languages × 2 test sets × 3 metrics there are 216 ways to find a
winner, and at α = 0.05 the chance of at least one false positive is essentially 1.

**B16 — one pre-declared primary family; everything else gets a CI and no p-value.**

- **Primary endpoint:** `cer_above_floor` on Set A, in **Hindi and Tamil only**.
  Hindi because it is nearly at parity with English (ppl 68.3 vs 60.9) and is
  where a regression would show; Tamil because it is 4.7× worse in perplexity and
  worst in Qwen3 fertility (10.13 tokens/word), so it is where an improvement
  would show. The family is `{arms} × {hi, ta}` — 12 tests at 6 arms — corrected
  with **Holm–Bonferroni** at α = 0.05.
- **Co-primary, not corrected into the same family:** `rtf_stream` and
  `ttfa_p95`. These are deterministic measurements with a reproducibility band,
  not samples from a population; they get the §5.2 spread check, not a p-value.
- **Everything else** — bn/te/mr, WER, Set C, SECS, Tier 2 — is reported with a
  95% bootstrap CI and **no significance test**. If the exploratory table must be
  tested, use Benjamini–Hochberg at q = 0.10 and label it exploratory.
- **Fix n in advance.** Scoring an arm, looking at the result, and deciding to
  synthesize more utterances inflates the error rate without leaving a trace. The
  n comes from the table in §8.2 before the first wav is written.

---

## 9. The harness extension

Six new modules and four edits to existing ones. Each says which file it extends
and what it must not duplicate.

### 9.1 `indic/eval/stats.py` — new

The statistics every other module calls. Nothing else computes a CI.

```
bootstrap_ratio_ci(pairs, n_boot=10000, alpha=0.05, blocks=None)
    pairs: [(errors, length), ...]; resamples and recomputes the ratio.
    blocks: optional [(block_key, index), ...] for the blockwise bootstrap.
paired_difference_ci(pairs_a, pairs_b, ids, n_boot=10000, alpha=0.05, blocks=None)
    one resample of ids; both corpus rates recomputed on it; returns
    (delta, lo, hi, p_two_sided).
holm(pvalues, alpha=0.05)        -> [(index, adjusted_p, reject)]
benjamini_hochberg(pvalues, q)   -> same shape
required_n(delta, sd, rho, k_tests, power=0.80, alpha=0.05) -> int
mde(n, sd, rho, k_tests, power=0.80, alpha=0.05) -> float
design_effect(cluster_sizes, icc) -> float
```

CLI — a calculator, so the n in a plan is never guessed:

```
python -m indic.eval.stats --required-n --delta 0.015 --sd 0.10 --rho 0.6 --k-tests 12
python -m indic.eval.stats --mde --n 400 --sd 0.10 --rho 0.6 --k-tests 12
python -m indic.eval.stats --from-scores runs/kitten2-A-hi/scores.*.csv --metric cer \
    --estimate-sd      # measure sigma and rho from a pilot instead of assuming them
```

The `--from-scores` path is the point: §8.2's σ and ρ are assumptions, and the
first real run replaces them.

### 9.2 `indic/eval/compare.py` — new. Extends `score.py`.

Reads two or more run directories' `scores.*.csv`, pairs rows by `id`, and answers
"is this difference real".

```
python -m indic.eval.compare \
    --baseline runs/kitten2-A-hi \
    --arm sarvam=runs/sarvam-A-hi --arm qwen3=runs/qwen3ctrl-A-hi \
    --metric cer_above_floor --metric secs_wavlm \
    --block-by ref_wav \
    --bootstrap 10000 --alpha 0.05 --correction holm \
    --out runs/compare-hi.json
```

Behaviour that is not optional:
- Refuses to compare runs whose `asr_model`, `asr_decoding`, `speaker_embedder`,
  `testset`, `split` or normalizer version differ. Those columns are already in
  `score.py`'s `COLUMNS`; this is the first thing that uses them.
- Refuses rows where either side has `error` set, and reports how many it dropped
  — a dropped failure silently favours the arm that failed.
- Prints the MDE for the n it actually has, next to the result. A null result at
  n = 118 means "we could not see 2.3 points", not "they are the same", and the
  output must say so in those words.
- `--block-by ref_wav` switches on the blockwise bootstrap for SECS.

### 9.3 `indic/bench/efficiency.py` — new. Sibling of `fit_config.py`.

`fit_config.py` measures *training* memory; this measures *inference* cost. It
reuses `fit_config`'s discipline (measure, don't predict; report reserved, not
allocated; an OOM row is a statement about this allocator state) and
`indic/hpc/capability.py` for dtype and attention selection.

```
python -m indic.bench.efficiency --model <spec> --lang hi --mode single \
    --bucket medium --repeats 5 --warmup 3 --out indic/data/efficiency.json
python -m indic.bench.efficiency --model <spec> --lang hi --mode sweep \
    --concurrency 1,2,4,8,16,32,64 --slo-ttfa-ms 300 --slo-rtf 1.0 \
    --out indic/data/efficiency_sweep.json
```

| flag | default | meaning |
|---|---|---|
| `--mode` | `single` | `single` \| `stream` \| `batch` \| `sweep` |
| `--bucket` | `all` | `short` \| `medium` \| `long` \| `all` |
| `--concurrency` | `1` | comma list; `sweep` walks it |
| `--repeats` | `5` | fresh subprocess each |
| `--warmup` | `3` | untimed, discarded |
| `--lock-clocks` | on | `nvidia-smi -pm 1 -lgc -lmc`; records what it applied |
| `--require-exclusive` | on | fails if NVML sees another compute process |
| `--energy` | on when NVML has the counter | board-level joules |
| `--gpu-price` | `0.0` | $/GPU-hour, an input, echoed into the JSON |
| `--max-duration-ratio` | `2.0` | hard cap; over it counts as a runaway |
| `--score-gate` | none | a `scores.csv` whose CER the rows are checked against |
| `--spread-threshold` | `0.05` | above it, rows are marked `reportable: false` |

Output JSON carries per-utterance rows and a summary, plus the full environment
block (device, driver, CUDA, clocks, power limit, dtype, attention impl, compile /
CUDA-graph / vLLM flags) and the `git_state` + `library_versions` that
`run_eval.py` already knows how to collect — **import them from `run_eval`, do not
re-implement them.**

### 9.4 `indic/bench/data_curve.py` — new. Extends `indic/train/prepare_shards.py`.

```
python -m indic.bench.data_curve plan \
    --shards data/shards --fractions 1/32,1/16,1/8,1/4,1/2,1 \
    --stratify lang,speaker --anchor-fractions 1/8,1 --seeds 3 \
    --steps 20000 --out indic/data/data_curve_plan.json

python -m indic.bench.data_curve manifest --plan indic/data/data_curve_plan.json \
    --fraction 1/8 --seed 0 --out data/subsets/f8-s0.jsonl

python -m indic.bench.data_curve fit --scores 'runs/curve/*/scores.*.csv' \
    --metric cer_above_floor --out indic/data/data_curve.json
```

- `plan` writes the nested stratified assignment (the crc32 key per utterance) and
  the run matrix, and **asserts nesting**: every fraction's id set is a subset of
  the next larger one. A plan that fails that assertion is a bug, not a result.
- `manifest` emits a shard manifest `indic/train/data.py` can consume unchanged.
- `fit` does the power-law fit, the profile interval on α, and reports `h*` — the
  smallest fraction whose CI overlaps the full-data CI — and refuses to
  extrapolate beyond 2× the largest measured `h`.

### 9.5 `indic/eval/external.py` — new. Extends `testsets.py`.

Adapters that turn a public benchmark into `testsets.Record` objects, so
`run_eval.py` synthesizes them with no changes at all, and that emit results in
the format each external scorer expects.

```
python -m indic.eval.external --benchmark seedtts-en --out runs/seedtts-en/manifest.jsonl
python -m indic.eval.external --benchmark cv3-ttfa-en --limit 50 --out ...
python -m indic.eval.external --benchmark indic-xlingual --lang ta --direction indic2indic --out ...
python -m indic.eval.external --benchmark librispeech-pc --out ...
python -m indic.eval.external --emit seedtts --run runs/seedtts-en   # writes meta.lst + wav dir
```

Each adapter records its source URL and commit/revision in the manifest, because a
public test set that moves silently is the classic way to lose a comparison.

### 9.6 `indic/eval/scorecard.py` — new. The analogue of `preflight.py`'s report.

Rolls every arm's `scores.*.csv` plus `efficiency.json` into **the one page**, in
Tier order, with gates first, CIs everywhere, and a `REPORTABLE` / `NOT
REPORTABLE` verdict per row driven by the Tier 0 gates and the efficiency spread
check.

```
python -m indic.eval.scorecard \
    --arm sarvam=runs/sarvam --arm qwen3=runs/qwen3 --baseline kitten2=runs/kitten2 \
    --efficiency indic/data/efficiency.json \
    --primary-langs hi,ta --correction holm \
    --out docs/indic/SCORECARD.md --json indic/data/scorecard.json
```

It must refuse to print a Tier 1 number whose Tier 0 gate failed, and must print
the MDE beside every comparison. A scorecard that shows a 0.4-point CER win
without showing that the MDE is 1.66 points is worse than no scorecard.

### 9.7 Edits to existing files

| File | Change |
|---|---|
| `indic/eval/testsets.py` | add `set_e()` and `"E"` to `TESTSETS` (§5.3); raise `REFERENCE_POOL` 4 → 32 (**B2**); add `--split dev+test` so `parallel_corpus` can union splits (§8.2) |
| `indic/eval/metrics.py` | add `SeedTTSWavLMEmbedder` (**B1**); add `script_fraction(text, lang)` for the Tier 0 script gate; add `secs_ceiling_floor(lang, split)` returning the same-speaker and different-speaker cosines from FLEURS |
| `indic/eval/score.py` | new columns `sim_seedtts`, `script_ok`, `runaway`; `--speaker-embedder seedtts`; call `stats.bootstrap_ratio_ci` in `summarize()` so the summary prints a CI instead of a bare rate |
| `indic/eval/asr.py` | add a `Qwen3ASR` backend behind `--asr qwen3` as a cross-check only (**B3**). Do not change the default |
| `indic/bench/teacher_forced_loss.py` | report `audio_bits_per_second = nats_per_token × frame_rate_hz / ln 2` alongside ppl (§10, item 8) |
| `indic/preflight.py` | two new stages after `fit`: `serve` (`efficiency_single`, `efficiency_sweep`, `needs_gpu=True`) and `report` (`scorecard`). `efficiency_*` must hold the existing GPU lock — two timing runs at once produce two wrong answers, which is the same reason the lock exists for memory |

### 9.8 What the whole scorecard costs per arm

| stage | work | cost |
|---|---|---|
| Set A synthesis | 6 langs × 400 utt, ~8 s mean audio, RTF 0.78 | ≈ 4.2 GPU-h |
| Set C synthesis | ~6 × 120 utt | ≈ 0.5 GPU-h |
| Set E efficiency | 60 utt × 5 repeats × 3 buckets + concurrency sweep | ≈ 1.5 GPU-h |
| Scoring | IndicConformer + WavLM ×2, shardable on CPU | ≈ 4 CPU-h |

**≈ 6 GPU-hours per arm**, against the 25–40 A100-hours the findings doc estimates
for one training run. ~20% overhead is the budget this design was written to. If
the benchmark grew to 30 GPU-hours per arm it would stop being run, and that is
the failure mode §10 exists to prevent.

---

## 10. What NOT to measure, and why

A bloated benchmark nobody runs is worse than a small one that gates every
decision. Each of these was considered and rejected.

1. **UTMOS / UTMOSv2 / DNSMOS / NISQA / SCOREQ as any kind of score.** On clean
   TTS the best of them (0.528 pairwise) does not beat clip duration (0.524), and
   optimising one collapses human preference ([arXiv
   2609.13150](https://arxiv.org/html/2609.13150)). UTMOS goes *negative* on
   in-the-wild MOS ([arXiv 2506.19441](https://arxiv.org/html/2506.19441v1)). Keep
   one as a same-arm tripwire; nothing more. (**B6**)
2. **A calibrated composite of MOS predictors.** The 2609.13150 authors show a
   composite beats the best single metric — but it needs ~1,000 in-domain human
   pairwise judgements to fit, per language. If we had 1,000 Hindi pairwise
   judgements we would not need the composite.
3. **MCD, PESQ, STOI, mel-distance on TTS output.** These need an aligned
   reference, and zero-shot TTS has none — the model is *supposed* to choose a
   different prosody. TTSDS2 finds their negative correlations consistent with
   oversmoothing: averaged outputs score well and are rated badly. MCD stays where
   it belongs, in `codec_roundtrip.py`, where the reference genuinely is the same
   waveform.
4. **F0 RMSE / duration RMSE against the ground-truth recording.** Same reason:
   there is no correct prosody for a sentence the model has never heard read.
5. **The mean of per-utterance CERs.** `metrics.py` already says why; the point
   here is that the bootstrap must obey the same rule, resampling `(errors,
   length)` and recomputing the ratio.
6. **Seed-TTS-eval `test-zh` and `test-hard`.** 2,020 Mandarin samples and a hard
   subset, a second ASR stack, in a language we do not ship. (**B9**)
7. **TTS Arena / Artificial Analysis Elo during development.** (**B10**)
8. **Teacher-forced perplexity across arms with different codecs.** This one is
   subtle and it would have produced a confident wrong answer. Our gate masks loss
   to the target *audio* segment over the 6,561 S3 rows, so per-token ppl **is**
   comparable across text tokenizers — the Sarvam and Qwen3 arms share the audio
   vocabulary. It is **not** comparable across *codecs*, and the project is
   choosing between codecs. Normalize instead:

   ```
   audio_bits_per_second = nats_per_token * frame_rate_hz / ln(2)
   ```

   | | ppl | bits/s |
   |---|---|---|
   | S3 @ 25 Hz, en (measured) | 60.9 | **148.2** |
   | S3 @ 25 Hz, hi | 68.3 | 152.3 |
   | S3 @ 25 Hz, ta | 287.2 | 204.1 |
   | a hypothetical 50 Hz codec at 3.0 nats | 20.1 | **216.4** |
   | a hypothetical 12.5 Hz codec at 4.110 nats | 60.9 | **74.1** |

   The 50 Hz codec has a 3× better perplexity and is 46% *worse* per second of
   audio. Ranking codecs by ppl would have chosen it.
9. **Energy or RTF on the dev laptop GPU.** Board power is not comparable across a
   laptop card with a display attached. The laptop is for memory fitting
   (`fit_config.py`) and correctness, not for timing. (§5.2)
10. **`kv_bytes_per_token` as a differentiator.** The arithmetic in §5.4 says KV
    exhausts at ~1,580 concurrent streams on an 80 GB A100, so it will never be the
    binding constraint at our sequence lengths. Keep it as a Tier 2 consistency
    check (derived must equal measured); do not optimise against it.
11. **Per-language significance tests on bn/te/mr.** Reported with CIs, not tested.
    Adding them triples the correction family and costs ~0.5 points of MDE on the
    primaries for languages that are not the decision. (**B16**)
12. **"Best on dev" checkpoint selection in the data-efficiency curve.** It is a
    selection over noise that biases the noisier small-data arms upward and would
    manufacture a flatter curve than the truth. (**B13**)
13. **Anything not in §2.** The scorecard is one page by construction. A new metric
    enters only by displacing one that is there, and only with a stated decision it
    would change.

---

## 11. Honest gaps

- **σ and ρ in §8.2 are assumptions.** The whole n table moves with them.
  `stats.py --from-scores --estimate-sd` exists precisely so the first pilot
  replaces them, and the n should be re-derived before the comparison matrix is
  committed to.
- **Set A cannot reach n = 400 today** and the fix is unverified: nobody has
  checked how large the 6-way FLEURS parallel intersection is on `test`, or on
  `dev + test`. That is a ten-minute check and it gates §8.
- **TTSDS2's 14 languages are not enumerated in the paper** we read. Whether it
  covers hi/bn/ta/te/mr is unknown, and B8 is contingent on it.
- **CV3-Eval has no Indic split.** Its 9 languages are zh/en/ja/ko/de/fr/ru/it/es;
  a 2026 extension adds ar/id/pt/th/vi/ms/tl, still no Hindi
  ([CV3-Eval](https://github.com/QwenAudio/CV3-Eval),
  [arXiv 2607.23938](https://arxiv.org/html/2607.23938v1)). We use it for the
  50-prompt English TTFA slice and nothing else.
- **No energy number is measurable on this machine.** NVML's cumulative energy
  counter is not present on every card, and the dev laptop is excluded anyway by
  §5.2. Energy becomes real when the cluster allocation does.
- **The Seed-TTS-eval speaker-verification checkpoint is distributed via Google
  Drive**, with the `bezzam/...` Hub mirror as the practical route. Pin a revision
  and record it; an unpinned mirror is a silent way to lose every SIM number.
- **Nothing here is run.** This is a specification. The first thing it should
  produce is the pilot that replaces §8.2's assumptions with measurements.

---

## 12. Sources

Metrics and their failure modes —
[TTSDS2 (ICLR 2026)](https://arxiv.org/abs/2506.19441) ·
[TTSDS2 full text](https://arxiv.org/html/2506.19441v1) ·
[ttsds/ttsds](https://github.com/ttsds/ttsds) ·
[Limits of reference-free speech quality metrics (2026)](https://arxiv.org/html/2609.13150) ·
[UTMOS](https://arxiv.org/pdf/2204.02152) ·
[DNSMOS](https://arxiv.org/abs/2010.15258) ·
[ProxyMOS](https://arxiv.org/html/2610.00419) ·
[UtterTune / UTMOSv2 on Walloon](https://arxiv.org/pdf/2508.09767) ·
[Codec speech-quality metric comparison](https://arxiv.org/html/2509.24457v1) ·
[SQuId multilingual naturalness](https://arxiv.org/pdf/2210.06324)

Benchmarks —
[Seed-TTS-eval](https://github.com/BytedanceSpeech/seed-tts-eval) ·
[Seed-TTS paper](https://arxiv.org/pdf/2406.02430) ·
[Seed-TTS-eval in EvalScope](https://evalscope.readthedocs.io/en/latest/benchmarks/seed_tts_eval.html) ·
[Open TTS Leaderboard blog (2026-09-30)](https://huggingface.co/blog/open-tts-leaderboard) ·
[Open TTS Leaderboard repo](https://github.com/huggingface/open-tts-leaderboard) ·
[Open TTS Leaderboard space](https://huggingface.co/spaces/hf-audio/open_tts_leaderboard) ·
[CV3-Eval](https://github.com/QwenAudio/CV3-Eval) ·
[F5-TTS](https://arxiv.org/html/2410.06885v3) ·
[F5-TTS (ACL 2025)](https://aclanthology.org/2025.acl-long.313.pdf)

Indic —
[Cross-Lingual TTS for Indian Languages (2026)](https://arxiv.org/html/2610.14826) ·
[Cross-Lingual Indic Test Set](https://github.com/Cross-Lingual-TTS-for-Indian-Languages/Cross-Lingual-TTS-Indic-Test-Set) ·
[IndicVoices-R (NeurIPS 2024 D&B)](https://arxiv.org/pdf/2409.05356) ·
[indicvoices_r on the Hub](https://huggingface.co/datasets/ai4bharat/indicvoices_r) ·
[IndicF5](https://huggingface.co/ai4bharat/IndicF5) ·
[Hindi flow-matching distillation (2026)](https://arxiv.org/pdf/2607.18662) ·
[PSP Indic accent benchmark (2026)](https://arxiv.org/html/2604.25476)

Efficiency —
[Chatterbox-Flash](https://arxiv.org/pdf/2605.30748) ·
[chatterbox-flash model card](https://huggingface.co/ResembleAI/chatterbox-flash) ·
[Unified ASR/TTS/VC autoregressive transformers — concurrency sweep](https://arxiv.org/pdf/2601.10770) ·
[Deepgram TTS latency](https://developers.deepgram.com/docs/text-to-speech-latency) ·
[Inworld on TTFB reporting](https://inworld.ai/resources/tts-latency-ttfb-p99) ·
[The Price of Prompting — energy tooling](https://arxiv.org/pdf/2407.16893) ·
[NeuralWatt energy methodology](https://docs.neuralwatt.com/energy-methodology.md)

Data efficiency —
[WavTTS (585 h vs 100K h)](https://arxiv.org/html/2606.03455v1) ·
[BASE TTS](https://arxiv.org/pdf/2402.08093) ·
[Zero-Voice](https://www.techrxiv.org/users/882196/articles/1260796-zero-voice-a-low-resource-approach-for-zero-shot-text-to-speech) ·
[BnTTS](https://aclanthology.org/2025.findings-naacl.279.pdf)

Statistics and listening tests —
[Bisani & Ney, bootstrap CIs for ASR (ICASSP 2004)](http://www-i6.informatik.rwth-aachen.de/PostScript/InterneArbeiten/Bisani_BootstrapEstimatesForConfidenceIntervalsInASRPerformanceEvaluation_ICASSP_2004.pdf) ·
[Blockwise bootstrap for ASR](https://www.researchgate.net/publication/338115886_Statistical_Testing_on_ASR_Performance_via_Blockwise_Bootstrap) ·
[MAPSSWE](https://1cademy.com/node/the-matched-pair-sentence-segment-word-error-mapsswe-test/XcGT42kzAqCeSmFBxTXL) ·
[ITU-T P.808 open implementation](https://arxiv.org/pdf/2005.08138) ·
[URGENT 2025 multilingual P.808](https://arxiv.org/pdf/2507.11306) ·
[MUSHRA overview](https://www.emergentmind.com/topics/mushra-listening-test)
