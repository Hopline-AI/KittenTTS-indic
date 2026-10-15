# Evaluation and baselines

## Baselines

| Model | Params | Langs | Cloning | Weights |
|---|---|---|---|---|
| [IndicF5](https://huggingface.co/ai4bharat/IndicF5) | ~336M | 11 | Yes, zero-shot | MIT, open. **Main cloning baseline** |
| [Indic Parler-TTS](https://huggingface.co/ai4bharat/indic-parler-tts) | ~0.9B | 21 | No (text description) | Apache-2.0, open |
| [svara-TTS v1](https://huggingface.co/kenpath/svara-tts-v1) | 3B (Orpheus) | 18 + Indian English | Clone model in beta | Apache-2.0, open |
| [Veena](https://huggingface.co/maya-research/Veena) | 3B (Llama + SNAC) | hi, en, code-mixed | No | Apache-2.0, open |
| [Indic-Mio](https://huggingface.co/MeghanaKap/miomio_cp1_public) | 0.6B | 22 + en | Yes | Apache-2.0, open |
| [Dhee-Indic-F5](https://huggingface.co/dheeyantra/dhee-indic-f5) | ~336M | Indic | Yes | Open |
| [Sarvam Bulbul v3](https://sarvam.ai/blogs/bulbul-v3) | ? | 11 | Enterprise only | API |
| ElevenLabs v3, Google Chirp 3 Instant Custom Voice | closed | many | Yes | API |
| Gnani Vachana, BharatGen Sooktam-2, Navana Bodhi | ? | 10–12 | varies | API or restricted |

## Metrics

- **Intelligibility: CER (primary)**
  - ASR: [IndicConformer-600M-multilingual](https://github.com/AI4Bharat/IndicConformerASR) (MIT, all 22 languages, RNNT decoding).
  - Cross-checks: Whisper-large-v3 (hi/bn/ta/te/mr/en) and MMS-1B-all.
  - Always also report the ground-truth recording's CER through the same ASR, as the floor.
  - Sarvam scored Bulbul with Gemini 2.5 Pro, so their numbers are not comparable to ours.
- **Speaker similarity:** WavLM-large + ECAPA cosine (the Seed-TTS-eval convention).
- **Naturalness:** UTMOSv2 and NORESQA-MOS, for **relative ranking only**. They were trained on English MOS and [transfer poorly](https://arxiv.org/pdf/2509.04830).
- **Accent/prosody:** [PSP benchmark](https://github.com/praxelhq/psp-eval). It scores retroflex collapse, aspiration, vowel length and Tamil zha, with golden sets for hi/te/ta. Add F0 RMSE and duration ratio.
- **Speed:** real-time factor (RTF) and first-chunk latency on fixed hardware.
- **Human:** CMOS against IndicF5 and Bulbul, plus MUSHRA-S for similarity, with at least 20 native raters per language.

## Test sets (~300 utterances each per language)

- **A, read:** IndicVoices-R test split, seen and unseen speakers. Clone from a 3–10 s clip of the same speaker, with a different target sentence.
- **B, expressive:** Rasa held-out.
- **C, robustness:** numbers, dates, currency, abbreviations, sentences longer than 40 words.
- **D, code-mixed:** hi-en, ta-en and so on, in native script and in Roman script. Normalize ASR output to one script with IndicXlit before computing CER. There is no standard benchmark for this.
- **E, accent:** the PSP golden sets.

## Harness on the HPC

- One Slurm array job per (model, language, test set), each writing `wav + meta.json`.
- A single scoring job that collects everything into a parquet file.
- Pin seeds and the versions of all ASR and MOS models.
- Apply NFC normalization to all text.

Also run an **S3 codec round-trip** on real Indic audio (encode to S3 tokens, decode with s3gen, compute CER). This checks the codec isn't the bottleneck before we train.
