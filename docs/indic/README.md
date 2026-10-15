# KittenTTS-indic: research and plan

Working notes for adding Indian-language support to KittenTTS 2. This is a brainstorming branch, so edit freely.

| File | What's in it |
|---|---|
| [tokenizer.md](tokenizer.md) | Should we use the Sarvam tokenizer? Fertility numbers, vocab-extension methods, precedent |
| [datasets.md](datasets.md) | Open Indic speech datasets, licenses, recommended training mix, text normalization tools |
| [evaluation.md](evaluation.md) | Baseline models, metrics, ASR choice, benchmark harness for the HPC |
| [finetuning.md](finetuning.md) | Weights license, what is trainable, codec suitability, current Indic bugs in the code |
| [training.md](training.md) | The training plan: data prep, stages, QAT, compute estimate, open questions |
| [sarvam-experiment.md](sarvam-experiment.md) | **The main experiment: replace Qwen3 with the Sarvam-1 tokenizer entirely** |

## Summary (2026-10-15)

- **Decision (2026-10-15):** the experiment uses the **Sarvam-1 tokenizer only**, with no Qwen3 text vocab, and keeps a Qwen3 fine-tune as the control arm. See [sarvam-experiment.md](sarvam-experiment.md).
- **Tokenizer research:** Qwen3 splits Indic text into 5–11 tokens per word, while Sarvam-1 needs about 2. Even so, no speech-LM TTS model has swapped its text tokenizer, and Sarvam-1's tokenizer is non-commercial. Plan: fine-tune with Qwen3 first. If Tamil and Telugu still lag, add vocab rows at the end of the table using our own tokenizer.
- **License:** the Stellon Labs Community License allows fine-tunes and releasing derivative weights. Commercial use requires registering, and the license ends above $1M revenue or $1M funding.
- **Trainable:** the HF repo ships a bf16 `lm/model.safetensors` (Qwen3ForCausalLM). Its linear layers must stay ternary, so training needs quantization-aware training (QAT).
- **Data:** about 800–900 h of clean, CC-BY data exists for hi/bn/ta/te/mr: SYSPIN, Rasa, IndicTTS and IndicVoices-R.
- **Eval:** CER from IndicConformer, WavLM-ECAPA speaker similarity, and the PSP accent benchmark. Baselines are IndicF5, Indic Parler-TTS and svara-TTS.

## Roadmap

1. Quick fixes in the text pipeline: split sentences on `।`/`॥`, and normalize Indic numbers.
2. Benchmark harness on the HPC. Score stock KittenTTS 2 and the baselines, plus an S3 codec round-trip test on Indic audio.
3. Training code: `spk_proj` in the training graph, a loss on audio tokens only, ternary QAT.
4. Pilot: a Hindi-only fine-tune on about 50 h.
5. Full fine-tune with 5 languages, keeping some English data in the mix.
6. Only if needed: vocab extension.
