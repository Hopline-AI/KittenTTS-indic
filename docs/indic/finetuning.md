# Fine-tuning feasibility, license, and current Indic gaps

## Weights license

Source: the [Stellon Labs Community License](https://huggingface.co/KittenML/kitten-tts-2/blob/main/LICENSE.md). This is a summary, so read the original before relying on it.

- **Fine-tunes, LoRAs and derivative releases are allowed.** §2 and §3 grant the right "to use, reproduce, distribute, and create Derivative Works", and §5 says derivative works include "fine tune" and "low-rank adaptation" models.
- **Commercial use:** you must register with Stellon Labs. The license terminates if you or your affiliates have **more than $1M annual revenue or more than $1M total cumulative funding** (all revenue, not just revenue from the model). Above that, you need a separate license.
- **Attribution (§4a):**
  - Ship the license.
  - Include a NOTICE file with the Stellon copyright line, **stating that the materials were changed and how**.
  - Display "Powered by Stellon Labs" prominently.
- **No trademark rights:** don't use Kitten or Stellon in the model's name.
- **§4b:** the model or its outputs may not be used to train or distill *other* models. Fine-tunes and LoRAs of this model are allowed.
- `speaker/LICENSE` (pyannote) is a separate license to check.

## What is in the HF repo

- `lm/model.safetensors`: 3.47 GB, **bf16, trainable**.
- `config.json`: `Qwen3ForCausalLM` with:
  - 28 layers, hidden 2048, FFN 6144
  - 16 query heads / 8 KV heads, head_dim 128
  - `tie_word_embeddings: true`, vocab 158,241, `audio_id_base` 151,675
- The packed checkpoint unpacks bit-exactly with `ternary.unpack_state_dict` (`ternary.py:76`).
- Loading is `AutoModelForCausalLM.from_config` + `load_state_dict(strict=False)` (`model.py:217-236`).

**Ternary weights:** q/k/v/o/gate/up/down in all 28 layers. Each 128-wide group holds values {-s, 0, +s} with its own bf16 scale (`ternary.py:28,36-48`). This differs from BitNet b1.58, which uses one scale per tensor. Activations are **not** quantized.

**Full precision:**
- `embed_tokens`, which is also the tied `lm_head`
- all RMSNorms, including q_norm/k_norm
- `model.norm`
- `spk_proj`

**Not yet in the training graph:**
- `spk_proj` = `Linear(512→2048)` + `LayerNorm` (`speaker.py:27-44`). It is injected at position 0 by a forward pre-hook during prefill only (`speaker.py:47-77`).
- The prompt layout is in `prompt.py`.

## Audio codec

- The vendored code is Chatterbox 0.1.7's S3Gen with CosyVoice's S3TokenizerV2 (25 Hz, 6561 tokens) (`s3gen.py:4-5,413,426`).
- The tokenizer was trained without Indic languages ([CosyVoice](https://arxiv.org/abs/2407.05407)).
- In practice, Chatterbox-Multilingual speaks Hindi with it. [chatterbox-indic-lora](https://huggingface.co/reenigne314/chatterbox-indic-lora) kept the codec frozen for 8 Indic languages and blamed CER differences on data, not the codec.
- **Verify with a round-trip test on ta/te audio before training.**

## Recipes to adapt

- [stlohrey/chatterbox-finetuning](https://github.com/stlohrey/chatterbox-finetuning): same codec and vocoder family, closest match.
- [Chatterbox Finnish train.py](https://huggingface.co/RASMUS/Chatterbox-Finnish/blob/main/train.py)
- [chatterbox-indic-lora](https://huggingface.co/reenigne314/chatterbox-indic-lora): includes vocab extension and Brahmic warm-start.
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice), see `examples/`.
- [HF: Fine-tuning LLMs to 1.58bit](https://huggingface.co/blog/1_58_llm_extreme_quantization): BitLinear with a straight-through estimator and lambda warmup.
- No public training code for ternary *speech* LMs was found.

## Current Indic bugs in the code

- **Danda doesn't end a sentence.** The splitter only breaks on `.!?` (`text.py:280`). The clause fallback only looks for `", "`, `"; "` and `" — "` (`text.py:298-300`). Otherwise it cuts hard at 380 characters, which can land mid-word.
- **Oversized chunks.** `_merge_short` allows up to 1.4× the budget (`text.py:327`). A 505-character Hindi text came back as one chunk.
- **Wrong punctuation after the danda.** `।` is missing from `_ALREADY_TERMINATED` (`text.py:134`) and `_TERMINAL` (`text.py:281`). The output becomes `"…है।."`, and chunk ends get `"।,"`.
- **Normalizer is English-only.** Numbers in Hindi text are read out in English. The docs tell users to pass `normalize=False`.
- `preprocess.py:787` has the same `.!?,;:` assumption.
- **Fix:** add `।` and `॥` to `_SENT_END`, `_ALREADY_TERMINATED` and `_TERMINAL`, and add Indic number normalization.
