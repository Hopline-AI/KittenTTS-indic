# Tokenizer: Sarvam or Qwen3?

## Fertility (measured)

Tokens per word on one sentence per language:

| | Qwen3 (current, vocab 151,669) | sarvamai/sarvam-1 (vocab 68,096) |
|---|---|---|
| Hindi | 6.00 | 2.11 |
| Bengali | 5.11 | 1.67 |
| Tamil | 9.25 | 2.00 |
| Telugu | 10.75 | 2.50 |
| Marathi | 5.20 | 1.50 |
| English | 1.08 | 1.17 |

Per character, Qwen3 needs about 1.0 token for Hindi, 1.4 for Telugu and 0.29 for English.

## Why it is not a drop-in swap

The model has one shared vocab (`kittenml/kittentts2/tokens.py`):

```
[0, 151675)            Qwen3 BPE text + named specials
[151675, +6561)        S3 audio codec tokens (audio_id_base = 151675)
end                    final_seg / reference_* marker rows   (total vocab 158,241)
```

- Text ids index `embed_tokens`, and that table is tied to `lm_head` (`model.py:231`). A different tokenizer means new ids with untrained embeddings.
- New rows have to go at the end of the table, after the marker rows. Inserting them earlier would shift every audio id. The tokenizer then needs a wrapper that maps new pieces to `orig_vocab_end + k`.
- Since `lm_head` is tied, new rows also become output logits. `logits.py` already restricts sampling to the audio range.

## Sarvam tokenizers

- **sarvam-1:** 68k, Llama-style SentencePiece, covers en plus 10 Indic languages, fertility 1.4–2.1. It is under the **Sarvam non-commercial license**, which likely covers the tokenizer files too. [HF](https://huggingface.co/sarvamai/sarvam-1)
- **sarvam-m:** Mistral-Small-3.1 base, so it uses Mistral's Tekken tokenizer. Apache 2.0.
- **sarvam-translate:** Gemma3-4B base, so it uses Gemma's 262k SentencePiece tokenizer.

## Methods for extending a tokenizer after pretraining

- **Embedding init:** FOCUS beats mean-of-subtokens/FVT, WECHSEL and OFA right after init, but they converge to similar quality once training starts. ([FOCUS](https://arxiv.org/pdf/2305.14481), [comparison](https://arxiv.org/html/2407.05841v2))
- **ZeTT** (zero-shot tokenizer transfer) needs a hypernetwork trained for the base model, and none exists for this model. ([paper](https://arxiv.org/abs/2405.07883))
- **Indic LLM precedents:**
  - OpenHathi: +16k tokens, about 2.5B tokens of training.
  - Tamil-LLaMA: +16k tokens, LoRA, about 48 A100-hours.
  - Nanda: +20% vocab, which halved fertility, then 65B tokens of training.
  - Takeaway: roughly 1B to tens of billions of text tokens, far more than a TTS corpus holds.

## Precedent in speech TTS

- Orpheus, Llasa, CosyVoice2 and Spark-TTS keep the base text tokenizer and only append audio tokens.
- Indic speech LMs keep the stock tokenizer and just fine-tune: Veena, svara-tts, Hindi Orpheus fine-tunes, and chatterbox-indic-lora.
- IndicF5 is character-level. Indic Parler-TTS trained its larger tokenizer from the start rather than retrofitting it.
- Praxy Voice romanizes text to ISO-15919 to feed a frozen tokenizer. ([arXiv 2604.25441](https://arxiv.org/abs/2604.25441))
- We found **no study showing that text-token fertility drives TTS quality**. Text is a small share of the sequence next to audio at 25 tokens/s. Fertility mostly costs context length and alignment difficulty, which is likelier to hurt te/ta than hi.

## Recommendation

1. Keep Qwen3 and fine-tune first. Measure CER per language.
2. If te/ta stay poor, do an append-only extension: 5–15k Indic pieces from a tokenizer we train ourselves, which avoids the Sarvam license. Init them from the mean of their Qwen sub-tokens, warm up the new embeddings first, then fine-tune jointly.
3. Avoid a hybrid Sarvam+Qwen tokenizer. It means two segmentations, the license problem, and the same new-row problem.

Ideas to brainstorm:
- An ablation of romanized input against native script.
- Character-level input for Indic spans only.
