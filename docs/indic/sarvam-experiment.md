# Experiment: Sarvam-only tokenizer (no Qwen3 text vocab)

**Hypothesis:** we replace the Qwen3 text tokenizer entirely with `sarvamai/sarvam-1`, a 68,096-piece SentencePiece tokenizer with byte fallback. Indic text then costs about 3–5× fewer tokens, which should make text–audio alignment easier and improve Indic TTS, especially ta/te. Along the way the vocab shrinks from 158k to about 75k.

## Tokenizer checks (measured)

- The tokenizer is a `LlamaTokenizer` with specials `<s>`, `</s>`, `<unk>` and **256 byte-fallback pieces**. Every input round-trips losslessly, and nothing becomes `<unk>`.
- Expression markup survives unchanged:
  - `[joyful] We won <laugh> I can (((hardly))) believe it!` → 19 tokens, decodes back exactly.
  - `{emo: 1}` → 7 tokens.
  - `Price ₹500 – 😀 Ω ü ß 中文` → 30 tokens, decodes back exactly.
- Fertility (tokens/word):

  | | hi | bn | ta | te | mr | en |
  |---|---|---|---|---|---|---|
  | Qwen3 | 6.0 | 5.1 | 9.3 | 10.8 | 5.2 | 1.08 |
  | Sarvam-1 | 2.1 | 1.7 | 2.0 | 2.5 | 1.5 | 1.17 |

- Each Sarvam piece maps to 5.68 Qwen tokens on average (max 27). That makes it practical to start each new embedding as the average of the Qwen embeddings it covers.

## New vocab layout

```
[0, 68096)               Sarvam-1 pieces (incl. <unk>=0, <s>, </s>, byte fallback)
68096 .. 68101           speech_start, speech_end, text_start, text_end, START, STOP
68102 .. 68102+6561      S3 audio codec tokens   (audio_id_base = 68102)
74663 .. 74667           final_seg, reference_speech_start/end, reference_text_start/end
                         total ≈ 74,668 rows (pad to a multiple of 64/128 for speed)
```

- The `token_map` in `config.json` gets the new ids. The code already reads everything through `TokenMap`, so prompt assembly and logit masking need no changes.
- `embed_tokens`, which is tied to `lm_head`, shrinks from 158,241×2048 (324M params) to about 74,700×2048 (153M params).

## Embedding init

| Rows | Init |
|---|---|
| 6561 audio tokens | **Copy** from old rows 151675.. (unchanged meaning) |
| 6 named specials + 5 markers | **Copy** from old ids |
| Sarvam pieces | **Average of Qwen sub-token embeddings**: decode the piece (`▁` → space), Qwen-encode it, take the mean of those old rows. FOCUS is an option for later |
| Byte-fallback `<0xNN>` | Qwen's byte-level token for that byte |

Keep a random-init variant as an ablation, to measure how much this initialization helps.

## Training stages

1. **Phase A: embedding warm-up.**
   - Freeze all 28 transformer layers and `spk_proj`.
   - Train only the new text rows.
   - Use the normal TTS loss on target audio tokens.
   - Data: Indic plus English, so English gets relearned through the new pieces too.
   - Stop when the loss plateaus (expect thousands of steps, not epochs).
2. **Phase B: joint fine-tune with ternary QAT.**
   - Unfreeze everything (see [training.md](training.md) §2–4).
   - Lower the learning rate for the body than for the embeddings.
3. **Optional Phase 0: text-only continued pretraining.**
   - Text-only next-token prediction on Indic and English text, so the body re-learns to read Sarvam pieces before any speech training.
   - The precedents (Tamil-LLaMA, OpenHathi) needed about 1B text tokens or more, so use this only if Phase A converges slowly.

**English must stay in the mix.** With Qwen gone, English is re-tokenized too: Sarvam's English fertility is about the same, but the pieces are different. Put real English recordings in every phase. LibriTTS-R and similar would work, but not KittenTTS outputs; see the license note in [finetuning.md](finetuning.md) §4b.

## Experiment matrix

The control is the only way to tell whether the tokenizer, rather than the extra data, made the difference. Every arm gets the same data, steps and eval.

| Arm | Tokenizer | Init | Purpose |
|---|---|---|---|
| **S1** (main) | Sarvam only | Mean of Qwen sub-tokens | The experiment |
| S2 | Sarvam only | Random | Measures the value of the init |
| C1 (control) | Qwen3 (unchanged) | n/a | Same fine-tune with Qwen3 |
| C0 | Qwen3, no fine-tune | n/a | Stock KittenTTS 2 |

Report per language (hi/bn/ta/te/mr/en):
- CER (IndicConformer; Whisper for en)
- speaker similarity
- UTMOSv2 (relative only)
- PSP accent scores
- tokens per second of audio
- real-time factor (RTF)

Also plot loss against step to compare convergence speed.

## Code changes needed

- `model.py:172`: `AutoTokenizer.from_pretrained(lm_dir)` already loads whatever tokenizer is in `lm/`, so ship the Sarvam tokenizer files there.
- `config.json` `token_map`: the new ids above.
- **Check:** does any inference code assume `add_special_tokens=False` produces no BOS? Sarvam/Llama tokenizers prepend `<s>` by default. `model.py:315,356,421` already pass `add_special_tokens=False`, which looks fine.
- **Check:** `text.py` chunking and danda fixes (see [finetuning.md](finetuning.md)) are independent of the tokenizer, but needed either way.
- The llama.cpp fork and the GGUF export need the new vocab and tokenizer type (SentencePiece instead of BPE).

## Risks

- **License:** the sarvam-1 tokenizer is under the Sarvam non-commercial license. That is fine for a research experiment. If S1 wins and we want to ship commercially, retrain a SentencePiece tokenizer of the same size on our own Indic and English text. The pipeline above works unchanged with any tokenizer.
- **Body has to re-learn reading text:** the model has never seen these ids. If Phase A stalls, add Phase 0.
- **English regression:** track English CER from the first checkpoint.
- **Expression tags:** tags like `[joyful]` used to be a few Qwen tokens and are now different Sarvam pieces. Rasa's emotion data is needed to re-learn them.
