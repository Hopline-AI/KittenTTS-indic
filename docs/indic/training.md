# Training plan

There is no training code in this repo, so everything below has to be written. The goal is to keep the parts that already work frozen (the audio codec, vocoder and speaker encoder) and adapt only the speech LM to Indic text.

## What gets trained and what stays frozen

| Component | Trained? | Notes |
|---|---|---|
| S3 tokenizer (audio → tokens) | Frozen | Used offline to tokenize the dataset. Verify it with a round-trip first |
| s3gen flow + vocoder (tokens → wav) | Frozen | Fine-tune only if the round-trip CER on Indic audio is bad |
| pyannote speaker-embedding model | Frozen | Precompute embeddings offline |
| `spk_proj` (512→2048 + LayerNorm) | Train with a low LR, or freeze at first | Full precision |
| `embed_tokens` / tied `lm_head` | Trained | Full precision. Indic BPE rows are likely under-trained today |
| 28 transformer layers (q/k/v/o/gate/up/down) | Trained **with ternary QAT**, or LoRA | Must end up ternary again |
| RMSNorms | Trained | Full precision |

## Training sequence format

The checkpoint is reference-supervised, because its `config.json` token map has the reference-role ids 158237–158240. So each training example should copy what inference builds (`prompt.py`):

```
pos 0:   speaker embedding → spk_proj (replaces hidden state of START)
[START]
<ref_text_start> ref transcript <ref_text_end>
<ref_speech_start> ref audio tokens (151675 + s3 id) <ref_speech_end>
[optional {emo: 1} control ids]           # when the target uses emotion tags
<text_start> target text <final_seg>
<speech_start> target audio tokens <speech_end>
```

- **Loss** applies only to the target audio tokens and `<speech_end>`. Everything else is masked out with -100.
- **Reference:** a *different* utterance from the same speaker, 3–10 s long. Sometimes drop the reference prefix entirely, since inference supports `reference_prefix=None`.
- **Speaker embedding:** computed from the reference clip.
- **Emotion tags:** Rasa's emotion labels can map onto the existing tags (angry, sad, joyful, surprised, ...). Fear and disgust have no existing tag.

## Pipeline

### 0. Data prep (CPU plus a few GPU-hours)

1. Download the datasets listed in [datasets.md](datasets.md). Resample to 16 kHz for S3 and 24 kHz for reference.
2. Clean the text:
   - NFC normalization
   - Indic number, date and currency normalization (NeMo for Hindi, indic-num2words for the rest)
   - Danda-aware sentence handling
3. Run IndicConformer over every clip and drop clips where CER against the transcript is above about 10–15%.
4. Precompute:
   - S3 tokens (25 tokens/s)
   - pyannote speaker embeddings
   - Qwen3 token ids for the text
5. Pair each utterance with its references, store as shards (parquet or webdataset), and hold out speakers for evaluation.

### 1. Plumbing (no real training)

- Load the bf16 `lm/model.safetensors` into `Qwen3ForCausalLM`.
- Make `spk_proj` a real module applied on every forward pass, not only through the prefill hook.
- Check that **teacher-forced loss on English data is low** before changing anything else. That proves the sequence format matches what the model was trained on.

### 2. Ternary QAT layer

Each 128-wide group is quantized with its own scale:

```python
def ternary_ste(W, group=128):
    Wg = W.view(-1, group)
    s = Wg.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)   # per-group scale (verify vs ternary.py)
    Q = (Wg / s).round().clamp(-1, 1) * s
    return (Wg + (Q - Wg).detach()).view_as(W)               # straight-through estimator
```

- Keep a bf16 "shadow" weight, use `ternary_ste(W)` in the forward pass, and update W with the optimizer.
- **Check that re-quantizing the released weights reproduces them exactly (a fixed point) before training.** The note at `model.py:173-176` hints the scale is not plain absmean.
- Export by packing with `ternary.py` back into `lm_packed` format, so the existing inference path and the llama.cpp fork keep working.

### 3. Pilot: Hindi only (~50 h SYSPIN Hindi)

- Pick one of two options:
  - **(a) LoRA:** adapters (r=64) on all linear layers, plus training the embeddings, over frozen ternary weights. Then merge them and run a short QAT pass to re-ternarize.
  - **(b) Full QAT fine-tune.** It is simpler and avoids the merge step.
- Evaluate CER, speaker similarity and UTMOS against stock KittenTTS 2 and IndicF5.
- **Goal:** prove the pipeline improves Hindi without breaking English.

### 4. Main run: 5 languages

- **Data:** about 800–900 h Indic, plus about 10–20% English and other existing-language data (real recordings, e.g. LibriTTS-R) to prevent forgetting.
- **Balancing:** temperature sampling across languages, so Tamil isn't swamped by Hindi.
- **Settings:** LR around 1e-5 to 5e-5 for the body, higher for embeddings; 2–4 epochs. Evaluate every N steps on the held-out set.
- **Optional curriculum:** single sentences first, then longer multi-sentence targets.

### 5. Optional: vocab extension (only if te/ta CER stays high)

1. Train our own 5–15k Indic SentencePiece or BPE tokenizer, which avoids the Sarvam license.
2. Append the new rows after id 158240.
3. Init each new row as the mean of its Qwen sub-token embeddings.
4. **Phase A:** train only the new rows with the body frozen.
5. **Phase B:** run steps 3–4 again jointly.
6. Mask the new rows out of audio sampling. `logits.py` already restricts sampling to the audio range.

### 6. Optional: s3gen fine-tuning

Do this only if the codec round-trip shows Indic-specific losses. It is the flow-matching decoder (`student_flow.py`) and is a separate project.

## Compute estimate

These are rough figures to check on the cluster.

- **Tokens per example:** about 250 target audio tokens (10 s) + 150–250 reference audio tokens + about 100 text tokens, so roughly 600.
- **Per epoch:** 900 h is about 400k utterances, so about 240M tokens per epoch.
- **FLOPs:** about 6 × 1.7e9 × 2.4e8 ≈ 2.5e18 per epoch. At about 40% utilization that is roughly **7 H100-hours or 15–20 A100-hours per epoch**.
- **Total:** 3 epochs plus QAT overhead and evals comes to roughly **50–100 A100-hours** for the main run. The pilot needs a few GPU-hours.
- **Memory:**
  - Full fine-tune with AdamW: about 16 bytes/param, or about 27 GB, plus activations. One 80 GB GPU is enough with gradient checkpointing; use DDP or FSDP across GPUs for speed.
  - LoRA fits on a 24–48 GB card.
- **Data prep:** S3 tokenizing 900 h takes a few GPU-hours, and IndicConformer filtering a few more.

## Open questions to brainstorm

- **How was the original model trained?** We don't know the SFT mix, the reference/no-reference ratio, the `final_seg` usage, or how the `{emo: 1}` control was trained. We need to infer these from inference code and loss checks.
- **Per-group scale formula:** confirm a fixed point on the released weights before trusting the QAT.
- **LoRA+merge or full QAT?** Compare them in the pilot.
- **Code-mixed text:** handle it with code-mixed data such as MUCS (license!) or generated Hinglish text paired with real audio.
- **Script choice:** native script, or a romanized ablation (Praxy Voice)?
- **License:** check Hopline's revenue and funding against the $1M threshold.
