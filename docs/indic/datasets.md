# Indic speech datasets

Target languages for the first pass are hi, bn, ta, te and mr. The audio needs to be 24 kHz.

| Dataset | Langs | Hours | Speakers | Audio | License | Where |
|---|---|---|---|---|---|---|
| **IndicVoices-R** | 22 (all 5) | 1,704 (9–175/lang) | 10,496 | 48 kHz, crowd audio enhanced, 93% unscripted | CC-BY-4.0, gated | [HF](https://huggingface.co/datasets/ai4bharat/indicvoices_r) |
| **IndicTTS (IIT Madras)** | 13+ (all 5) | ~10 h/lang | 2/lang | 48 kHz studio | Custom; commercial use reportedly OK with citation. **Read the license itself.** | [HF](https://huggingface.co/collections/SPRINGLab/indictts-datasets) |
| **Rasa** (expressive) | 22 | 1,145 | 44 speaker-lang pairs | 48 kHz studio, 6 emotions + styles | CC-BY-4.0 | [HF](https://huggingface.co/datasets/ai4bharat/Rasa) |
| **SYSPIN / LIMMITS** | 9 (hi, bn, mr, te; **no ta**) | 47–80 h/lang | 18 artists | Studio | CC-BY-4.0 | [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/iisc_syspin_s1_0_corpus.html) |
| SPRING-INX | 10 (no te) | ~2,000 | many | 16 kHz crowd/phone | CC-BY-4.0 | [IITM](https://asr.iitm.ac.in/dataset) |
| IndicVoices | 22 | 11.2k transcribed | ~51k | 16 kHz phone, noisy | CC-BY-4.0, gated | [HF](https://huggingface.co/datasets/ai4bharat/IndicVoices) |
| Kathbath | 12 | 1,684 | 1,218 | 16 kHz phone, read | CC0 | [HF](https://huggingface.co/datasets/ai4bharat/Kathbath) |
| Shrutilipi | 12 | 6,400 | radio | 16 kHz broadcast, mined | CC-BY-4.0 | [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/shrutilipi.html) |
| FLEURS | many | ~10 h/lang | few | 16 kHz read | CC-BY-4.0 | [HF](https://huggingface.co/datasets/google/fleurs) |
| OpenSLR 37/64/65/66 | bn, mr, ta, te | 3–9 h/lang | 10–50 | 48 kHz | **CC-BY-SA** | [openslr](https://openslr.org) |
| MUCS 2021 (code-switch) | hi-en, bn-en | ~600 | many | 8–16 kHz | SLR103 restrictive / SLR104 CC-BY-SA | [openslr](https://openslr.org/104/) |
| Svarah | Indian English | 9.6 | 117 | phone | No license stated (likely CC-BY-4.0, **not confirmed**) | [GitHub](https://github.com/AI4Bharat/Svarah) |

There were no significant new open datasets in 2025–26. Sarvam and Maya Research released models only. AI4Bharat trained IndicF5 on 1,417 h of Rasa, IndicTTS, LIMMITS and IndicVoices-R.

## Recommended starter mix (~800–900 h)

1. **Core (~400–500 h):** SYSPIN for hi/bn/mr/te, plus Rasa and IndicTTS for all five languages.
2. **Speaker variety for cloning (~400 h):** IndicVoices-R, filtered by quality score and SNR.
3. **Tamil top-up:** more IndicVoices-R and Rasa Tamil.
4. **Keep 16 kHz crowd/broadcast audio out of the TTS targets:** Shrutilipi, IndicVoices raw and SPRING-INX. At most use them for text or auxiliary training.
5. **Keep some English and existing-language data** (replay) to avoid forgetting.

Pipeline notes:
- Resample to 24 kHz.
- Re-check transcripts with IndicConformer and drop clips with high CER.
- Apply NFC normalization to all text.
- Accept the gating terms on HF before downloading on the cluster.
- Attribution is required for all CC-BY sources.

## Text normalization tools

The current normalizer is English-only.

- [NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing): WFST normalization for **Hindi only**.
- [indic-num2words](https://pypi.org/project/indic-num2words): digits to words for all 5 languages. It doesn't handle dates or currency.
- [ai4bharat/indic-numtowords](https://github.com/ai4bharat/indic-numtowords): a lightweight alternative.
- [IndicNLP library](https://pypi.org/project/indic-nlp-library): Unicode cleanup, sentence splitting, transliteration.
- Sarvam's normalization is server-side only, with no open code.
- G2P isn't needed, since Indic scripts are near-phonetic. Schwa deletion (hi/mr/bn) is learned from data.

Plan: NeMo for Hindi, and indic-num2words plus our own date, currency and phone rules for the others.
