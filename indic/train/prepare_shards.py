"""Turn a corpus of audio and transcripts into the parquet shards training reads.

This is the long pole of the whole project. Training 900 hours costs about 26
A100-hours; getting those 900 hours into shards costs a comparable 30-80, and
unlike training it cannot be skipped or shortened. It is also embarrassingly
parallel, so it is written to be run as a Slurm array: pass `--shard i
--num-shards n` and each task writes its own file, touching nothing another task
owns.

Everything expensive happens exactly once here, so the training loop never opens
an audio file:

    waveform -> S3 codec tokens      25 Hz, the model's actual output alphabet
    waveform -> speaker embedding    512-d, what spk_proj was trained to receive
    transcript -> text ids           tokenizer-specific; recorded in the shard
    transcript vs ASR -> cer         the quality filter, measured not assumed

**References are not paired here.** One row per utterance; `ShardDataset` picks
a reference from the same speaker at read time, so the pairing varies per epoch
and the codec tokens are stored once rather than once per use. See
`shard_schema`.

**The CER column is the filter that matters.** Crowd-sourced Indic corpora carry
transcripts that do not match the audio, and a TTS model trained on a mismatched
pair learns to say the wrong word confidently. Clips are scored against their own
transcript with the same recognizer the evaluation harness uses and the score is
written down rather than applied, so the threshold stays a training-time
decision.

    python -m indic.train.prepare_shards --manifest corpus.jsonl \\
        --out shards/ --tokenizer sarvamai/sarvam-1 --asr auto

Manifest is JSON Lines, one object per utterance:

    {"audio": "/path/clip.wav", "text": "...", "speaker_id": "spk42",
     "language": "hi", "emotion": null}
"""

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.train.shard_schema import SPEAKER_EMBEDDING_DIM, write_shard

# Clips outside this range are dropped before any GPU work. Too short carries no
# prosody and destabilises the model; too long blows the per-step token budget
# (see the fitted memory relation in indic/bench/fit_config.py) and is usually a
# segmentation failure rather than a real utterance.
MIN_SECONDS = 1.0
MAX_SECONDS = 30.0


def read_manifest(path):
    """Utterance records from JSON Lines, validated enough to fail early."""
    required = {"audio", "text", "speaker_id", "language"}
    rows = []
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            missing = required - set(record)
            if missing:
                raise ValueError(f"{path}:{number}: missing {sorted(missing)}")
            rows.append(record)
    return rows


def normalize_text(text, language, apply_fixes=True):
    """Transcript cleanup, before tokenization.

    NFC and Indic digits, for the reason given in `indic/text/indic_text.py`:
    the same Devanagari syllable has several decomposed spellings and the
    tokenizer treats them as different pieces, so a corpus mixing them splits
    the vocabulary's mass across duplicates.
    """
    if not apply_fixes:
        return text
    from indic.text.indic_text import has_indic, normalize_indic

    return normalize_indic(text) if has_indic(text) else text


class Encoders:
    """The three models this job runs, loaded once and shared across clips."""

    def __init__(self, device, cache_dir=None, tokenizer=None, asr="none"):
        self.device = device
        self.cache_dir = cache_dir
        self.asr_mode = asr
        self._codec = None
        self._speaker = None
        self._asr = {}
        self.tokenizer_name = tokenizer
        self._tokenizer = None

    @property
    def codec(self):
        if self._codec is None:
            from kittenml.kittentts2.vocoder import S3Codec
            self._codec = S3Codec(device=self.device, cache_dir=self.cache_dir)
        return self._codec

    @property
    def speaker(self):
        if self._speaker is None:
            from huggingface_hub import hf_hub_download

            from kittenml.kittentts2.speaker import SpeakerEncoder
            weights = hf_hub_download("KittenML/kitten-tts-2",
                                      "speaker/model.safetensors",
                                      cache_dir=self.cache_dir)
            self._speaker = SpeakerEncoder(weights, device=self.device)
        return self._speaker

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, cache_dir=self.cache_dir)
        return self._tokenizer

    def recognizer(self, language):
        """The ASR used for the quality score, or None when disabled."""
        if self.asr_mode == "none":
            return None
        if language not in self._asr:
            from indic.eval.asr import get_asr
            self._asr[language] = get_asr(self.asr_mode, language,
                                          device=self.device)
        return self._asr[language]

    def encode(self, record, apply_fixes=True):
        """One manifest record -> one shard row, or None when it is unusable."""
        import soundfile as sf

        path = record["audio"]
        info = sf.info(path)
        duration = info.frames / float(info.samplerate)
        if not MIN_SECONDS <= duration <= MAX_SECONDS:
            return None, f"duration {duration:.1f}s outside [{MIN_SECONDS}, {MAX_SECONDS}]"

        text = normalize_text(record["text"], record["language"], apply_fixes)
        if not text.strip():
            return None, "empty transcript"

        audio_tokens = self.codec.encode_reference(path)
        if not audio_tokens:
            return None, "codec produced no tokens"
        embedding = self.speaker(path)
        text_ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if not text_ids:
            return None, "tokenizer produced no ids"

        cer = -1.0
        recognizer = self.recognizer(record["language"])
        if recognizer is not None:
            from indic.eval.metrics import cer as compute_cer
            try:
                cer = float(compute_cer(text, recognizer.transcribe(
                    path, record["language"])))
            except Exception as exc:                 # a scorer failure is not a data failure
                print(f"  [asr] {path}: {type(exc).__name__}: {exc}", flush=True)

        return {
            "utt_id": record.get("utt_id") or pathlib.Path(path).stem,
            "language": record["language"],
            "speaker_id": record["speaker_id"],
            "text_ids": text_ids,
            "audio_tokens": audio_tokens,
            "speaker_embedding": list(map(float, embedding)),
            "duration_s": duration,
            "cer": cer,
            "emotion": record.get("emotion"),
            # Absolute, so a shard stays readable from a different working
            # directory -- which is the normal case on a cluster.
            "audio_path": str(pathlib.Path(path).resolve()),
        }, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--tokenizer", default="sarvamai/sarvam-1")
    ap.add_argument("--asr", default="none", choices=["none", "auto", "whisper",
                                                      "indicconformer"],
                    help="quality scorer; 'none' writes cer=-1 (unmeasured)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir")
    ap.add_argument("--rows-per-shard", type=int, default=2000)
    ap.add_argument("--shard", type=int, default=0, help="this task's index")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--no-text-fixes", action="store_true",
                    help="skip NFC and Indic-digit normalization")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    records = read_manifest(args.manifest)
    # Strided rather than blocked: manifests arrive grouped by speaker or by
    # language, so contiguous blocks would hand one task every long clip and
    # another every short one.
    mine = records[args.shard::args.num_shards]
    if args.limit:
        mine = mine[:args.limit]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"task {args.shard}/{args.num_shards}: {len(mine)} of {len(records)} "
          f"utterances -> {args.out}", flush=True)

    encoders = Encoders(args.device, args.cache_dir, args.tokenizer, args.asr)
    buffer, written, skipped, reasons = [], 0, 0, {}
    part = 0
    started = time.time()

    def flush():
        nonlocal buffer, part, written
        if not buffer:
            return
        path = args.out / f"part-{args.shard:05d}-{part:04d}.parquet"
        write_shard(path, buffer, tokenizer=args.tokenizer,
                    embedding_dim=SPEAKER_EMBEDDING_DIM,
                    extra={"source_manifest": str(args.manifest),
                           "text_fixes": not args.no_text_fixes,
                           "asr": args.asr})
        print(f"  wrote {path.name} ({len(buffer)} rows)", flush=True)
        written += len(buffer)
        buffer = []
        part += 1

    for index, record in enumerate(mine):
        try:
            row, reason = encoders.encode(record, apply_fixes=not args.no_text_fixes)
        except Exception as exc:               # one unreadable clip must not end the job
            row, reason = None, f"{type(exc).__name__}: {exc}"
            if skipped == 0:                     # the first one, in full, so a smoke run says why
                print(f"  skip {record.get('utt_id')}: {reason}", flush=True)
        if row is None:
            skipped += 1
            reasons[reason.split(":")[0]] = reasons.get(reason.split(":")[0], 0) + 1
            continue
        buffer.append(row)
        if len(buffer) >= args.rows_per_shard:
            flush()
        if index and index % 200 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index + 1}/{len(mine)}  {rate:.1f} clips/s  "
                  f"{skipped} skipped", flush=True)
    flush()

    summary = {"task": args.shard, "num_shards": args.num_shards,
               "written": written, "skipped": skipped, "skip_reasons": reasons,
               "seconds": round(time.time() - started, 1),
               "tokenizer": args.tokenizer, "asr": args.asr}
    (args.out / f"prepare-{args.shard:05d}.json").write_text(
        json.dumps(summary, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
