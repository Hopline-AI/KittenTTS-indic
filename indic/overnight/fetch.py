"""Pull a capped, filtered slice of one IndicVoices-R language onto disk.

Runs on the login node (the only host with a network). Downloads whole parquet
files, strided across the language's file list so the slice spans many
speakers, writes each clip's audio bytes as-is, and deletes the parquet.

    python -m indic.overnight.fetch --lang Hindi --code hi --out DATA --hours 10
    python -m indic.overnight.fetch --lang Hindi --code hi --out DATA --probe   # 3 clips
    python -m indic.overnight.fetch --lang Hindi --code hi --out DATA --dry     # synthetic

Writes DATA/audio/<code>/*, DATA/manifests/<code>.{train,heldout}.jsonl and
finally DATA/manifests/<code>.DONE (json stats) or <code>.FAILED (the error).
Heldout clips come from the dataset's own test split.
"""

import argparse
import json
import os
import pathlib
import sys
import traceback

REPO = "ai4bharat/indicvoices_r"
MIN_S, MAX_S = 1.0, 20.0
PER_SPEAKER = 150          # ponytail: crude speaker balance; per-speaker hours if it matters
HELDOUT = 24
SAMPLE_TEXT = {"hi": "नमस्ते, आज मौसम अच्छा है।", "ta": "வணக்கம், இன்று வானிலை நன்றாக உள்ளது.",
               "ur": "آج موسم اچھا ہے۔"}


def _cer_ok(value):
    try:
        cer = float(value)
    except (TypeError, ValueError):
        return True                      # unmeasured: keep
    if cer > 1.0:
        cer /= 100.0                     # some rows store percent
    return cer <= 0.25


def _write_clip(out_dir, utt_id, audio):
    ext = pathlib.Path(audio.get("path") or "x.wav").suffix or ".wav"
    path = out_dir / f"{utt_id}{ext}"
    path.write_bytes(audio["bytes"])
    return path


def _rows(parquet_file):
    """Rows of a pyarrow ParquetFile, a batch at a time (files are ~450 MB)."""
    for batch in parquet_file.iter_batches(batch_size=64):
        yield from batch.to_pylist()


def take(rows, code, tag, audio_dir, budget_s, per_speaker, seen, limit=None):
    """Filter rows into manifest records. `seen` counts clips per speaker."""
    records, used = [], 0.0
    for row in rows:
        duration = float(row.get("duration") or 0)
        text = (row.get("text") or row.get("normalized") or "").strip()
        speaker = f"{code}_{row.get('speaker_id')}"
        audio = row.get("audio") or {}
        if not (MIN_S <= duration <= MAX_S and text and audio.get("bytes")
                and _cer_ok(row.get("cer")) and seen.get(speaker, 0) < per_speaker):
            continue
        utt_id = f"{code}_{tag}{sum(seen.values()):06d}"
        path = _write_clip(audio_dir, utt_id, audio)
        seen[speaker] = seen.get(speaker, 0) + 1
        records.append({"audio": str(path), "text": text, "speaker_id": speaker,
                        "language": code, "utt_id": utt_id})
        used += duration
        if used >= budget_s or (limit and len(records) >= limit):
            break
    return records, used


def _files(lang, token):
    from huggingface_hub import HfApi
    names = [f for f in HfApi(token=token).list_repo_files(REPO, repo_type="dataset")
             if f.startswith(f"{lang}/") and f.endswith(".parquet")]
    return (sorted(f for f in names if "/train-" in f),
            sorted(f for f in names if "/test-" in f))


def _strided(files, n):
    if n >= len(files):
        return files
    step = len(files) / n
    return [files[int(i * step)] for i in range(n)]


def fetch(args):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    token = os.environ.get("HF_TOKEN")
    audio_dir = args.out / "audio" / args.code
    audio_dir.mkdir(parents=True, exist_ok=True)
    tmp = args.out / "tmp" / args.code
    train_files, test_files = _files(args.lang, token)
    if not train_files:
        raise RuntimeError(f"no parquet files for {args.lang} (gated terms accepted?)")

    def load(name):
        path = hf_hub_download(REPO, name, repo_type="dataset", token=token, local_dir=tmp)
        try:
            yield from _rows(pq.ParquetFile(path))
        finally:
            os.remove(path)

    seen, heldout = {}, []
    if args.probe:
        from huggingface_hub import HfFileSystem
        handle = pq.ParquetFile(HfFileSystem(token=token).open(
            f"datasets/{REPO}/{test_files[0]}"))
        rows = handle.read_row_group(0).to_pylist()
        train, _ = take(rows, args.code, "p", audio_dir, 1e9, PER_SPEAKER, seen, limit=3)
        return train, [], 0.0

    # ~0.7 h of audio per parquet file; ask for a few spare for the filters
    want = max(1, int(args.hours / 0.6) + 1)
    budget, train = args.hours * 3600, []
    for name in _strided(train_files, want):
        got, used = take(load(name), args.code, "t", audio_dir, budget, PER_SPEAKER, seen)
        train += got
        budget -= used
        print(f"[{args.code}] {name}: +{len(got)} clips, {budget / 3600:.1f} h left",
              flush=True)
        if budget <= 0:
            break
    test_seen = {}
    heldout, _ = take(load(test_files[0]), args.code, "h", audio_dir, 1e9, 3, test_seen,
                      limit=HELDOUT)
    return train, heldout, args.hours * 3600 - budget


def fake(args):
    """Synthetic clips with real-script text, for the local dry run."""
    import numpy as np
    import soundfile as sf

    audio_dir = args.out / "audio" / args.code
    audio_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    text = SAMPLE_TEXT.get(args.code, SAMPLE_TEXT["hi"])

    def clips(n, tag):
        out = []
        for i in range(n):
            path = audio_dir / f"{args.code}_{tag}{i}.wav"
            sf.write(path, rng.normal(0, 0.1, 24000 * 2).astype("float32"), 24000)
            out.append({"audio": str(path), "text": text, "language": args.code,
                        "speaker_id": f"{args.code}_s{i % 3}",
                        "utt_id": f"{args.code}_{tag}{i}"})
        return out

    return clips(12, "t"), clips(4, "h"), 24.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", required=True, help="IndicVoices-R config, e.g. Hindi")
    ap.add_argument("--code", required=True, help="our language code, e.g. hi")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--hours", type=float, default=10)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    manifests = args.out / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    for stale in (".DONE", ".FAILED"):
        (manifests / f"{args.code}{stale}").unlink(missing_ok=True)
    try:
        train, heldout, seconds = (fake if args.dry else fetch)(args)
        for split, rows in (("train", train), ("heldout", heldout)):
            tmp = manifests / f"{args.code}.{split}.jsonl.tmp"
            tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                           encoding="utf-8")
            tmp.replace(manifests / f"{args.code}.{split}.jsonl")
        if not train:
            raise RuntimeError("no clips survived the filters")
        stats = {"lang": args.lang, "code": args.code, "train_clips": len(train),
                 "heldout_clips": len(heldout), "hours": round(seconds / 3600, 2),
                 "speakers": len({r["speaker_id"] for r in train})}
        (manifests / f"{args.code}.DONE").write_text(json.dumps(stats))
        print(json.dumps(stats), flush=True)
    except Exception:
        (manifests / f"{args.code}.FAILED").write_text(traceback.format_exc())
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
