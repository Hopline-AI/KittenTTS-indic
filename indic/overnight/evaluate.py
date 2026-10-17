"""Baseline vs fine-tuned on held-out speech, per language, then REPORT.md.

The score is teacher-forced loss on the *audio* tokens of held-out IndicVoices-R
test clips: how well each model predicts the real recording given its text.
Both models are scored on identical audio targets, each through its own text
tokenizer, so the comparison is fair across a tokenizer swap. It is a proxy,
not intelligibility: CER through an ASR is the next step (see the report).

    python -m indic.overnight.evaluate --data DATA --baseline SNAPSHOT_DIR \\
        --finetuned DATA/run/export [--device cuda] [--dry]

REPORT.md is written even when scoring fails, so the morning starts from it.
"""

import argparse
import glob
import json
import math
import os
import pathlib
import statistics
import sys
import time
import traceback


def heldout_rows(data):
    import pyarrow.parquet as pq

    from indic.train.prepare_shards import normalize_text

    texts = {}
    for path in glob.glob(str(data / "manifests" / "*.heldout.jsonl")):
        for line in open(path, encoding="utf-8"):
            record = json.loads(line)
            texts[record["utt_id"]] = (record["text"], record["audio"])
    rows = []
    for path in sorted(glob.glob(str(data / "shards" / "heldout" / "*" / "*.parquet"))):
        table = pq.read_table(path, columns=["utt_id", "language", "audio_tokens",
                                             "speaker_embedding"])
        for row in table.to_pylist():
            if row["utt_id"] in texts:
                text, audio = texts[row["utt_id"]]
                row["text"] = normalize_text(text, row["language"])
                row["audio"] = audio
                rows.append(row)
    return rows


def load_model(spec, device, dry):
    """(model, tokenizer, token_map) for a repo dir, or the toy stand-ins."""
    import torch

    from indic.train import loop
    if dry:
        from indic.overnight.prepare import DryTokenizer
        from indic.train.data import tiny_token_map
        token_map = tiny_token_map()
        model = loop.toy_model(token_map, dtype=torch.float32, seed=0)
        loop.attach_speaker_head(model, speaker_dim=512, dtype=torch.float32)
        if spec != "baseline":
            state = torch.load(spec, map_location="cpu", weights_only=False)["model"]
            if any(k.startswith("base_model.") for k in state):   # a LoRA run
                model = loop.apply_lora(model)
            model.load_state_dict(state)
        return model.to(device).eval(), DryTokenizer(), token_map

    from transformers import AutoTokenizer
    spec = pathlib.Path(spec)
    token_map = loop._tokens.TokenMap.from_config(json.loads((spec / "config.json").read_text()))
    model = loop.released_model(spec / "lm", dtype=torch.bfloat16)
    return model.to(device).eval(), AutoTokenizer.from_pretrained(spec / "lm"), token_map


def score(model, tokenizer, token_map, rows, device):
    """Mean audio-token loss per language, from per-utterance losses."""
    import torch

    from indic.train.data import Collator, build_example
    from indic.train.loop import forward_batch

    per_lang = {}
    with torch.no_grad():
        for row in rows:
            ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            input_ids, labels = build_example(token_map, ids, row["audio_tokens"])
            batch = Collator()([{"input_ids": torch.tensor(input_ids),
                                 "labels": torch.tensor(labels),
                                 "speaker_embedding": torch.tensor(row["speaker_embedding"])}])
            loss = float(forward_batch(model, batch, device).loss)
            per_lang.setdefault(row["language"], []).append(loss)
    return per_lang


def samples(spec, rows, out, device):
    """Two cloned sentences per language, for listening."""
    import soundfile as sf

    from kittenml import KittenTTS

    model = KittenTTS(str(spec), device=device)
    written = []
    by_lang = {}
    for row in rows:
        by_lang.setdefault(row["language"], []).append(row)
    for lang, items in sorted(by_lang.items()):
        if len(items) < 3:
            continue
        reference = items[0]
        for i, target in enumerate(items[1:3]):
            path = out / f"{lang}_{i}.wav"
            try:
                audio = model.generate(target["text"], reference=reference["audio"],
                                       reference_text=reference["text"])
                sf.write(path, audio, model.sample_rate)
                written.append(str(path))
            except Exception as error:      # one bad sentence must not cost the rest
                print(f"[samples] {spec} {lang}: {type(error).__name__}: {error}",
                      flush=True)
    return written


def evaluate(args):
    data = pathlib.Path(args.data)
    rows = heldout_rows(data)
    if not rows:
        raise RuntimeError("no held-out rows (no heldout shards matched the manifests)")
    results = {"n_rows": len(rows), "models": {}}
    for name, spec in (("baseline", args.baseline), ("finetuned", args.finetuned)):
        started = time.time()
        model, tokenizer, token_map = load_model(spec, args.device, args.dry)
        results["models"][name] = score(model, tokenizer, token_map, rows, args.device)
        del model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"scored {name} in {time.time() - started:.0f}s", flush=True)
        if not args.dry:
            try:
                out = data / "samples" / name
                out.mkdir(parents=True, exist_ok=True)
                results.setdefault("samples", {})[name] = samples(spec, rows, out, args.device)
            except Exception:
                traceback.print_exc()
    (data / "eval.json").write_text(json.dumps(results, indent=1))
    return results


# ── report ────────────────────────────────────────────────────────────────────
def _read_json(path):
    try:
        return json.loads(pathlib.Path(path).read_text())
    except Exception:
        return None


def report(data, error=None):
    data = pathlib.Path(data)
    lines = ["# KittenTTS-indic overnight run", "",
             f"Written {time.strftime('%Y-%m-%d %H:%M')}. Data dir: `{data}`", ""]

    lines += ["## Stages", "", "| stage | status |", "|---|---|"]
    for stage in ("fetch", "prepare", "train", "eval"):
        status = data / "status"
        if (status / f"{stage}.failed").exists():
            state = f"**failed**: {(status / f'{stage}.failed').read_text().strip()}"
        elif (status / f"{stage}.ok").exists():
            state = "ok"
        elif (status / f"{stage}.started").exists():
            state = "started, never finished (killed or timed out: check logs/)"
        else:
            state = "not run"
        lines.append(f"| {stage} | {state} |")

    lines += ["", "## Data (IndicVoices-R)", "",
              "| lang | hours | train clips | speakers | heldout | train rows in shards |",
              "|---|---|---|---|---|---|"]
    for done in sorted(glob.glob(str(data / "manifests" / "*.DONE"))):
        s = _read_json(done) or {}
        code = s.get("code", pathlib.Path(done).stem)
        rows = sum((_read_json(p) or {}).get("written", 0) for p in
                   glob.glob(str(data / "shards" / "train" / code / "prepare-*.json")))
        lines.append(f"| {code} | {s.get('hours')} | {s.get('train_clips')} | "
                     f"{s.get('speakers')} | {s.get('heldout_clips')} | {rows} |")
    failed = sorted(pathlib.Path(p).stem for p in glob.glob(str(data / "manifests" / "*.FAILED")))
    if failed:
        lines += ["", f"Fetch failed for: {', '.join(failed)} (see `manifests/<code>.FAILED`)."]

    log = data / "run" / "train.jsonl"
    if log.exists():
        steps = [r for r in map(json.loads, open(log)) if "step" in r]
        if steps:
            warm = [r for r in steps if r["step"] >= 20] or steps
            head = statistics.mean(r["loss"] for r in steps[:20])
            tail = statistics.mean(r["loss"] for r in steps[-20:])
            hours = (steps[-1]["time"] - steps[0]["time"]) / 3600
            lines += ["", "## Training", "",
                      f"- steps logged: {steps[0]['step']}–{steps[-1]['step']} over {hours:.1f} h",
                      f"- loss: first 20 avg **{head:.3f}** → last 20 avg **{tail:.3f}**",
                      f"- throughput: median {statistics.median(r.get('tok_per_s', 0) for r in warm):,.0f} tokens/s, "
                      f"peak memory {max((r.get('max_mem_gb') or 0) for r in steps):.1f} GB"]
    export = data / "run" / "export"
    lines.append(f"- exported model: `{export}`" if export.exists() else
                 "- exported model: **missing** (training did not finish)")

    ev = _read_json(data / "eval.json")
    if ev and len(ev.get("models", {})) == 2:
        base, tuned = ev["models"]["baseline"], ev["models"]["finetuned"]
        lines += ["", "## Baseline vs fine-tuned (held-out, audio-token loss, lower is better)", "",
                  "| lang | n | baseline | fine-tuned | change | better? |", "|---|---|---|---|---|---|"]
        wins = 0
        for lang in sorted(set(base) & set(tuned)):
            b, t = statistics.mean(base[lang]), statistics.mean(tuned[lang])
            wins += t < b
            lines.append(f"| {lang} | {len(base[lang])} | {b:.3f} | {t:.3f} | "
                         f"{t - b:+.3f} ({math.exp(b - t):.2f}× ppl) | {'yes' if t < b else 'no'} |")
        lines += ["", f"Fine-tuned is better on **{wins}/{len(set(base) & set(tuned))}** languages."]
        for name, paths in (ev.get("samples") or {}).items():
            lines.append(f"- {name} samples: {len(paths)} wavs in `{data / 'samples' / name}`")
    if error:
        lines += ["", "## Evaluation error", "", "```", error.strip()[-3000:], "```"]

    lines += ["", "## What this does and does not show", "",
              "- Loss is a proxy for how well each model predicts real held-out speech; it is not "
              "intelligibility. Listen to `samples/`, then run ASR CER (IndicConformer, needs the "
              "gated model) before claiming quality.",
              "- Training data is real IndicVoices-R speech, capped per language (Gujarati and "
              "Sindhi have less than the cap); held-out clips come from its test split.",
              "- Urdu, Sindhi, Kashmiri, Santali, Manipuri (and partly Assamese) fall to byte "
              "fallback under the Sarvam tokenizer; expect them to lag."]
    (data / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {data / 'REPORT.md'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--baseline", default="baseline")
    ap.add_argument("--finetuned", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry", action="store_true", help="toy models (see prepare.py)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.dry and os.environ.get("HF_HUB_OFFLINE") == "1":
        from indic.overnight.prepare import use_staged_files
        use_staged_files()                 # the samples' vocoder, by staged path
    error = None
    if not args.report_only:
        try:
            evaluate(args)
        except Exception:
            error = traceback.format_exc()
            print(error, flush=True)
    report(args.data, error)
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(main())
