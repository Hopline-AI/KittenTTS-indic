"""Where should a 20-30 second voice sample actually be spent?

The product promise is that a user records half a minute and the model speaks in
their voice. That recording reaches the model by two completely separate routes,
and the code spends it very unevenly:

    LM prompt prefix   the WHOLE clip, as codec tokens
                       30 s = 750 tokens = 82 MiB of KV cache before a word is
                       generated -- and we measured the entire reference prefix
                       to be worth 0.062 nats of teacher-forced loss
    decoder conditioning   capped at DECODER_COND_SECONDS = 10 (vocoder.py:30),
                       or STUDENT_REF_SECONDS = 4.0 for the fast decoder

So two thirds of a 30 second recording is discarded by the stage that carries
timbre, while all of it is paid for by the stage that appears to barely use it.
That is a strange allocation, and nobody has checked whether it is the right
one -- the 0.062 nats measurement cannot settle it, because teacher-forced loss
reads a transcript the model is being fed anyway and may simply be blind to
identity.

This measures it the only way that can answer it: generate, then score speaker
similarity against the reference with an embedder that is **not** the one the
model conditions on. Using the model's own x-vector would ask whether the
conditioning vector survived its own pipeline; WavLM asks whether it sounds like
the person.

    python -m indic.bench.reference_budget --langs hi,en --speakers 3

Each cell of the grid is one generation, so cost is (prefix options) x (decoder
options) x speakers x languages. Keep it small; the effect sizes here are large
or they do not matter.
"""

import argparse
import itertools
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.langs import LANGUAGES

# The codec's frame rate, so a seconds budget converts to a token count and to
# the KV cache it will occupy. 112 KiB/token measured from the architecture:
# 2 (K and V) x 28 layers x 8 kv-heads x 128 head_dim x 2 bytes.
TOKENS_PER_SECOND = 25
KV_KIB_PER_TOKEN = 112

# A sentence per language that is long enough to judge a voice on and short
# enough to stay in one chunk. Deliberately plain: this measures identity, not
# the model's handling of hard text.
TARGETS = {
    "hi": "समुद्र के किनारे शाम की हवा बहुत ठंडी थी, और हम देर तक वहीं बैठे रहे।",
    "bn": "সন্ধ্যার আলোয় নদীর ধারে বসে আমরা অনেকক্ষণ গল্প করেছিলাম।",
    "ta": "மாலை நேரத்தில் கடற்கரையில் காற்று மிகவும் குளிர்ச்சியாக இருந்தது.",
    "te": "సాయంత్రం సమయంలో సముద్రతీరంలో గాలి చాలా చల్లగా ఉంది.",
    "mr": "संध्याकाळी समुद्रकिनारी वारा खूप थंड होता आणि आम्ही तिथेच बसून राहिलो.",
    "en": "The evening air by the sea was unusually cold, and we stayed there talking for a long while.",
}


def build_reference(lang, speaker_index, target_seconds, cache_dir=None):
    """A reference clip of about `target_seconds`, from one FLEURS speaker.

    FLEURS has no speaker-id column, but it records each sentence several times
    and names the file by a per-recording hash, so clips cannot be grouped by
    speaker from the metadata alone. Grouping is therefore done the only
    reliable way available: cluster the pool by the repo's own speaker encoder
    and take the tightest cluster. Concatenation is needed because single FLEURS
    clips top out around 26 s and most are under 15.
    """
    import numpy as np
    import soundfile as sf

    from indic.eval.testsets import ensure_fleurs_audio, fleurs_audio_rows

    # fleurs_audio_rows returns {sentence_id: (file_name, seconds)}. Longest
    # first: fewer joins to reach the budget means fewer seams, and a seam inside
    # a reference is a discontinuity the speaker encoder reads as a second voice.
    rows = fleurs_audio_rows(lang, "dev")
    ordered = sorted(rows.values(), key=lambda row: -(row[1] or 0))[:60]
    names = [name for name, _ in ordered]
    # ensure_fleurs_audio returns the directory it extracted into, not the files.
    directory = ensure_fleurs_audio(lang, "dev", names, cache_dir=cache_dir)
    paths = [directory / name for name in names]
    clips = []
    for path in paths:
        try:
            audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
        except Exception:
            continue
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        clips.append((path, audio, rate))
    if not clips:
        raise RuntimeError(f"no readable FLEURS audio for {lang}")

    # Group by voice using the model's own encoder: cheap, and the only signal
    # available. The chosen group only has to be one person, not a known person.
    from indic.eval.metrics import WavLMSpeakerEmbedder
    embedder = WavLMSpeakerEmbedder(device="cpu")
    vectors = np.stack([embedder.embed(str(p)) for p, _, _ in clips])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-9
    similarity = vectors @ vectors.T

    order = np.argsort(-similarity.sum(axis=1))
    seed = int(order[speaker_index % len(order)])
    neighbours = np.argsort(-similarity[seed])

    rate = clips[seed][2]
    want = int(target_seconds * rate)
    pieces, used = [], []
    for index in neighbours:
        if similarity[seed][index] < 0.80 and pieces:
            break
        pieces.append(clips[index][1])
        used.append(str(clips[index][0]))
        if sum(len(p) for p in pieces) >= want:
            break
    audio = np.concatenate(pieces)[:want]
    return audio, rate, used


def grid_generate(model, reference_path, target_text, prefix_seconds,
                  decoder_seconds, seed=0):
    """One generation under a specific reference budget.

    Both budgets are imposed at the two places the reference actually enters,
    rather than by trimming the file: the LM prefix is the token list returned
    by `encode_reference`, and the decoder's share is the module constant
    `DECODER_COND_SECONDS` that `reference_conditioning` reads. Both caches are
    cleared first, or the second cell of the grid silently reuses the first
    cell's conditioning.
    """
    from kittenml.kittentts2 import vocoder as vocoder_module

    codec = model.codec
    model._reference_cache.clear()
    codec._ref_cache.clear()

    original_encode = codec.encode_reference
    original_seconds = vocoder_module.DECODER_COND_SECONDS

    def truncated_encode(wav_path):
        tokens = original_encode(wav_path)
        if prefix_seconds is None:
            return tokens
        return tokens[:int(prefix_seconds * TOKENS_PER_SECOND)]

    codec.encode_reference = truncated_encode
    vocoder_module.DECODER_COND_SECONDS = float(decoder_seconds)
    try:
        started = time.time()
        audio = model.generate(target_text, reference=str(reference_path),
                               advanced={"seed": seed})
        elapsed = time.time() - started
    finally:
        codec.encode_reference = original_encode
        vocoder_module.DECODER_COND_SECONDS = original_seconds
        model._reference_cache.clear()
        codec._ref_cache.clear()
    return audio, elapsed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--langs", default="hi,en")
    ap.add_argument("--speakers", type=int, default=2, help="voices per language")
    ap.add_argument("--reference-seconds", type=float, default=30.0,
                    help="length of the reference built for each voice")
    ap.add_argument("--prefix-seconds", default="0,5,10,30",
                    help="LM prompt budgets to try; 'full' means the whole clip")
    ap.add_argument("--decoder-seconds", default="4,10,20,30")
    ap.add_argument("--model", default="KittenML/kitten-tts-2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir")
    ap.add_argument("--audio-out", type=pathlib.Path,
                    default=pathlib.Path(".cache/reference_budget"))
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf

    from indic.eval.metrics import WavLMSpeakerEmbedder
    from kittenml import KittenTTS

    langs = [code.strip() for code in args.langs.split(",") if code.strip()]
    prefixes = [None if value.strip() == "full" else float(value)
                for value in args.prefix_seconds.split(",")]
    decoders = [float(value) for value in args.decoder_seconds.split(",")]
    args.audio_out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.model} on {args.device} ...", flush=True)
    model = KittenTTS(args.model, device=args.device, cache_dir=args.cache_dir)
    embedder = WavLMSpeakerEmbedder(device=args.device)

    results = []
    for lang in langs:
        if lang not in TARGETS:
            print(f"  no target sentence for {lang}, skipping")
            continue
        for speaker in range(args.speakers):
            audio, rate, used = build_reference(lang, speaker,
                                                args.reference_seconds,
                                                args.cache_dir)
            reference_path = args.audio_out / f"ref-{lang}-{speaker}.wav"
            sf.write(str(reference_path), audio, rate)
            reference_vector = embedder.embed(str(reference_path))
            seconds = len(audio) / rate
            print(f"\n== {LANGUAGES[lang].name} voice {speaker}: "
                  f"{seconds:.1f}s reference from {len(used)} clips ==", flush=True)
            print(f"{'prefix':>8}{'decoder':>9}{'SECS':>8}{'tokens':>8}"
                  f"{'KV MiB':>8}{'gen s':>8}")

            for prefix, decoder in itertools.product(prefixes, decoders):
                try:
                    generated, elapsed = grid_generate(
                        model, reference_path, TARGETS[lang], prefix, decoder)
                except Exception as exc:
                    print(f"{str(prefix):>8}{decoder:>9}  FAILED {type(exc).__name__}: {exc}")
                    results.append({"language": lang, "speaker": speaker,
                                    "prefix_seconds": prefix,
                                    "decoder_seconds": decoder,
                                    "error": f"{type(exc).__name__}: {exc}"})
                    continue
                name = (f"{lang}-{speaker}-p{'full' if prefix is None else int(prefix)}"
                        f"-d{int(decoder)}.wav")
                out_path = args.audio_out / name
                sf.write(str(out_path), generated, model.sample_rate)
                secs = float(np.dot(
                    *[v / (np.linalg.norm(v) + 1e-9)
                      for v in (embedder.embed(str(out_path)), reference_vector)]))
                tokens = (int(seconds * TOKENS_PER_SECOND) if prefix is None
                          else int(prefix * TOKENS_PER_SECOND))
                kv_mib = tokens * KV_KIB_PER_TOKEN / 1024
                print(f"{('full' if prefix is None else int(prefix)):>8}"
                      f"{int(decoder):>9}{secs:>8.3f}{tokens:>8}"
                      f"{kv_mib:>8.1f}{elapsed:>8.1f}")
                results.append({
                    "language": lang, "speaker": speaker,
                    "prefix_seconds": prefix, "decoder_seconds": decoder,
                    "secs": secs, "prefix_tokens": tokens, "kv_mib": kv_mib,
                    "generate_seconds": elapsed, "wav": str(out_path),
                    "reference_seconds": seconds,
                })

    report(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"model": args.model, "reference_seconds": args.reference_seconds,
             "results": results}, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


def report(results):
    """Marginal effect of each budget, averaged over voices and languages."""
    scored = [r for r in results if "secs" in r]
    if not scored:
        print("\nno successful generations")
        return

    def mean(rows, key="secs"):
        return sum(r[key] for r in rows) / len(rows)

    print("\n== Marginal effect of the LM prompt prefix ==")
    print("(speaker similarity, averaged over decoder budgets and voices)")
    for prefix in sorted({r["prefix_seconds"] for r in scored},
                         key=lambda v: -1 if v is None else v):
        rows = [r for r in scored if r["prefix_seconds"] == prefix]
        label = "full" if prefix is None else f"{int(prefix)}s"
        print(f"  prefix {label:>5}   SECS {mean(rows):.3f}   "
              f"KV {mean(rows, 'kv_mib'):6.1f} MiB   gen {mean(rows, 'generate_seconds'):.1f}s")

    print("\n== Marginal effect of the decoder conditioning budget ==")
    print("(speaker similarity, averaged over prompt budgets and voices)")
    for decoder in sorted({r["decoder_seconds"] for r in scored}):
        rows = [r for r in scored if r["decoder_seconds"] == decoder]
        print(f"  decoder {int(decoder):>3}s   SECS {mean(rows):.3f}")

    best = max(scored, key=lambda r: r["secs"])
    cheapest = min((r for r in scored if r["secs"] >= best["secs"] - 0.02),
                   key=lambda r: r["kv_mib"])
    print("\n== Best and cheapest-within-0.02 ==")
    for label, row in (("best   ", best), ("cheapest", cheapest)):
        prefix = "full" if row["prefix_seconds"] is None else f"{int(row['prefix_seconds'])}s"
        print(f"  {label} prefix {prefix:>5}  decoder {int(row['decoder_seconds']):>3}s  "
              f"SECS {row['secs']:.3f}  KV {row['kv_mib']:.1f} MiB")


if __name__ == "__main__":
    main()
