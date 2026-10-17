"""Time to first audio, real-time factor, and what the chunk budget costs.

Our worst published number is 17 seconds to first audio. It has been read as an
architecture problem -- autoregressive decoding is slow, so move to something
non-autoregressive -- and that reading is wrong in a way worth measuring before
anyone acts on it.

The model does not stream tokens. It generates a whole chunk and then vocodes
it, so time to first audio is the time to synthesise the *first chunk* in full.
`text.py` sets that chunk at 380 characters, which at the checkpoint's measured
20 characters/second of speech is about 19 seconds of audio: roughly 475
autoregressive steps before a single sample is emitted. The 17 seconds is the
chunk policy, not the decoder.

That matters for a decision. A codec at half the token rate would take 17 s to
8.5 s and still be unusable, while a 95-character chunk gets there with no model
change at all. So this measures time to first audio against the one knob that
actually moves it, and reports the cost of turning it.

**The cost is real and has to be in the same table.** Short chunks mean more
seams, and `text.py` says the model is least stable on short inputs -- it
over-generates and holds a token instead of terminating. So every row also
carries total wall clock, the number of chunks, and the ratio of generated
duration to what the text should take. A row that wins on latency by truncating
or by running away is not a win, and the duration ratio is what shows it.

    python -m indic.bench.efficiency --chunks 380,190,95,60 --lang hi

Timing convention follows `docs/indic/research/benchmark.md`: three untimed
warmups, median over repeats, and the measurement runs on one idle GPU. Numbers
from a laptop card are for *ranking chunk sizes*, never for publication.
"""

import argparse
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from indic.langs import LANGUAGES

# Measured delivery rate for this checkpoint (model.py:35). Converts a character
# budget into the seconds of audio a chunk represents, which is the quantity
# that actually sets latency.
CHARS_PER_SECOND = 20.0
TOKENS_PER_SECOND = 25

# Long enough to produce several chunks at every budget, so chunk count and
# seam cost are visible rather than inferred.
PASSAGES = {
    "hi": ("पहाड़ों की तरफ जाने वाली सड़क सुबह से ही धुंध में डूबी हुई थी। हम लोग "
           "धीरे-धीरे ऊपर चढ़ते रहे, और हर मोड़ पर घाटी का नज़ारा थोड़ा और खुलता गया। "
           "दोपहर तक हवा साफ़ हो चुकी थी और दूर बर्फ़ से ढकी चोटियाँ दिखाई देने लगीं। "
           "हमने एक छोटे से गाँव में रुककर चाय पी और वहाँ के लोगों से बातें कीं।"),
    "ta": ("மலைப்பகுதிக்குச் செல்லும் சாலை காலை முதலே பனியில் மூழ்கியிருந்தது. "
           "நாங்கள் மெதுவாக மேலே ஏறிக்கொண்டே இருந்தோம், ஒவ்வொரு திருப்பத்திலும் "
           "பள்ளத்தாக்கின் காட்சி சற்று விரிந்தது. மதியம் வாக்கில் காற்று தெளிவாகியது."),
    "en": ("The road up into the hills was buried in mist from early morning. We "
           "climbed slowly, and at every turn a little more of the valley opened "
           "below us. By midday the air had cleared and the snow line was visible "
           "in the distance. We stopped at a small village for tea and talked to "
           "the people there for a while."),
}


def measure(model, text, chunk_chars, repeats=3, warmups=1, voice=None,
            reference=None, seed=0):
    """Time to first audio and total wall clock at one chunk budget.

    Timed through `generate_stream`, which yields each chunk as it is finished,
    so the first yield *is* time to first audio. Calling `generate` instead would
    measure the whole passage and report it as latency, which is how a
    non-streaming system flatters itself.
    """
    import numpy as np

    kwargs = {"advanced": {"chunk_chars": chunk_chars, "seed": seed}}
    if reference:
        kwargs["reference"] = str(reference)
    elif voice:
        kwargs["voice"] = voice

    first_times, total_times, chunk_counts, durations = [], [], [], []
    for run in range(warmups + repeats):
        started = time.perf_counter()
        first = None
        pieces = []
        for piece in model.generate_stream(text, **kwargs):
            if first is None:
                first = time.perf_counter() - started
            pieces.append(piece)
        total = time.perf_counter() - started
        if run < warmups:
            continue
        audio = np.concatenate(pieces) if pieces else np.zeros(0)
        first_times.append(first if first is not None else total)
        total_times.append(total)
        chunk_counts.append(len(pieces))
        durations.append(len(audio) / model.sample_rate)

    audio_seconds = statistics.median(durations)
    expected = len(text) / CHARS_PER_SECOND
    return {
        "chunk_chars": chunk_chars,
        "chunk_seconds": round(chunk_chars / CHARS_PER_SECOND, 1),
        "ttfa_s": statistics.median(first_times),
        "ttfa_min": min(first_times),
        "ttfa_max": max(first_times),
        "total_s": statistics.median(total_times),
        "chunks": statistics.median(chunk_counts),
        "audio_s": audio_seconds,
        # RTF is wall clock over OUTPUT duration, so a run that truncates looks
        # fast. duration_ratio is the guard: well under 1 means the model stopped
        # early, well over means it ran away, and either voids the latency win.
        "rtf": statistics.median(total_times) / max(audio_seconds, 1e-6),
        "duration_ratio": audio_seconds / max(expected, 1e-6),
        "repeats": repeats,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chunks", default="380,190,130,95,60",
                    help="chunk_chars budgets to compare; 380 is the default")
    ap.add_argument("--lang", default="hi")
    ap.add_argument("--voice", default="Bruno")
    ap.add_argument("--reference", help="clone from this clip instead of a voice")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--model", default="KittenML/kitten-tts-2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cache-dir")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    import torch

    from kittenml import KittenTTS

    if args.lang not in PASSAGES:
        sys.exit(f"no passage for {args.lang}; have {sorted(PASSAGES)}")
    budgets = [int(value) for value in args.chunks.split(",")]
    text = PASSAGES[args.lang]

    device_name = (torch.cuda.get_device_name(0)
                   if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"loading {args.model} on {args.device} ({device_name}) ...", flush=True)
    model = KittenTTS(args.model, device=args.device, cache_dir=args.cache_dir)

    print(f"\n{LANGUAGES[args.lang].name}, {len(text)} characters, "
          f"{args.repeats} timed repeats after {args.warmups} warmup\n")
    print(f"{'chunk':>7}{'~audio':>8}{'TTFA s':>9}{'total s':>9}{'chunks':>8}"
          f"{'RTF':>7}{'dur×':>7}")
    rows = []
    for budget in budgets:
        row = measure(model, text, budget, repeats=args.repeats,
                      warmups=args.warmups, voice=args.voice,
                      reference=args.reference)
        rows.append(row)
        print(f"{row['chunk_chars']:>7}{row['chunk_seconds']:>7.1f}s"
              f"{row['ttfa_s']:>9.2f}{row['total_s']:>9.2f}{row['chunks']:>8.0f}"
              f"{row['rtf']:>7.2f}{row['duration_ratio']:>7.2f}")

    report(rows)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(
            {"model": args.model, "device": device_name, "language": args.lang,
             "characters": len(text), "rows": rows}, indent=2, ensure_ascii=False))
        print(f"\nwrote {args.out}")


def report(rows):
    """What the chunk budget buys and what it costs."""
    if not rows:
        return
    baseline = max(rows, key=lambda r: r["chunk_chars"])
    # A duration ratio far from 1 means the model truncated or ran away, so the
    # row's latency is not a latency a user would ever see.
    sound = [r for r in rows if 0.7 <= r["duration_ratio"] <= 1.6]
    print("\n== Against the 380-character default ==")
    for row in rows:
        speedup = baseline["ttfa_s"] / max(row["ttfa_s"], 1e-6)
        throughput = row["total_s"] / max(baseline["total_s"], 1e-6)
        warning = "" if row in sound else "   <- duration ratio off; not a real win"
        print(f"  chunk {row['chunk_chars']:>4}  TTFA {row['ttfa_s']:5.2f}s "
              f"({speedup:4.1f}x faster)   total {throughput:4.2f}x the time"
              f"{warning}")
    if sound:
        best = min(sound, key=lambda r: r["ttfa_s"])
        print(f"\n  Lowest sound latency: chunk {best['chunk_chars']} at "
              f"{best['ttfa_s']:.2f}s, costing {best['total_s'] / baseline['total_s']:.2f}x "
              f"the total wall clock and {best['chunks']:.0f} seams.")
    print("\n  Latency here is a chunking decision. Compare against what a codec "
          "change\n  could buy before treating it as an architecture problem.")


if __name__ == "__main__":
    main()
