"""What a CPU can actually deliver, across the whole model-size range.

CPU inference inverts the architecture argument, and the inversion is not
intuitive. On a GPU the comparison is about parallelism: non-autoregressive flow
matching processes every frame at once and wins easily over token-by-token
decoding. On a CPU the binding constraint changes, and so does the answer.

    autoregressive, quantised   memory-bound. Each token re-reads the weights,
                                so cost is (weight bytes) x (tokens). Ternary
                                packing makes those bytes small, which is
                                exactly why a 1.7B ternary model can keep up.

    flow matching, float        compute-bound. Each of N function evaluations
                                touches every frame, so cost is
                                2 x params x frames x NFE -- and `frames` is
                                ~940 for ten seconds. At 16 NFE that is a
                                multi-TFLOP job on a processor measured in
                                hundreds of GFLOPS.

So the honest CPU ranking is not the GPU ranking, and the way to know is to
measure this machine rather than reason about it. This times every model the
repository can load, reports real-time factor and time to first audio, and
prints the arithmetic that explains each number.

    python -m indic.bench.cpu_speed
    python -m indic.bench.cpu_speed --models mini,micro,nano --threads 8

RTF below 1.0 means faster than real time. Below ~0.3 leaves enough headroom to
serve several streams from one machine, which is the point of going to CPU.
"""

import argparse
import json
import pathlib
import platform
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

# The repository's own range, smallest first. The 0.8 models are
# non-autoregressive and do not clone; kitten-tts-2 is the autoregressive
# cloning model. Both are here because the comparison is the finding.
MODELS = {
    "nano": ("KittenML/kitten-tts-nano-0.8-fp32", 15, "NAR, no cloning"),
    "micro": ("KittenML/kitten-tts-micro-0.8", 40, "NAR, no cloning"),
    "mini": ("KittenML/kitten-tts-mini-0.8", 80, "NAR, no cloning"),
    "kitten2": ("KittenML/kitten-tts-2", 1735, "AR ternary, cloning"),
}

SENTENCE = ("One day, a little girl named Lily found a needle in her room, "
            "and she carried it carefully to her mother.")


def measure_model(key, repo, text, repeats=3, warmups=1, device="cpu",
                  cache_dir=None, chunk_chars=None):
    """Real-time factor and time to first audio for one model."""
    import numpy as np

    from kittenml import KittenTTS

    load_started = time.perf_counter()
    model = KittenTTS(repo, device=device, cache_dir=cache_dir)
    load_seconds = time.perf_counter() - load_started

    kwargs = {}
    if key == "kitten2":
        kwargs["voice"] = "Bruno"
        if chunk_chars:
            kwargs["advanced"] = {"chunk_chars": chunk_chars, "seed": 0}
    else:
        kwargs["voice"] = model.available_voices[0]

    totals, firsts, durations = [], [], []
    for run in range(warmups + repeats):
        started = time.perf_counter()
        first = None
        pieces = []
        try:
            for piece in model.generate_stream(text, **kwargs):
                if first is None:
                    first = time.perf_counter() - started
                pieces.append(piece)
            audio = np.concatenate(pieces) if pieces else np.zeros(0)
        except (AttributeError, NotImplementedError, TypeError):
            # The 0.8 models synthesise in one pass; there is no first chunk
            # distinct from the whole utterance, and that is the honest answer
            # for time to first audio rather than a missing value.
            audio = model.generate(text, **kwargs)
            first = time.perf_counter() - started
        total = time.perf_counter() - started
        if run < warmups:
            continue
        totals.append(total)
        firsts.append(first if first is not None else total)
        # The 0.8 models are fixed at 24 kHz and expose no `sample_rate`; only
        # KittenTTS 2 carries one, from its config.
        durations.append(len(audio) / getattr(model, "sample_rate", 24000))

    audio_seconds = statistics.median(durations)
    del model
    return {
        "model": key, "repo": repo,
        "load_s": round(load_seconds, 2),
        "total_s": statistics.median(totals),
        "ttfa_s": statistics.median(firsts),
        "audio_s": audio_seconds,
        "rtf": statistics.median(totals) / max(audio_seconds, 1e-6),
        "realtime_x": max(audio_seconds, 1e-6) / statistics.median(totals),
        "repeats": repeats,
    }


def machine():
    """Enough about this CPU to make the numbers interpretable elsewhere."""
    import torch

    flags = []
    try:
        info = pathlib.Path("/proc/cpuinfo").read_text()
        for flag in ("avx2", "avx512f", "avx_vnni", "amx_int8", "amx_bf16"):
            if flag in info:
                flags.append(flag)
    except OSError:
        pass

    # A matmul measured here rather than a spec sheet: it is what actually
    # bounds a compute-bound model on this box.
    size = 2048
    a, b = torch.randn(size, size), torch.randn(size, size)
    for _ in range(2):
        a @ b
    started = time.perf_counter()
    for _ in range(8):
        a @ b
    seconds = (time.perf_counter() - started) / 8
    return {
        "cpu": platform.processor() or platform.machine(),
        "threads": torch.get_num_threads(),
        "flags": flags,
        "fp32_gflops": round(2 * size ** 3 / seconds / 1e9),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default="nano,micro,mini",
                    help=f"comma-separated from {', '.join(MODELS)}; "
                         "kitten2 is the 1.7B cloning model and is slow on CPU")
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--chunk-chars", type=int, default=0,
                    help="override the chunk budget for kitten2")
    ap.add_argument("--cache-dir")
    ap.add_argument("--out", type=pathlib.Path)
    args = ap.parse_args()

    import torch

    if args.threads:
        torch.set_num_threads(args.threads)
    info = machine()
    print(f"cpu        {info['cpu']}  {info['threads']} threads")
    print(f"isa        {', '.join(info['flags']) or 'unknown'}")
    print(f"measured   {info['fp32_gflops']} GFLOPS fp32 (2048 matmul)\n")

    wanted = [name.strip() for name in args.models.split(",") if name.strip()]
    rows = []
    print(f"{'model':>9}{'params':>9}{'RTF':>8}{'xRT':>8}{'TTFA s':>9}"
          f"{'audio s':>9}{'load s':>8}  note")
    for name in wanted:
        if name not in MODELS:
            print(f"  unknown model {name!r}")
            continue
        repo, params, note = MODELS[name]
        try:
            row = measure_model(name, repo, SENTENCE, repeats=args.repeats,
                                warmups=args.warmups, cache_dir=args.cache_dir,
                                chunk_chars=args.chunk_chars or None)
        except Exception as exc:
            print(f"{name:>9}{params:>8}M   FAILED {type(exc).__name__}: {exc}")
            rows.append({"model": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        row["params_m"] = params
        row["note"] = note
        rows.append(row)
        print(f"{name:>9}{params:>8}M{row['rtf']:>8.3f}{row['realtime_x']:>8.1f}"
              f"{row['ttfa_s']:>9.2f}{row['audio_s']:>9.2f}{row['load_s']:>8.1f}  {note}")

    report(rows, info)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"machine": info, "rows": rows}, indent=2))
        print(f"\nwrote {args.out}")


def report(rows, info):
    """What the numbers mean for a CPU-first design."""
    good = [r for r in rows if "rtf" in r]
    if not good:
        return
    print("\n== Reading these ==")
    for row in good:
        streams = int(1 / row["rtf"]) if row["rtf"] > 0 else 0
        verdict = ("comfortable" if row["rtf"] < 0.3 else
                   "usable" if row["rtf"] < 1.0 else "slower than real time")
        print(f"  {row['model']:<9} {row['realtime_x']:5.1f}x real time, "
              f"about {streams} concurrent stream(s) per machine — {verdict}")

    fastest = min(good, key=lambda r: r["rtf"])
    slowest = max(good, key=lambda r: r["rtf"])
    if fastest is not slowest:
        print(f"\n  {slowest['params_m'] / max(fastest['params_m'], 1):.0f}x the "
              f"parameters costs {slowest['rtf'] / max(fastest['rtf'], 1e-9):.0f}x "
              f"the time on this CPU.")
    print("\n  A cloning model has to be large enough to have learned identity, and\n"
          "  small enough to run here. Those pull in opposite directions, and this\n"
          "  table is where the trade is priced.")


if __name__ == "__main__":
    main()
