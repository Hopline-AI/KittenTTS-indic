"""`indic.train.prepare_shards`, with stand-in encoders when KINDIC_DRY=1.

The dry encoders keep the real `encode()` (duration filter, text normalisation,
row layout) and fake only the three models, sized for `loop.py --toy`
(64 audio ids, 150 text ids), so a laptop can run the whole pipeline.
"""

import os
import pathlib
import sys

from indic.train import prepare_shards


class DryTokenizer:
    """Character ids, small enough for the toy model's 150-row text table."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) % 150 for c in text]}


class DryEncoders(prepare_shards.Encoders):
    class _Codec:
        @staticmethod
        def encode_reference(path):
            import random

            import soundfile as sf
            seconds = sf.info(path).duration
            rng = random.Random(str(path))
            return [rng.randrange(64) for _ in range(int(seconds * 25))]

    @property
    def codec(self):
        return self._Codec()

    @property
    def speaker(self):
        import random
        return lambda path: [random.Random(str(path)).gauss(0, 1) for _ in range(512)]

    @property
    def tokenizer(self):
        return DryTokenizer()


def staged_path(repo_id, filename=None, **_):
    """A staged file (or, with no filename, a snapshot dir) by path, never by hub lookup.

    ponytail: offline hf_hub_download failed on eX3 with a LocalEntryNotFoundError
    that names no file; a glob over the staged snapshots names what is missing.
    """
    hub = pathlib.Path(os.environ["HF_HOME"]) / "hub"
    snaps = sorted(hub.glob(f"models--{repo_id.replace('/', '--')}/snapshots/*/"))
    hits = [d / filename for d in snaps if (d / filename).exists()] if filename else snaps
    if not hits:
        raise FileNotFoundError(f"{repo_id} {filename or ''} is not staged under {hub} "
                                f"(snapshots: {[str(d) for d in snaps]})")
    return str(hits[0])


def use_staged_files():
    """Point every download the encoders make at the staged cache."""
    import huggingface_hub
    # Before kittenml.kittentts2.vocoder is imported, so its
    # `from huggingface_hub import hf_hub_download` binds to this.
    huggingface_hub.hf_hub_download = staged_path
    if "kittenml.kittentts2.vocoder" in sys.modules:
        sys.modules["kittenml.kittentts2.vocoder"].hf_hub_download = staged_path

    class StagedEncoders(prepare_shards.Encoders):
        @property
        def tokenizer(self):
            if self._tokenizer is None:
                from transformers import AutoTokenizer
                name = self.tokenizer_name     # the shards still record the repo id
                local = name if os.path.isdir(name) else staged_path(name, "tokenizer_config.json")
                self._tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(local) if
                                                                local.endswith(".json") else local)
            return self._tokenizer

    prepare_shards.Encoders = StagedEncoders


if __name__ == "__main__":
    if os.environ.get("KINDIC_DRY") == "1":
        prepare_shards.Encoders = DryEncoders
    elif os.environ.get("HF_HUB_OFFLINE") == "1":
        use_staged_files()
    sys.exit(prepare_shards.main())
