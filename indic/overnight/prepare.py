"""`indic.train.prepare_shards`, with stand-in encoders when KINDIC_DRY=1.

The dry encoders keep the real `encode()` (duration filter, text normalisation,
row layout) and fake only the three models, sized for `loop.py --toy`
(64 audio ids, 150 text ids), so a laptop can run the whole pipeline.
"""

import os
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


if __name__ == "__main__":
    if os.environ.get("KINDIC_DRY") == "1":
        prepare_shards.Encoders = DryEncoders
    sys.exit(prepare_shards.main())
