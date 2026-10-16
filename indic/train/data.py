"""The training example, built by the same code that builds a generation prompt.

A speech LM fine-tune fails silently when the training sequence differs from the
inference sequence -- the loss falls, the model learns the layout it was shown,
and generation is subtly wrong in a way no training metric reports. So the
prompt half of every example comes from `prompt.py` itself rather than from a
restatement of it here:

    pos 0    [START], whose *input embedding* is overwritten by spk_proj
    ...      <reference_text_start> ref transcript <reference_text_end>
             <reference_speech_start> ref audio <reference_speech_end>
    ...      [emotion control ids]
             <text_start> target text <final_seg> <speech_start>
    ---- everything above is -100 ----
             target audio tokens
             <speech_end>

Position 0 is worth stating plainly: the `[START]` id there is never embedded.
`speaker.py` replaces that position's input embedding with the projected speaker
vector, so the token is a placeholder holding a slot. An example whose speaker
embedding fails to reach the model therefore trains on a *different input
sequence*, not merely a less-conditioned one -- see the speaker check in
`selfcheck.py`.

Loss covers the target audio tokens and `<speech_end>` and nothing else. The
first supervised position is the first audio token, which the model predicts
from `<speech_start>` -- the same position inference generates from.

There is no corpus yet, so `SyntheticDataset` fabricates well-formed examples.
That is what makes the pipeline provable today: every check in `selfcheck.py`
runs against it.
"""

import numpy as np
import torch

from . import load_kittentts2

_prompt = load_kittentts2("prompt")
_tokens = load_kittentts2("tokens")

TokenMap = _tokens.TokenMap

# Positions the loss ignores. torch's cross_entropy default, and what
# `Qwen3ForCausalLM.forward` passes to it.
IGNORE_INDEX = -100


def build_example(token_map, target_text_ids, target_audio_tokens,
                  reference_text_ids=None, reference_audio_tokens=None,
                  emotion_ids=None):
    """One training example as (input_ids, labels), both plain lists.

    `reference_text_ids=None` drops the in-context reference, which inference
    also supports (`build_generation_prompt(reference_prefix=None)`). Training a
    fraction of examples that way is what keeps the no-reference path working;
    a run that never sees it will degrade there without ever being measured.

    The supervised span is computed from the prompt's length, not found by
    scanning the finished sequence for markers. That matters for the self-check:
    the mask and the check that verifies it are then two independent
    derivations, and an off-by-one in one of them does not hide in the other.
    """
    # Both branches go through prompt.py, which is not a style preference:
    # `[START]` lives *inside* `build_reference_prefix`, so dropping the
    # reference is not concatenating one less block -- `build_generation_prompt`
    # emits its own. Assembling either path by hand gets two [START]s or none.
    # Note also that `text_end_id` (151672) exists in the token map and is
    # emitted nowhere: the target text is closed by `<final_seg>` and the
    # reference by `reference_text_end_id`. That is the layout the checkpoint
    # was prompted with, not an oversight to correct.
    reference_prefix = None
    if reference_text_ids is not None and reference_audio_tokens is not None:
        reference_prefix = _prompt.build_reference_prefix(
            token_map, list(reference_text_ids), list(reference_audio_tokens))

    prompt_ids = _prompt.build_generation_prompt(
        token_map, list(target_text_ids), reference_prefix=reference_prefix,
        emotion_ids=list(emotion_ids) if emotion_ids else None)

    supervised = [token_map.audio_id_base + t for t in target_audio_tokens]
    supervised.append(token_map.speech_end_id)

    input_ids = [*prompt_ids, *supervised]
    labels = [IGNORE_INDEX] * len(prompt_ids) + supervised
    return input_ids, labels


def tiny_token_map(n_text=150, n_audio=64):
    """A `TokenMap` with the released layout at a size that fits a toy model.

    Same block order as the real checkpoint -- text, six named specials, the
    audio range, then the marker rows at the tail -- because `prompt.py` and the
    logit masking both read positions out of that layout. Shrinking it is the
    only way the self-check runs a real forward pass in seconds on a CPU.
    """
    specials = n_text
    audio_base = specials + 6
    markers = audio_base + n_audio
    return TokenMap(
        audio_id_base=audio_base, num_audio_tokens=n_audio,
        speech_start_id=specials, speech_end_id=specials + 1,
        text_start_id=specials + 2, text_end_id=specials + 3,
        start_id=specials + 4, stop_id=specials + 5,
        final_seg_id=markers, reference_speech_start_id=markers + 1,
        reference_speech_end_id=markers + 2, reference_text_start_id=markers + 3,
        reference_text_end_id=markers + 4)


class SyntheticDataset(torch.utils.data.Dataset):
    """Well-formed examples with no corpus behind them.

    The token ids are noise, so nothing here says anything about speech. What it
    does say is that the sequence *layout* is assembled, masked, padded, batched
    and differentiated correctly, which is the part that can be proved before any
    data exists -- and the part that is hardest to debug once real data hides it.

    Deterministic per index: example `i` is the same example on every epoch, in
    every process and after a restart, so a resume check can compare losses
    step for step.
    """

    def __init__(self, token_map, size=64, speaker_dim=512, seed=0,
                 text_len=(4, 12), audio_len=(8, 24), reference_probability=0.8,
                 n_speakers=4):
        self.token_map = token_map
        self.size = size
        self.speaker_dim = speaker_dim
        self.seed = seed
        self.text_len = text_len
        self.audio_len = audio_len
        self.reference_probability = reference_probability
        self.n_speakers = n_speakers

    def __len__(self):
        return self.size

    def speaker_embedding(self, speaker):
        """Unit-norm, as `SpeakerEncoder` returns and `spk_proj` was trained on.

        Drawn from the speaker index alone, so the same speaker is the same
        vector across examples -- otherwise "speaker conditioning changes the
        loss" would be measuring fresh noise rather than identity.
        """
        rng = np.random.default_rng([self.seed, 0xBEEF, speaker])
        raw = rng.standard_normal(self.speaker_dim).astype(np.float32)
        return torch.from_numpy(raw / np.linalg.norm(raw))

    def __getitem__(self, index):
        if not 0 <= index < self.size:
            raise IndexError(index)
        tm = self.token_map
        rng = np.random.default_rng([self.seed, index])
        # Text ids stay below the specials block: an id drawn from the audio
        # range would put audio tokens inside the text span and quietly make the
        # synthetic example malformed in exactly the way the suite is supposed
        # to catch.
        n_text = int(rng.integers(*self.text_len))
        n_audio = int(rng.integers(*self.audio_len))
        text_ids = rng.integers(0, tm.speech_start_id, size=n_text).tolist()
        audio = rng.integers(0, tm.num_audio_tokens, size=n_audio).tolist()

        reference = rng.random() < self.reference_probability
        reference_text = reference_audio = None
        if reference:
            reference_text = rng.integers(0, tm.speech_start_id,
                                          size=int(rng.integers(*self.text_len))).tolist()
            reference_audio = rng.integers(0, tm.num_audio_tokens,
                                           size=int(rng.integers(*self.audio_len))).tolist()

        input_ids, labels = build_example(tm, text_ids, audio,
                                          reference_text_ids=reference_text,
                                          reference_audio_tokens=reference_audio)
        speaker = int(rng.integers(0, self.n_speakers))
        return {"input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "speaker_embedding": self.speaker_embedding(speaker),
                "speaker": speaker}


class ShardDataset(torch.utils.data.Dataset):
    """Training examples assembled from precomputed parquet shards.

    The shards hold one row per *utterance* (see `shard_schema`), not one per
    training example: codec tokens, speaker embeddings and text ids are all
    computed offline, so nothing here touches audio. What this class adds is the
    part that must not be frozen at preprocessing time -- which utterance serves
    as the reference for which target.

    **Pairing happens here, per epoch.** The reference is a different utterance
    by the same speaker, drawn from `(seed, epoch, index)`, so it is identical
    across ranks and after a restart but varies between epochs. Freezing the
    pairing on disk would show the model one reference per target forever; this
    costs nothing and is free regularisation.

    **Everything is held in memory, deliberately.** 900 hours is about 325k
    utterances: ~325 MB of codec tokens, ~41 MB of text ids and ~665 MB of
    speaker embeddings, so roughly 1 GB -- nothing on a node with 32 GB, and it
    buys true random access, which a row-group reader cannot give a shuffling
    sampler without thrashing. Ragged columns are stored as one flat buffer plus
    offsets rather than a list of arrays, because 325k small Python objects cost
    more than the data. A corpus ten times this size needs the lazy path
    instead; `float16` embeddings halve the largest term first.
    """

    def __init__(self, shards, token_map, tokenizer=None, *,
                 reference_probability=0.8, reference_max_tokens=250,
                 max_tokens=None, min_audio_tokens=25, max_audio_tokens=750,
                 max_cer=0.15, languages=None, language_temperature=0.0,
                 max_epochs_per_language=4, seed=0, epoch=0,
                 embedding_dtype="float32"):
        from indic.train.shard_schema import check_compatible

        paths = _resolve_shards(shards)
        for path in paths:
            check_compatible(path, tokenizer)

        self.token_map = token_map
        self.reference_probability = reference_probability
        self.reference_max_tokens = reference_max_tokens
        self.max_tokens = max_tokens
        self.seed = seed

        self._load(paths, embedding_dtype)
        self._keep = self._filter(min_audio_tokens, max_audio_tokens, max_cer,
                                  languages)
        self._by_speaker = self._index_speakers()
        self.language_temperature = language_temperature
        self.max_epochs_per_language = max_epochs_per_language
        self.set_epoch(epoch)

    # ── loading ───────────────────────────────────────────────────────────
    def _load(self, paths, embedding_dtype):
        """Read every shard into flat buffers once."""
        import pyarrow.parquet as pq

        text_flat, text_off = [], [0]
        audio_flat, audio_off = [], [0]
        embeddings = []
        self.utt_id, self.language, self.speaker_id, self.emotion = [], [], [], []
        cer, duration = [], []

        for path in paths:
            table = pq.read_table(str(path))
            self.utt_id += table["utt_id"].to_pylist()
            self.language += table["language"].to_pylist()
            self.speaker_id += table["speaker_id"].to_pylist()
            self.emotion += table["emotion"].to_pylist()
            cer += table["cer"].to_pylist()
            duration += table["duration_s"].to_pylist()
            for column, flat, offsets in (("text_ids", text_flat, text_off),
                                          ("audio_tokens", audio_flat, audio_off)):
                for row in table[column]:
                    values = row.values.to_numpy(zero_copy_only=False)
                    flat.append(values.astype(np.int32, copy=False))
                    offsets.append(offsets[-1] + len(values))
            embeddings.append(
                np.asarray(table["speaker_embedding"].to_numpy(zero_copy_only=False)
                           .tolist(), dtype=embedding_dtype))

        self.text_flat = np.concatenate(text_flat) if text_flat else np.zeros(0, np.int32)
        self.audio_flat = np.concatenate(audio_flat) if audio_flat else np.zeros(0, np.int32)
        self.text_off = np.asarray(text_off, dtype=np.int64)
        self.audio_off = np.asarray(audio_off, dtype=np.int64)
        self.embeddings = np.concatenate(embeddings) if embeddings else np.zeros((0, 512))
        self.cer = np.asarray(cer, dtype=np.float32)
        self.duration_s = np.asarray(duration, dtype=np.float32)
        self.n_rows = len(self.utt_id)

    def _filter(self, min_audio, max_audio, max_cer, languages):
        """Row indices worth training on.

        A clip too short carries no prosody and a clip too long blows the token
        budget; `cer` is the preprocessing job's own ASR agreement score, and
        -1 means it was never measured, which must not be read as perfect.
        """
        lengths = np.diff(self.audio_off)
        keep = (lengths >= min_audio) & (lengths <= max_audio)
        measured = self.cer >= 0
        keep &= ~measured | (self.cer <= max_cer)
        if languages:
            wanted = set(languages)
            keep &= np.array([lang in wanted for lang in self.language], dtype=bool)
        return np.nonzero(keep)[0]

    def _index_speakers(self):
        """speaker -> the kept rows belonging to them, for reference pairing."""
        by_speaker = {}
        for row in self._keep:
            by_speaker.setdefault(self.speaker_id[row], []).append(int(row))
        return {k: np.asarray(v, dtype=np.int64) for k, v in by_speaker.items()}

    # ── epoch state ───────────────────────────────────────────────────────
    def set_epoch(self, epoch):
        """Re-pair references and re-balance languages for `epoch`.

        Call it from the training loop exactly as `DistributedSampler.set_epoch`
        is called, and before the sampler, since this changes `len(self)`.
        """
        self.epoch = int(epoch)
        self._order = self._resample_languages()

    def _resample_languages(self):
        """Temperature sampling across languages, with a hard repeat cap.

        With 175 hours of Hindi against 9 of Tamil, natural sampling shows the
        model Tamil in one batch out of twenty, and Tamil is exactly the language
        this project exists to fix. Temperature `t` samples language `l` in
        proportion to `count_l ** t`: 1.0 is the natural distribution, 0.0 is
        uniform across languages.

        **The default is 0.0, not the 0.3 that the multilingual-text literature
        would suggest.** Both controlled sweeps on *speech* go flatter than text
        practice: with exact uniform sampling over four languages, Tamil WER went
        139.96 -> 39.19 while English and Mandarin also improved, so the flat
        setting was not paid for by the head languages.

        **`max_epochs_per_language` is the part that makes flat sampling safe.**
        Uniform sampling over a 20:1 imbalance means passing over the small
        language twenty times per pass over the large one, and repeating a small
        fraction of the data very many times can cost as much as halving the
        model -- a harm that shows up specifically at large model size and long
        training, which is us. The cap bounds how often any row may repeat in one
        epoch; the shortfall is left unfilled rather than redistributed, because
        redistributing it would push some other language past its own cap.
        """
        buckets = {}
        for row in self._keep:
            buckets.setdefault(self.language[row], []).append(int(row))
        counts = {lang: len(rows) for lang, rows in buckets.items()}
        if self.language_temperature >= 1.0 and not self._capped(counts):
            return self._keep.copy()

        weights = {lang: n ** self.language_temperature for lang, n in counts.items()}
        total_weight = sum(weights.values())
        target_total = sum(counts.values())
        rng = np.random.default_rng([self.seed, 0x5A17, self.epoch])
        order, self.capped_languages = [], {}
        for lang, rows in buckets.items():
            want = int(round(target_total * weights[lang] / total_weight))
            ceiling = len(rows) * self.max_epochs_per_language
            if want > ceiling:
                self.capped_languages[lang] = {"wanted": want, "capped_to": ceiling,
                                               "rows": len(rows)}
                want = ceiling
            if want <= 0:
                continue
            order.append(rng.choice(rows, size=want,
                                    replace=want > len(rows)).astype(np.int64))
        return np.concatenate(order) if order else self._keep.copy()

    def _capped(self, counts):
        """True when any language would exceed its repeat ceiling."""
        if not counts or not self.max_epochs_per_language:
            return False
        target_total = sum(counts.values())
        weights = {lang: n ** self.language_temperature for lang, n in counts.items()}
        total_weight = sum(weights.values())
        return any(target_total * weights[lang] / total_weight
                   > n * self.max_epochs_per_language for lang, n in counts.items())

    # ── access ────────────────────────────────────────────────────────────
    def __len__(self):
        return len(self._order)

    def _text(self, row):
        return self.text_flat[self.text_off[row]:self.text_off[row + 1]].tolist()

    def _audio(self, row):
        return self.audio_flat[self.audio_off[row]:self.audio_off[row + 1]].tolist()

    def _pick_reference(self, row, rng):
        """A different utterance by the same speaker, or None.

        None when the speaker has only this one utterance -- a single-utterance
        speaker cannot supply a reference that is not the target, and using the
        target as its own reference teaches copying.
        """
        candidates = self._by_speaker.get(self.speaker_id[row])
        if candidates is None or len(candidates) < 2:
            return None
        choice = int(rng.integers(0, len(candidates) - 1))
        # Skip over the target itself without resampling, so the draw is uniform
        # over the other utterances rather than rejection-sampled.
        pick = int(candidates[choice])
        return int(candidates[-1]) if pick == row else pick

    def __getitem__(self, index):
        row = int(self._order[index])
        # Keyed on the row rather than on `index`: with oversampling the same
        # row appears several times in one epoch, and keying on the position
        # would give every copy a different reference, which is what we want --
        # so the position is mixed in too.
        rng = np.random.default_rng([self.seed, self.epoch, row, index])

        reference = self._pick_reference(row, rng)
        use_reference = reference is not None and rng.random() < self.reference_probability

        reference_text = reference_audio = None
        if use_reference:
            reference_text = self._text(reference)
            reference_audio = self._audio(reference)
            if self.reference_max_tokens:
                # Truncate rather than drop the example. The reference is
                # conditioning, so a shorter one is a weaker signal; dropping the
                # row would throw away a target that is perfectly trainable, and
                # it is always the long references that would go.
                reference_audio = reference_audio[:self.reference_max_tokens]

        audio = self._audio(row)
        input_ids, labels = build_example(
            self.token_map, self._text(row), audio,
            reference_text_ids=reference_text,
            reference_audio_tokens=reference_audio)

        if self.max_tokens and len(input_ids) > self.max_tokens:
            # Second line of defence, after the length filter. Shedding the
            # reference keeps the target intact; the alternative is an example
            # that silently blows the step's memory budget.
            input_ids, labels = build_example(self.token_map, self._text(row), audio)

        # The embedding comes from the *reference* clip, matching inference,
        # where it is computed from whatever clip the caller cloned. Falling back
        # to the target's own embedding when there is no reference would leak the
        # target into its own conditioning.
        source = reference if reference is not None else row
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "speaker_embedding": torch.from_numpy(
                self.embeddings[source].astype(np.float32)),
            "speaker": self.speaker_id[row],
            "language": self.language[row],
            "utt_id": self.utt_id[row],
        }

    def describe(self):
        """Counts a training log should record before the first step."""
        from collections import Counter

        kept = Counter(self.language[r] for r in self._keep)
        sampled = Counter(self.language[int(r)] for r in self._order)
        hours = {lang: round(float(sum(self.duration_s[r] for r in self._keep
                                       if self.language[r] == lang)) / 3600, 1)
                 for lang in kept}
        return {
            "rows_total": self.n_rows,
            "rows_kept": int(len(self._keep)),
            "rows_per_epoch": int(len(self._order)),
            "speakers": len(self._by_speaker),
            "single_utterance_speakers": sum(
                1 for v in self._by_speaker.values() if len(v) < 2),
            "hours_by_language": hours,
            "kept_by_language": dict(kept),
            "sampled_by_language": dict(sampled),
            "language_temperature": self.language_temperature,
            "max_epochs_per_language": self.max_epochs_per_language,
            # Non-empty means a language wanted more repeats than the cap allows:
            # the mix is flatter than requested, and that is worth seeing in a
            # training log rather than discovering from a loss curve.
            "capped_languages": getattr(self, "capped_languages", {}),
            "epoch": self.epoch,
        }


def _resolve_shards(shards):
    """Accept a directory, a glob, a single file, or a list of any of those."""
    import glob
    import pathlib

    if isinstance(shards, (str, pathlib.Path)):
        shards = [shards]
    out = []
    for entry in shards:
        path = pathlib.Path(entry)
        if path.is_dir():
            out += sorted(path.glob("*.parquet"))
        elif any(ch in str(entry) for ch in "*?["):
            out += sorted(pathlib.Path(p) for p in glob.glob(str(entry)))
        else:
            out.append(path)
    if not out:
        raise FileNotFoundError(f"no parquet shards matched {shards!r}")
    missing = [p for p in out if not p.exists()]
    if missing:
        raise FileNotFoundError(f"shards do not exist: {missing}")
    return out


class Collator:
    """Pad a batch to a rectangle without disturbing the mask or position 0.

    **Right padding, and it is not interchangeable.** `speaker.py` writes the
    projected speaker vector over sequence position 0 unconditionally. Left
    padding would put a pad token at position 0, so the speaker would overwrite
    padding while `[START]` sat at some per-row offset -- a batch where speaker
    conditioning is attached to nothing, with no error raised. Right padding
    keeps `[START]` at index 0 for every row, which is also where inference puts
    it (`build_generation_prompt` emits `start_id` first in both branches).

    Training is teacher-forced, so right padding costs nothing: the pads sit
    after the last real token, causal attention never looks forward at them, and
    their labels are -100. That is the opposite of batched *generation*, which
    needs left padding to keep every row's last token at the same index -- the
    inference path sidesteps the question by running one sequence at a time.
    """

    def __init__(self, pad_id=0):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(item["input_ids"]) for item in batch)
        input_ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), IGNORE_INDEX, dtype=torch.long)
        attention_mask = torch.zeros((len(batch), width), dtype=torch.long)
        for row, item in enumerate(batch):
            length = len(item["input_ids"])
            input_ids[row, :length] = item["input_ids"]
            labels[row, :length] = item["labels"]
            attention_mask[row, :length] = 1
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "speaker_embedding": torch.stack([item["speaker_embedding"] for item in batch]),
        }


def supervised_span(input_ids, token_map):
    """Where the loss *should* apply, found by reading the token stream.

    The second, independent derivation of the mask: `build_example` gets it from
    the prompt's length while this one locates `<speech_start>` and
    `<speech_end>` in the finished sequence. The self-check compares them, which
    only means something because neither is computed from the other.

    Returns (first, last) inclusive label indices, or None when the row has no
    target segment. `<speech_start>` is taken as the last occurrence: a
    checkpoint without reference-role tokens wraps its reference in the plain
    speech markers too (`prompt.build_reference_prefix`), so the first one
    belongs to the reference.
    """
    ids = input_ids.tolist() if torch.is_tensor(input_ids) else list(input_ids)
    starts = [i for i, t in enumerate(ids) if t == token_map.speech_start_id]
    ends = [i for i, t in enumerate(ids) if t == token_map.speech_end_id]
    if not starts or not ends:
        return None
    start = starts[-1]
    end = next((i for i in ends if i > start), None)
    return None if end is None else (start + 1, end)
