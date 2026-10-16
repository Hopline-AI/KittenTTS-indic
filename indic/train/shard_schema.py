"""The on-disk contract between the preprocessing job and the training reader.

One file so there is exactly one definition. The reader and the writer live in
different modules with very different dependencies -- the writer needs torch,
librosa and a GPU, the reader needs neither -- and the failure mode when they
drift is not an exception. It is a model that trains on misaligned columns and
produces noise, discovered after the allocation is spent.

**One row per utterance, not per training example.** The obvious schema stores a
target and its reference side by side, which is how `build_example` consumes
them. It is the wrong thing to write down: it duplicates every clip's codec
tokens (each utterance serves as a reference for its speaker's other
utterances), and it freezes the pairing at preprocessing time, so the model sees
the same reference for the same target in every epoch. Pairing at read time
costs nothing, halves the file, and lets the reference vary per epoch, which is
free regularisation.

**Codec ids are stored 0-based**, exactly as `S3Codec.encode_reference` returns
them. The `audio_id_base` offset belongs to `build_example`, which is the only
place that knows the token map. Storing pre-offset ids would silently bind every
shard to one checkpoint's vocabulary layout -- and the whole point of this
project is that the layout changes.

**Text ids are tokenizer-specific.** A shard written for Qwen3 cannot train a
Sarvam model. `tokenizer` in the file metadata records which one, and the reader
refuses a mismatch rather than training on ids that mean something else.
"""

import json

# Bumped when a column changes meaning. The reader refuses a shard whose version
# it does not know, because the alternative is reading a renamed column as if it
# were the old one.
SCHEMA_VERSION = 1

COLUMNS = {
    "utt_id": "string, unique within the corpus",
    "language": "string, ISO 639-1 as in indic/langs.py",
    "speaker_id": "string, unique per speaker ACROSS languages",
    "text_ids": "list<int32>, add_special_tokens=False, no audio offset",
    "audio_tokens": "list<int32>, S3 codec ids at 25 Hz, 0-based",
    "speaker_embedding": "fixed_size_list<float32>[512], unit norm",
    "duration_s": "float32, source clip duration",
    "cer": "float32, ASR agreement with the transcript; -1 when unmeasured",
    "emotion": "string or null, an expression tag such as 'joyful'",
}

# The codec's frame rate. Stored here because the reader converts a token budget
# to seconds without importing anything from the inference stack.
TOKENS_PER_SECOND = 25
SPEAKER_EMBEDDING_DIM = 512


def arrow_schema(tokenizer, embedding_dim=SPEAKER_EMBEDDING_DIM, extra=None):
    """The pyarrow schema, with the provenance a reader needs in its metadata."""
    import pyarrow as pa

    metadata = {
        "schema_version": str(SCHEMA_VERSION),
        "tokenizer": tokenizer,
        "tokens_per_second": str(TOKENS_PER_SECOND),
        "embedding_dim": str(embedding_dim),
        **{k: json.dumps(v) for k, v in (extra or {}).items()},
    }
    return pa.schema(
        [
            pa.field("utt_id", pa.string()),
            pa.field("language", pa.string()),
            pa.field("speaker_id", pa.string()),
            pa.field("text_ids", pa.list_(pa.int32())),
            pa.field("audio_tokens", pa.list_(pa.int32())),
            pa.field("speaker_embedding", pa.list_(pa.float32(), embedding_dim)),
            pa.field("duration_s", pa.float32()),
            pa.field("cer", pa.float32()),
            pa.field("emotion", pa.string()),
        ],
        metadata={k.encode(): str(v).encode() for k, v in metadata.items()},
    )


def read_metadata(path):
    """Schema metadata of one shard, as plain strings."""
    import pyarrow.parquet as pq

    raw = pq.read_schema(str(path)).metadata or {}
    return {k.decode(): v.decode() for k, v in raw.items()}


def check_compatible(path, tokenizer=None):
    """Raise unless this shard is one the reader understands.

    Checked per file rather than once per directory: shards are written by
    separate array tasks, and a half-regenerated directory holding two
    tokenizers' ids is exactly the state this catches.
    """
    meta = read_metadata(path)
    version = int(meta.get("schema_version", -1))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"{path}: shard schema v{version}, reader expects v{SCHEMA_VERSION}. "
            "Regenerate the shards with indic.train.prepare_shards.")
    if tokenizer and meta.get("tokenizer") != tokenizer:
        raise ValueError(
            f"{path}: text ids were written for tokenizer {meta.get('tokenizer')!r}, "
            f"but this run uses {tokenizer!r}. The ids mean different strings; "
            "training on them would be training on noise.")
    return meta


def write_shard(path, rows, tokenizer, embedding_dim=SPEAKER_EMBEDDING_DIM,
                row_group_size=512, extra=None):
    """Write one parquet shard from an iterable of row dicts.

    `row_group_size` is small on purpose. A reader that wants only part of a
    shard pays for a whole row group, and 512 rows is roughly 1-2 MB here.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = list(rows)
    if not rows:
        raise ValueError(f"{path}: refusing to write an empty shard")
    schema = arrow_schema(tokenizer, embedding_dim, extra)
    table = pa.Table.from_pydict(
        {
            "utt_id": [r["utt_id"] for r in rows],
            "language": [r["language"] for r in rows],
            "speaker_id": [r["speaker_id"] for r in rows],
            "text_ids": [list(map(int, r["text_ids"])) for r in rows],
            "audio_tokens": [list(map(int, r["audio_tokens"])) for r in rows],
            "speaker_embedding": [list(map(float, r["speaker_embedding"])) for r in rows],
            "duration_s": [float(r["duration_s"]) for r in rows],
            "cer": [float(r.get("cer", -1.0)) for r in rows],
            "emotion": [r.get("emotion") for r in rows],
        },
        schema=schema,
    )
    pq.write_table(table, str(path), row_group_size=row_group_size,
                   compression="zstd")
    return len(rows)
