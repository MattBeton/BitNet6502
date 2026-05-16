"""Word-level tokenisation for TinyStories.

Differs from `modelling/tinystories.py` in three ways:
  1. Vocab is a list of *words* (not characters). Token IDs map 1:1 to words.
  2. All "name" words (lily, tom, ben, ...) collapse to a single `<NAME>` token,
     rendered as `lily` when generating — recovers ~3-20 vocab slots that
     would otherwise be eaten by character names.
  3. Sentences are filtered to those whose words (after name collapse) all
     lie in the chosen vocab.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from modelling.tinystories import (
        DEFAULT_TRAIN_PATH, DEFAULT_VALID_PATH, DEFAULT_CACHE_DIR,
        TokenStreamDataset, normalize_text, _TERMINATOR_RE,
    )
    from modelling.shakespeare import Vocabulary
except ModuleNotFoundError:  # pragma: no cover
    from tinystories import (  # type: ignore
        DEFAULT_TRAIN_PATH, DEFAULT_VALID_PATH, DEFAULT_CACHE_DIR,
        TokenStreamDataset, normalize_text, _TERMINATOR_RE,
    )
    from shakespeare import Vocabulary

WORD_NORMALIZE_VERSION = 1
DELIMITER = "<|endoftext|>"

# Names that should collapse to a single token. Identified by scanning the
# top-500 TinyStories filter vocab. Comma-separating is intentional so the
# linter doesn't word-wrap.
NAMES: frozenset[str] = frozenset({
    "lily", "tom", "ben", "timmy", "tim", "sam", "anna", "max", "jack",
    "mia", "lucy", "john", "bob", "sue", "sarah", "jane", "joe", "daisy",
    "teddy", "benny", "billy",
})

NAME_TOKEN = "<NAME>"        # symbolic representation in vocab
NAME_DISPLAY = "lily"        # what gets written out on token emit


def build_word_vocab(
    top_words_path: str | Path,
    vocab_size: int,
    names: Iterable[str] = NAMES,
) -> Vocabulary:
    """Construct a word vocabulary of size `vocab_size`.

    Reads the top-frequency word list, drops any words in `names`, takes the
    top (vocab_size - 1) remaining, and appends NAME_TOKEN at position
    vocab_size - 1. The resulting vocab has exactly `vocab_size` entries.
    """
    name_set = set(names)
    all_words = [w.strip() for w in Path(top_words_path).read_text().splitlines() if w.strip()]
    non_name = [w for w in all_words if w not in name_set]
    if vocab_size - 1 > len(non_name):
        raise ValueError(
            f"asked for vocab_size={vocab_size} but only {len(non_name)} non-name words available"
        )
    itos = non_name[: vocab_size - 1] + [NAME_TOKEN]
    stoi = {w: i for i, w in enumerate(itos)}
    return Vocabulary(stoi=stoi, itos=itos)


def _collapse_names(words: Sequence[str], name_set: set[str]) -> list[str]:
    return [NAME_TOKEN if w in name_set else w for w in words]


# ---------------------------------------------------------------------------
# Cache & build
# ---------------------------------------------------------------------------


def _signature(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _cache_key(source_path: Path, vocab_path: Path, vocab: Vocabulary) -> dict:
    h = hashlib.sha256(("\n".join(vocab.itos)).encode("utf-8")).hexdigest()
    return {
        "source": _signature(source_path),
        "top_words": _signature(vocab_path),
        "vocab_size": vocab.size,
        "vocab_sha256": h,
        "name_token": NAME_TOKEN,
        "names_sha256": hashlib.sha256(
            ("\n".join(sorted(NAMES))).encode("utf-8")
        ).hexdigest(),
        "normalize_version": WORD_NORMALIZE_VERSION,
    }


def filter_and_word_tokenize_stream(
    source_path: Path,
    vocab: Vocabulary,
    progress_every: int = 500_000,
) -> np.ndarray:
    """Walk `source_path` line by line. Split into sentences, normalise each
    sentence to lowercase, collapse names, drop sentences with any word not in
    the vocab, tokenise the rest to word IDs and concatenate."""
    vocab_set = set(vocab.itos)
    name_set = set(NAMES)
    stoi = vocab.stoi

    out: list[int] = []
    kept = 0
    total = 0
    t0 = time.time()
    with source_path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            if raw.strip() == DELIMITER:
                continue
            for piece in _TERMINATOR_RE.split(raw):
                text = normalize_text(piece)
                if not text:
                    continue
                total += 1
                words = _collapse_names(text.split(), name_set)
                if all(w in vocab_set for w in words):
                    out.extend(stoi[w] for w in words)
                    kept += 1
            if progress_every and i % progress_every == 0:
                elapsed = time.time() - t0
                print(
                    f"  {source_path.name}: {i:>10,} lines | "
                    f"{kept:>10,}/{total:>10,} kept ({100 * kept / max(total, 1):5.2f}%) | "
                    f"{elapsed:5.1f}s",
                    flush=True,
                )

    arr = np.asarray(out, dtype=np.int16)
    print(
        f"  {source_path.name}: kept {kept:,}/{total:,} sentences "
        f"({100 * kept / max(total, 1):.2f}%); {len(arr):,} tokens; "
        f"{time.time() - t0:.1f}s",
        flush=True,
    )
    return arr


def filtered_word_tokens_with_cache(
    source_path: str | Path,
    top_words_path: str | Path,
    vocab: Vocabulary,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    rebuild: bool = False,
) -> np.ndarray:
    """Load filtered+word-tokenised stream from cache, or build it and cache."""
    source_path = Path(source_path)
    top_words_path = Path(top_words_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{source_path.stem}__V{vocab.size}__words"
    tokens_path = cache_dir / f"{stem}.tokens.npy"
    meta_path = cache_dir / f"{stem}.meta.json"

    key = _cache_key(source_path, top_words_path, vocab)

    if not rebuild and tokens_path.exists() and meta_path.exists():
        try:
            cached = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = None
        if cached == key:
            print(f"cache hit:  {tokens_path.name}")
            return np.load(tokens_path)
        print(f"cache stale: {tokens_path.name} (rebuilding)")

    print(f"building cache: {tokens_path.name}")
    tokens = filter_and_word_tokenize_stream(source_path, vocab)
    np.save(tokens_path, tokens)
    meta_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(
        f"wrote cache: {tokens_path.name} "
        f"({tokens.nbytes / 1024**2:.1f} MB, {len(tokens):,} tokens)"
    )
    return tokens


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_word_datasets(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    valid_path: str | Path = DEFAULT_VALID_PATH,
    block_size: int = 64,
    *,
    top_words_path: str | Path,
    vocab_size: int = 64,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    rebuild_cache: bool = False,
) -> tuple[Dataset, Dataset, Vocabulary]:
    vocab = build_word_vocab(top_words_path, vocab_size)
    train_tokens = filtered_word_tokens_with_cache(
        train_path, top_words_path, vocab, cache_dir, rebuild=rebuild_cache
    )
    valid_tokens = filtered_word_tokens_with_cache(
        valid_path, top_words_path, vocab, cache_dir, rebuild=rebuild_cache
    )
    train_ds = TokenStreamDataset(train_tokens, block_size=block_size)
    valid_ds = TokenStreamDataset(valid_tokens, block_size=block_size)
    return train_ds, valid_ds, vocab


if __name__ == "__main__":
    # Smoke test on the validation file (small).
    vocab_file = DEFAULT_CACHE_DIR.parent / "tinystories_vocab_top500.txt"
    train_ds, valid_ds, vocab = build_word_datasets(
        train_path=DEFAULT_VALID_PATH,
        valid_path=DEFAULT_VALID_PATH,
        top_words_path=vocab_file,
        vocab_size=64,
    )
    print(f"vocab size:   {vocab.size}")
    print(f"vocab head:   {vocab.itos[:10]}")
    print(f"vocab tail:   {vocab.itos[-5:]}")
    print(f"train tokens: {len(train_ds):,}")
    x, y = train_ds[0]
    print(f"first window IDs: {x.tolist()}")
    print(f"first window words: {[vocab.itos[i] for i in x.tolist()]}")
