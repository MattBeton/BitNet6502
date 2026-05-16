"""TinyStories data pipeline.

  text file ─► normalise (lowercase, strip non-alpha) ─► filter by word vocab
            ─► char-tokenise with `CHAR_VOCAB` ─► np.int8 buffer ─► cache on disk
            ─► sliding `block_size` windows  ─► (x, y) batches

`Vocabulary` is the small dataclass our checkpoints reference; kept here in the
same module as the dataset so there's one obvious import path.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


# ----------------------------------------------------------------------------- #
# Vocabulary
# ----------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Vocabulary:
    """Char → token-id mapping. Both directions exposed so encode/decode are O(1)."""
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        chars = sorted(set(text))
        return cls(stoi={ch: idx for idx, ch in enumerate(chars)}, itos=chars)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in token_ids)

    @property
    def size(self) -> int:
        return len(self.itos)


# 27-char canonical vocab: space + a-z. Hardcoded so the model's vocab_size
# doesn't depend on whatever corpus slice happens to be in front of us.
CHAR_ALPHABET = " abcdefghijklmnopqrstuvwxyz"
CHAR_VOCAB = Vocabulary(
    stoi={c: i for i, c in enumerate(CHAR_ALPHABET)},
    itos=list(CHAR_ALPHABET),
)


# ----------------------------------------------------------------------------- #
# Paths & cache version
# ----------------------------------------------------------------------------- #

DEFAULT_TRAIN_PATH = Path(__file__).with_name("data") / "TinyStories-train-oneline.txt"
DEFAULT_VALID_PATH = Path(__file__).with_name("data") / "TinyStories-valid-oneline.txt"
DEFAULT_CACHE_DIR = Path(__file__).with_name("data") / "cache"
DEFAULT_VOCAB_PATH = Path(__file__).with_name("data") / "tinystories_vocab_top500.txt"

# Bump whenever `normalize_text` or `_TERMINATOR_RE` changes — invalidates caches.
NORMALIZE_VERSION = 2

DELIMITER = "<|endoftext|>"
_KEEP_RE = re.compile(r"[^a-z ]")
_WS_RE = re.compile(r"\s+")
_TERMINATOR_RE = re.compile(r"[.!?]+")


# ----------------------------------------------------------------------------- #
# Text normalisation
# ----------------------------------------------------------------------------- #


def normalize_text(text: str) -> str:
    """Project to the 27-char vocab.

    NFKD-normalise to decompose diacritics, drop non-ASCII residue, lowercase,
    drop apostrophes so contractions fuse, replace any other non-letter with a
    space, collapse whitespace runs.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("'", "")
    text = _KEEP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ----------------------------------------------------------------------------- #
# Sliding-window dataset over a 1-D token id buffer
# ----------------------------------------------------------------------------- #


class TokenStreamDataset(Dataset):
    """Sliding `block_size` windows over a 1-D token id buffer.

    Backing store should be int8 numpy or a torch tensor — Python lists are
    ~28 bytes/element vs 1 byte for int8 numpy.
    """

    def __init__(self, token_ids, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        if len(token_ids) <= block_size:
            raise ValueError("token sequence must be longer than block_size")
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


# ----------------------------------------------------------------------------- #
# Word-vocab filter + char tokenisation, cached on disk
# ----------------------------------------------------------------------------- #


def load_word_vocab(vocab_path: str | Path) -> list[str]:
    return [w.strip() for w in Path(vocab_path).read_text(encoding="utf-8").splitlines() if w.strip()]


def _file_signature(path: Path) -> dict:
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _cache_key(source_path: Path, vocab_path: Path, vocab_words: Sequence[str]) -> dict:
    digest = hashlib.sha256("\n".join(vocab_words).encode("utf-8")).hexdigest()
    return {
        "source": _file_signature(source_path),
        "vocab": _file_signature(vocab_path),
        "vocab_size": len(vocab_words),
        "vocab_sha256": digest,
        "normalize_version": NORMALIZE_VERSION,
        "alphabet": CHAR_ALPHABET,
    }


def filter_and_tokenize_stream(
    source_path: Path,
    vocab_words: Sequence[str],
    progress_every: int = 500_000,
) -> np.ndarray:
    """Stream `source_path`, keep sentences whose words are all in `vocab_words`,
    join them with single spaces, and return an int8 array of char-token ids."""
    word_set = set(vocab_words)
    stoi = CHAR_VOCAB.stoi
    pieces: list[str] = []
    n_kept = 0
    n_total = 0
    t0 = time.time()
    with source_path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, 1):
            if raw.strip() == DELIMITER:
                continue
            for piece in _TERMINATOR_RE.split(raw):
                text = normalize_text(piece)
                if not text:
                    continue
                n_total += 1
                if all(w in word_set for w in text.split()):
                    pieces.append(text)
                    n_kept += 1
            if progress_every and i % progress_every == 0:
                elapsed = time.time() - t0
                rate = n_kept / max(n_total, 1)
                print(
                    f"  {source_path.name}: {i:>10,} lines | "
                    f"{n_kept:>10,}/{n_total:>10,} sentences kept ({100*rate:5.2f}%) | "
                    f"{elapsed:5.1f}s",
                    flush=True,
                )

    full_text = " ".join(pieces)
    elapsed = time.time() - t0
    print(
        f"  {source_path.name}: kept {n_kept:,}/{n_total:,} sentences "
        f"({100*n_kept/max(n_total,1):.2f}%); {len(full_text):,} chars; {elapsed:.1f}s",
        flush=True,
    )

    # Vectorised lookup (much faster than `[stoi[c] for c in text]`).
    arr = np.frombuffer(full_text.encode("ascii"), dtype=np.uint8)
    table = np.zeros(256, dtype=np.int8)
    for ch, i in stoi.items():
        table[ord(ch)] = i
    return table[arr].astype(np.int8)


def filtered_tokens_with_cache(
    source_path: str | Path,
    vocab_path: str | Path,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    rebuild: bool = False,
) -> np.ndarray:
    """Load filtered+tokenised stream from cache, or build it and cache.

    Cache invalidates automatically when the source file, the vocab file, or
    `NORMALIZE_VERSION` changes.
    """
    source_path = Path(source_path)
    vocab_path = Path(vocab_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stem = f"{source_path.stem}__{vocab_path.stem}"
    tokens_path = cache_dir / f"{stem}.tokens.npy"
    meta_path = cache_dir / f"{stem}.meta.json"

    vocab_words = load_word_vocab(vocab_path)
    key = _cache_key(source_path, vocab_path, vocab_words)

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
    tokens = filter_and_tokenize_stream(source_path, vocab_words)
    np.save(tokens_path, tokens)
    meta_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(
        f"wrote cache: {tokens_path.name} "
        f"({tokens.nbytes / 1024**2:.1f} MB, {len(tokens):,} tokens)"
    )
    return tokens


# ----------------------------------------------------------------------------- #
# Top-level entry point used by training
# ----------------------------------------------------------------------------- #


def build_datasets(
    block_size: int = 64,
    *,
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    valid_path: str | Path = DEFAULT_VALID_PATH,
    vocab_path: str | Path = DEFAULT_VOCAB_PATH,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    rebuild_cache: bool = False,
) -> tuple[Dataset, Dataset, Vocabulary]:
    """Build (train, valid, vocab) datasets ready to feed into a DataLoader."""
    train_tokens = filtered_tokens_with_cache(
        train_path, vocab_path, cache_dir, rebuild=rebuild_cache
    )
    valid_tokens = filtered_tokens_with_cache(
        valid_path, vocab_path, cache_dir, rebuild=rebuild_cache
    )
    train_ds = TokenStreamDataset(train_tokens, block_size=block_size)
    valid_ds = TokenStreamDataset(valid_tokens, block_size=block_size)
    return train_ds, valid_ds, CHAR_VOCAB


if __name__ == "__main__":
    # Smoke test against the validation file with the default top-500 word filter.
    train_ds, valid_ds, vocab = build_datasets(
        block_size=64,
        train_path=DEFAULT_VALID_PATH,  # use small valid file as 'train' for smoke
        valid_path=DEFAULT_VALID_PATH,
    )
    x, y = train_ds[0]
    print(f"vocab size:    {vocab.size}")
    print(f"vocab chars:   {vocab.itos}")
    print(f"train samples: {len(train_ds):,}")
    print(f"first window:  {vocab.decode(x.tolist())!r}")
