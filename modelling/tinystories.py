from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:  # works whether imported as a script or as `modelling.tinystories`
    from modelling.shakespeare import ShakespeareCharDataset, Vocabulary
except ModuleNotFoundError:  # pragma: no cover
    from shakespeare import ShakespeareCharDataset, Vocabulary  # type: ignore


DEFAULT_TRAIN_PATH = Path(__file__).with_name("data") / "TinyStories-train-oneline.txt"
DEFAULT_VALID_PATH = Path(__file__).with_name("data") / "TinyStories-valid-oneline.txt"
DEFAULT_CACHE_DIR = Path(__file__).with_name("data") / "cache"

# Bump whenever `normalize_text` or `_TERMINATOR_RE` changes — invalidates caches.
NORMALIZE_VERSION = 2

# 27-char canonical vocab. Stable by construction (matches what normalize_text emits),
# so we don't have to derive it from a particular corpus slice.
CHAR_ALPHABET = " abcdefghijklmnopqrstuvwxyz"
CHAR_VOCAB = Vocabulary(
    stoi={c: i for i, c in enumerate(CHAR_ALPHABET)},
    itos=list(CHAR_ALPHABET),
)

DELIMITER = "<|endoftext|>"

_KEEP_RE = re.compile(r"[^a-z ]")
_WS_RE = re.compile(r"\s+")
_TERMINATOR_RE = re.compile(r"[.!?]+")


def normalize_text(text: str) -> str:
    """Project to the 27-char vocab (26 lowercase letters + space).

    Steps: NFKD-normalize so diacritics decompose ("é" → "e" + combining mark);
    drop the combining marks and any non-ascii residue; lowercase; drop ASCII
    apostrophes so contractions fuse ("didn't" → "didnt"); replace any other
    non-letter character with a single space; collapse runs of whitespace.
    """
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("'", "")
    text = _KEEP_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Token-stream dataset (numpy / tensor backed; avoids 12 GB Python lists).
# ---------------------------------------------------------------------------


class TokenStreamDataset(Dataset):
    """Sliding `block_size` windows over a 1-D token id buffer.

    Backing store can be a list, numpy array, or torch tensor. Numpy / tensor
    backings are strongly preferred for large corpora — a Python list of ints
    is ~28 bytes per element, vs 1 byte for an int8 numpy array."""

    def __init__(self, token_ids, block_size: int) -> None:
        if block_size < 1:
            raise ValueError("block_size must be at least 1")
        n = len(token_ids)
        if n <= block_size:
            raise ValueError("token sequence must be longer than block_size")
        self.token_ids = token_ids
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.token_ids) - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.token_ids[idx : idx + self.block_size]
        y = self.token_ids[idx + 1 : idx + self.block_size + 1]
        return torch.as_tensor(x, dtype=torch.long), torch.as_tensor(y, dtype=torch.long)


# ---------------------------------------------------------------------------
# Filtering by word vocab + caching layer
# ---------------------------------------------------------------------------


def load_word_vocab(vocab_path: str | Path) -> list[str]:
    """Load a one-word-per-line vocab file (as written by the analysis notebook)."""
    path = Path(vocab_path)
    return [w.strip() for w in path.read_text(encoding="utf-8").splitlines() if w.strip()]


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
    join them with single spaces, and return an int8 array of char-token ids
    (using `CHAR_VOCAB`)."""
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
        f"({100*n_kept/max(n_total,1):.2f}%); {len(full_text):,} chars; "
        f"{elapsed:.1f}s",
        flush=True,
    )

    # Tokenize. Vectorized lookup is much faster than [stoi[c] for c in text].
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
    """Load filtered+tokenized stream from cache, or build it and write the cache.

    Cache invalidates automatically when the source file, the vocab file, or
    `NORMALIZE_VERSION` changes."""
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
    # Atomic-ish write: tokens first, then meta. If meta is missing on next
    # load we simply rebuild.
    np.save(tokens_path, tokens)
    meta_path.write_text(json.dumps(key, indent=2), encoding="utf-8")
    print(
        f"wrote cache: {tokens_path.name} "
        f"({tokens.nbytes/1024**2:.1f} MB, {len(tokens):,} tokens)"
    )
    return tokens


# ---------------------------------------------------------------------------
# Unfiltered (legacy) loader — used when no vocab_path is given.
# ---------------------------------------------------------------------------


def _read_oneline_file(
    path: Path,
    max_stories: int | None = None,
    max_chars: int | None = None,
) -> str:
    parts: list[str] = []
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f):
            if max_stories is not None and i >= max_stories:
                break
            piece = normalize_text(raw)
            if not piece:
                continue
            parts.append(piece)
            total += len(piece) + 1
            if max_chars is not None and total >= max_chars:
                break
    return " ".join(parts)


def load_tinystories_text(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    valid_path: str | Path = DEFAULT_VALID_PATH,
    max_train_stories: int | None = None,
    max_train_chars: int | None = None,
    max_valid_stories: int | None = None,
    max_valid_chars: int | None = None,
) -> tuple[str, str]:
    train_text = _read_oneline_file(Path(train_path), max_train_stories, max_train_chars)
    valid_text = _read_oneline_file(Path(valid_path), max_valid_stories, max_valid_chars)
    return train_text, valid_text


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def build_datasets(
    train_path: str | Path = DEFAULT_TRAIN_PATH,
    valid_path: str | Path = DEFAULT_VALID_PATH,
    block_size: int = 64,
    *,
    vocab_path: str | Path | None = None,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    rebuild_cache: bool = False,
    # Only honored when vocab_path is None (unfiltered legacy path).
    max_train_stories: int | None = None,
    max_train_chars: int | None = None,
    max_valid_stories: int | None = None,
    max_valid_chars: int | None = None,
) -> tuple[Dataset, Dataset, Vocabulary]:
    """Build train/valid datasets and the 27-char vocab.

    If `vocab_path` is given, sentences are filtered to only those whose words
    are all in that word-vocab file, and the resulting token streams are
    cached on disk under `cache_dir` (auto-invalidated when source / vocab /
    NORMALIZE_VERSION change). Otherwise reads the full one-line files
    unfiltered, optionally capped via `max_*` args."""
    if vocab_path is not None:
        train_tokens = filtered_tokens_with_cache(
            train_path, vocab_path, cache_dir, rebuild=rebuild_cache
        )
        valid_tokens = filtered_tokens_with_cache(
            valid_path, vocab_path, cache_dir, rebuild=rebuild_cache
        )
        train_ds = TokenStreamDataset(train_tokens, block_size=block_size)
        valid_ds = TokenStreamDataset(valid_tokens, block_size=block_size)
        return train_ds, valid_ds, CHAR_VOCAB

    # Legacy unfiltered path — reads everything into memory as Python text /
    # list ids. Fine for the valid file or capped slices; do not use on the
    # full 1.9 GB train file.
    train_text, valid_text = load_tinystories_text(
        train_path=train_path,
        valid_path=valid_path,
        max_train_stories=max_train_stories,
        max_train_chars=max_train_chars,
        max_valid_stories=max_valid_stories,
        max_valid_chars=max_valid_chars,
    )
    train_ids = CHAR_VOCAB.encode(train_text)
    valid_ids = CHAR_VOCAB.encode(valid_text)
    train_ds = ShakespeareCharDataset(train_ids, block_size=block_size)
    valid_ds = ShakespeareCharDataset(valid_ids, block_size=block_size)
    return train_ds, valid_ds, CHAR_VOCAB


if __name__ == "__main__":
    # Smoke test against the (small) valid file with a vocab-filter.
    vocab_file = DEFAULT_CACHE_DIR.parent / "tinystories_vocab_top500.txt"
    if vocab_file.exists():
        train_ds, valid_ds, vocab = build_datasets(
            train_path=DEFAULT_VALID_PATH,  # use small valid file as 'train' for the smoke test
            valid_path=DEFAULT_VALID_PATH,
            vocab_path=vocab_file,
        )
        print(f"vocab file:    {vocab_file.name}")
    else:
        train_ds, valid_ds, vocab = build_datasets(
            train_path=DEFAULT_VALID_PATH,
            valid_path=DEFAULT_VALID_PATH,
            max_train_stories=5_000,
            max_valid_stories=1_000,
        )
        print(f"vocab file:    (none — unfiltered smoke test)")

    x, y = train_ds[0]
    print(f"vocab size:    {vocab.size}")
    print(f"vocab chars:   {vocab.itos}")
    print(f"train samples: {len(train_ds):,}")
    print(f"valid samples: {len(valid_ds):,}")
    print(f"x shape: {tuple(x.shape)}, y shape: {tuple(y.shape)}")
    print(f"first window:  {vocab.decode(x.tolist())!r}")
