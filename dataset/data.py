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

# Character names appearing in the top-500 TinyStories vocab. When the
# `dedup_names` flag is on, every occurrence of one of these is rewritten to
# a small set of canonical names *before* the vocab-filter check — so stories
# using rare names like "billy" no longer get dropped, and the model only
# ever sees a handful of character names.
#
# Two canonical names — one female ("lily"), one male ("tom") — preserve
# pronoun cues ("she said", "he ran") and avoid the "lily and lily are
# friends" artifact we saw with a single canonical name.
FEMALE_NAMES: frozenset[str] = frozenset({
    "lily", "anna", "mia", "lucy", "sarah", "sara", "jane", "daisy", "sue",
})
MALE_NAMES: frozenset[str] = frozenset({
    "tom", "ben", "timmy", "tim", "sam", "max", "jack", "john", "bob",
    "joe", "teddy", "benny", "billy",
})
FEMALE_CANONICAL = "lily"
MALE_CANONICAL = "tom"
# Union; kept for code that wants to know "is this token a known name?"
DEDUP_NAMES: frozenset[str] = FEMALE_NAMES | MALE_NAMES


def _apply_name_dedup(text: str) -> str:
    """Replace each whole-word match of a known name with its gendered canonical."""
    out: list[str] = []
    for w in text.split():
        if w in FEMALE_NAMES:
            out.append(FEMALE_CANONICAL)
        elif w in MALE_NAMES:
            out.append(MALE_CANONICAL)
        else:
            out.append(w)
    return " ".join(out)


# Boilerplate TinyStories template that creates massive repetition in the
# training stream: nearly every story starts with "once upon a time" and ends
# with "the end". With many stories concatenated into 64-char training windows,
# the model overfits to these phrases. Strip them sentence-by-sentence.
#
# Also strips the character-introduction template that dominates the dedup'd
# corpus ("there was a little girl named lily" — ~17k occurrences in just 5 MB
# of training data). Without this, the model finetunes to spam the intro
# template every few tokens. Names elsewhere in the story stay, so the model
# still learns who the characters are.
_BOILERPLATE_PREFIX_RE = re.compile(
    r"^("
    r"once upon a time|"
    r"one day|"
    r"there (?:was|were) (?:a |an |some )?(?:little |small |big |young )?"
    r"(?:girl|boy|child|kid|man|woman)(?: named (?:lily|tom))?"
    r")\s+"
)
_BOILERPLATE_SUFFIX_RE = re.compile(r"\s+the end$")


def _strip_boilerplate(sentence: str) -> str:
    """Remove TinyStories template wrappers from a normalised sentence.

    Stripped patterns (all leave the sentence body intact):
        leading  'once upon a time '
        leading  'one day '
        leading  'there was a (little)? (girl|boy|...) (named lily|tom)?'
        trailing ' the end'
        whole-sentence 'the end' → empty
    """
    if sentence == "the end":
        return ""
    # Strip multiple stacked prefixes ("once upon a time there was a girl named lily ...").
    while True:
        new = _BOILERPLATE_PREFIX_RE.sub("", sentence)
        if new == sentence:
            break
        sentence = new
    sentence = _BOILERPLATE_SUFFIX_RE.sub("", sentence)
    return sentence


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


def _cache_key(source_path: Path, vocab_path: Path, vocab_words: Sequence[str],
               *, dedup_names: bool = False, strip_boilerplate: bool = False) -> dict:
    digest = hashlib.sha256("\n".join(vocab_words).encode("utf-8")).hexdigest()
    return {
        "source": _file_signature(source_path),
        "vocab": _file_signature(vocab_path),
        "vocab_size": len(vocab_words),
        "vocab_sha256": digest,
        "normalize_version": NORMALIZE_VERSION,
        "alphabet": CHAR_ALPHABET,
        "dedup_names": dedup_names,
        # Versioned digest of the dedup spec; v2 = gendered (female→lily, male→tom)
        "dedup_names_sha256": hashlib.sha256(
            ("v2|" + "|".join(sorted(FEMALE_NAMES)) + "||" + "|".join(sorted(MALE_NAMES))).encode("utf-8")
        ).hexdigest() if dedup_names else None,
        # Versioned: v2 adds 'there was a (girl|boy) named (lily|tom)' to the strip set.
        "strip_boilerplate": strip_boilerplate,
        "strip_boilerplate_version": 2 if strip_boilerplate else 0,
    }


def filter_and_tokenize_stream(
    source_path: Path,
    vocab_words: Sequence[str],
    progress_every: int = 500_000,
    *,
    dedup_names: bool = False,
    strip_boilerplate: bool = False,
) -> np.ndarray:
    """Stream `source_path`, keep sentences whose words are all in `vocab_words`,
    join them with single spaces, and return an int8 array of char-token ids.

    `dedup_names=True`: rewrite known character names to a small set of
    canonical names (female→'lily', male→'tom') so the model doesn't waste
    capacity disambiguating many characters.

    `strip_boilerplate=True`: drop TinyStories templating ('once upon a time',
    'one day', 'the end') sentence-by-sentence. These template phrases otherwise
    dominate the training distribution and the model overfits to emitting them."""
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
                if dedup_names:
                    text = _apply_name_dedup(text)
                if strip_boilerplate:
                    text = _strip_boilerplate(text)
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
    dedup_names: bool = False,
    strip_boilerplate: bool = False,
) -> np.ndarray:
    """Load filtered+tokenised stream from cache, or build it and cache.

    Cache invalidates automatically when the source file, vocab file,
    `NORMALIZE_VERSION`, dedup spec, or boilerplate-strip flag changes.
    """
    source_path = Path(source_path)
    vocab_path = Path(vocab_path)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    suffix = ""
    if dedup_names:
        suffix += "__dedup"
    if strip_boilerplate:
        suffix += "__stripped"
    stem = f"{source_path.stem}__{vocab_path.stem}{suffix}"
    tokens_path = cache_dir / f"{stem}.tokens.npy"
    meta_path = cache_dir / f"{stem}.meta.json"

    vocab_words = load_word_vocab(vocab_path)
    key = _cache_key(source_path, vocab_path, vocab_words,
                     dedup_names=dedup_names, strip_boilerplate=strip_boilerplate)

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
    tokens = filter_and_tokenize_stream(source_path, vocab_words,
                                        dedup_names=dedup_names,
                                        strip_boilerplate=strip_boilerplate)
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
    dedup_names: bool = False,
    strip_boilerplate: bool = False,
) -> tuple[Dataset, Dataset, Vocabulary]:
    """Build (train, valid, vocab) datasets ready to feed into a DataLoader."""
    train_tokens = filtered_tokens_with_cache(
        train_path, vocab_path, cache_dir, rebuild=rebuild_cache,
        dedup_names=dedup_names, strip_boilerplate=strip_boilerplate,
    )
    valid_tokens = filtered_tokens_with_cache(
        valid_path, vocab_path, cache_dir, rebuild=rebuild_cache,
        dedup_names=dedup_names, strip_boilerplate=strip_boilerplate,
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
