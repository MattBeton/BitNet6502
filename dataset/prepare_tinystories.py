"""Preprocess TinyStories into one-story-per-line files.

Reads TinyStories-train.txt / TinyStories-valid.txt, where stories span
multiple paragraphs (newline-separated) and are delimited by `<|endoftext|>`
lines. Writes -oneline.txt siblings where each line is one full story
(internal paragraph breaks replaced with spaces).

Vocabulary normalization (lowercasing, stripping punctuation, etc.) is NOT
done here — that is the dataloader's job at runtime so we don't have to
re-preprocess when the vocab spec changes.

The train file is ~1.9 GB so this script streams line by line and never
holds the full corpus in memory.

Usage:
    python modelling/data/prepare_tinystories.py
"""
from __future__ import annotations

import sys
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
DELIMITER = "<|endoftext|>"

PAIRS = [
    ("TinyStories-train.txt", "TinyStories-train-oneline.txt"),
    ("TinyStories-valid.txt", "TinyStories-valid-oneline.txt"),
]


def collapse(in_path: Path, out_path: Path) -> tuple[int, int]:
    stories_written = 0
    skipped_empty = 0
    buffer: list[str] = []

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.rstrip("\n")
            if line.strip() == DELIMITER:
                story = " ".join(s for s in (p.strip() for p in buffer) if s)
                buffer.clear()
                if story:
                    fout.write(story)
                    fout.write("\n")
                    stories_written += 1
                else:
                    skipped_empty += 1
            else:
                buffer.append(line)

        # Trailing story (file may not end with a delimiter).
        story = " ".join(s for s in (p.strip() for p in buffer) if s)
        if story:
            fout.write(story)
            fout.write("\n")
            stories_written += 1

    return stories_written, skipped_empty


def main() -> int:
    for src_name, dst_name in PAIRS:
        src = DATA_DIR / src_name
        dst = DATA_DIR / dst_name
        if not src.exists():
            print(f"skip: {src} not found", file=sys.stderr)
            continue
        print(f"{src_name} -> {dst_name} ...", flush=True)
        n, skipped = collapse(src, dst)
        size_mb = dst.stat().st_size / (1024 * 1024)
        print(f"  wrote {n:,} stories, skipped {skipped} empty, {size_mb:.1f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
