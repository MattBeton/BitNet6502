from __future__ import annotations

import re
import string
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


# DEFAULT_DATA_PATH = Path(__file__).with_name("data") / "shakespeare.txt"
DEFAULT_DATA_PATH = Path(__file__).with_name("data") / "shakespeare_speech_romeo_juliet.txt"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("\n", " ")
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class Vocabulary:
    stoi: dict[str, int]
    itos: list[str]

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        chars = sorted(set(text))
        return cls(
            stoi={ch: idx for idx, ch in enumerate(chars)},
            itos=chars,
        )

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(self.itos[idx] for idx in token_ids)

    @property
    def size(self) -> int:
        return len(self.itos)


class ShakespeareCharDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, token_ids: list[int], block_size: int) -> None:
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
        return (
            torch.tensor(x, dtype=torch.long),
            torch.tensor(y, dtype=torch.long),
        )


def load_shakespeare_text(path: str | Path = DEFAULT_DATA_PATH) -> str:
    return normalize_text(Path(path).read_text(encoding="utf-8"))


def split_token_ids(token_ids: list[int], train_fraction: float = 0.9) -> tuple[list[int], list[int]]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    split_idx = int(len(token_ids) * train_fraction)
    train_ids = token_ids[:split_idx]
    test_ids = token_ids[split_idx:]
    return train_ids, test_ids


def build_datasets(
    path: str | Path = DEFAULT_DATA_PATH,
    block_size: int = 128,
    train_fraction: float = 0.9,
) -> tuple[ShakespeareCharDataset, ShakespeareCharDataset, Vocabulary]:
    text = load_shakespeare_text(path)
    vocab = Vocabulary.from_text(text)
    token_ids = vocab.encode(text)
    train_ids, test_ids = split_token_ids(token_ids, train_fraction=train_fraction)

    train_dataset = ShakespeareCharDataset(train_ids, block_size=block_size)
    test_dataset = ShakespeareCharDataset(test_ids, block_size=block_size)
    return train_dataset, test_dataset, vocab


if __name__ == "__main__":
    train_dataset, test_dataset, vocab = build_datasets()
    sample_x, sample_y = train_dataset[0]

    print(f"vocab size: {vocab.size}")
    print(f"train samples: {len(train_dataset)}")
    print(f"test samples: {len(test_dataset)}")
    print(f"sample x shape: {tuple(sample_x.shape)}")
    print(f"sample y shape: {tuple(sample_y.shape)}")
