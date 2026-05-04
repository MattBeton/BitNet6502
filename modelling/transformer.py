from __future__ import annotations

from dataclasses import dataclass
import math
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from modelling.shakespeare import build_datasets
except ModuleNotFoundError:
    from shakespeare import build_datasets


@dataclass
class Config:
    block_size: int = 64
    batch_size: int = 128
    n_embd: int = 60
    n_head: int = 4
    n_layer: int = 3
    ff_mult: int = 2
    dropout: float = 0.05
    learning_rate: float = 2e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    train_fraction: float = 0.9
    num_steps: int = 8000
    warmup_steps: int = 300
    eval_interval: int = 2000
    eval_batches: int = 20
    generation_tokens: int = 300
    seed: int = 1337
    device: str | None = None


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(block_size, block_size))
        self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, channels = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:, :, :seq_len, :seq_len] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, channels)
        y = self.proj(y)
        return self.resid_dropout(y)


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        hidden_dim = ff_mult * n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = FeedForward(n_embd, ff_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class CausalTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.block_size = cfg.block_size
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.ff_mult, cfg.dropout)
                for _ in range(cfg.n_layer)
            ]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("sequence length exceeds block size")

        positions = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


def next_train_batch(
    train_loader: DataLoader,
    train_iter: object,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, object]:
    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)
    return xb.to(device), yb.to(device), train_iter


def get_lr(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if step >= cfg.num_steps:
        return cfg.min_learning_rate
    decay_ratio = (step - cfg.warmup_steps) / (cfg.num_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)


@torch.no_grad()
def estimate_loss(
    model: CausalTransformerLM,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split_name, loader in (("train", train_loader), ("test", test_loader)):
        split_losses = []
        loader_iter = iter(loader)
        for _ in range(cfg.eval_batches):
            try:
                xb, yb = next(loader_iter)
            except StopIteration:
                break
            xb = xb.to(device)
            yb = yb.to(device)
            _, loss = model(xb, yb)
            if loss is not None:
                split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / len(split_losses)
    model.train()
    return losses


def train(cfg: Config) -> tuple[CausalTransformerLM, object]:
    start_time = time.perf_counter()
    torch.manual_seed(cfg.seed)
    device = resolve_device(cfg.device)
    train_dataset, test_dataset, vocab = build_datasets(
        block_size=cfg.block_size,
        train_fraction=cfg.train_fraction,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    train_iter = iter(train_loader)

    print(f"device: {device}", flush=True)
    print(f"vocab size: {vocab.size}", flush=True)
    print(f"train batches: {len(train_loader)}", flush=True)
    print(f"test batches: {len(test_loader)}", flush=True)

    model = CausalTransformerLM(vocab.size, cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {num_params:,}", flush=True)

    if num_params >= 100_000:
        raise ValueError(f"parameter budget exceeded: {num_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )
    model.train()
    for step in range(cfg.num_steps):
        lr = get_lr(step, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        xb, yb, train_iter = next_train_batch(train_loader, train_iter, device)
        _, loss = model(xb, yb)
        if loss is None:
            raise RuntimeError("training loss was not computed")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.eval_interval == 0 or step == cfg.num_steps - 1:
            losses = estimate_loss(model, train_loader, test_loader, cfg, device)
            print(
                f"step {step:4d} | lr {lr:.2e} | "
                f"train loss {losses['train']:.4f} | test loss {losses['test']:.4f}",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time
    print(f"elapsed: {elapsed:.1f}s", flush=True)
    return model, vocab


def main() -> None:
    cfg = Config()
    device = resolve_device(cfg.device)
    model, vocab = train(cfg)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(context, max_new_tokens=cfg.generation_tokens)[0].tolist()
    print(vocab.decode(generated))


if __name__ == "__main__":
    main()
