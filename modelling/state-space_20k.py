from __future__ import annotations

from dataclasses import dataclass
import math

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
    n_embd: int = 38
    n_layer: int = 4
    state_size: int = 5
    s5_state_size: int = 128
    conv_kernel: int = 4
    block_type: str = "s4d"
    use_gate: bool = False
    dropout: float = 0.02
    learning_rate: float = 2.6e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.04
    grad_clip: float = 1.0
    train_fraction: float = 0.9
    num_steps: int = 6000
    warmup_steps: int = 300
    eval_interval: int = 1200
    eval_batches: int = 40
    generation_tokens: int = 500
    generation_temperature: float = 0.82
    generation_top_k: int = 12
    device: str | None = None


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class DiagonalSSMLayer(nn.Module):
    """S4D-style diagonal SSM block, optionally with a Mamba-like gate."""

    def __init__(self, n_embd: int, state_size: int, conv_kernel: int, dropout: float, use_gate: bool) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.use_gate = use_gate

        self.norm = nn.LayerNorm(n_embd)
        self.in_proj = nn.Linear(n_embd, 2 * n_embd if use_gate else n_embd)
        self.conv = nn.Conv1d(
            n_embd,
            n_embd,
            kernel_size=conv_kernel,
            groups=n_embd,
            padding=conv_kernel - 1,
        )

        # log_a is transformed to stable recurrence coefficients in (0, 1).
        self.log_a = nn.Parameter(torch.empty(n_embd, state_size))
        self.B = nn.Parameter(torch.empty(n_embd, state_size))
        self.C = nn.Parameter(torch.empty(n_embd, state_size))
        self.D = nn.Parameter(torch.ones(n_embd))

        self.out_proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.log_a, mean=-2.0, std=0.35)
        nn.init.normal_(self.B, mean=0.0, std=0.12)
        nn.init.normal_(self.C, mean=0.0, std=0.12)
        nn.init.ones_(self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        projected = self.in_proj(self.norm(x))
        if self.use_gate:
            u, gate = projected.chunk(2, dim=-1)
        else:
            u = projected
            gate = None
        u = self.conv(u.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)
        u = F.silu(u)

        y = self.ssm_convolution(u)
        if gate is not None:
            y = y * F.silu(gate)
        y = self.out_proj(y)
        return residual + self.dropout(y)

    def ssm_convolution(self, u: torch.Tensor) -> torch.Tensor:
        seq_len = u.size(1)
        decay = torch.exp(-F.softplus(self.log_a))
        powers = torch.arange(seq_len, device=u.device, dtype=decay.dtype)
        kernel = (self.B * self.C).unsqueeze(-1) * decay.unsqueeze(-1).pow(powers)
        kernel = kernel.sum(dim=1).unsqueeze(1)

        u_channels = u.transpose(1, 2)
        y = F.conv1d(u_channels, kernel.flip(-1), padding=seq_len - 1, groups=self.n_embd)
        y = y[:, :, :seq_len] + u_channels * self.D.view(1, -1, 1)
        return y.transpose(1, 2)


class S5Layer(nn.Module):
    """Small S5-like block with dense input/output maps and diagonal state dynamics."""

    def __init__(self, n_embd: int, state_size: int, dropout: float) -> None:
        super().__init__()
        self.state_size = state_size
        self.norm = nn.LayerNorm(n_embd)
        self.in_proj = nn.Linear(n_embd, state_size)
        self.log_a = nn.Parameter(torch.empty(state_size))
        self.B = nn.Parameter(torch.empty(state_size))
        self.C = nn.Parameter(torch.empty(state_size))
        self.D = nn.Parameter(torch.ones(state_size))
        self.out_proj = nn.Linear(state_size, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.log_a, mean=-2.0, std=0.35)
        nn.init.normal_(self.B, mean=0.0, std=0.12)
        nn.init.normal_(self.C, mean=0.0, std=0.12)
        nn.init.ones_(self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        u = F.silu(self.in_proj(self.norm(x)))
        seq_len = u.size(1)
        decay = torch.exp(-F.softplus(self.log_a))
        powers = torch.arange(seq_len, device=u.device, dtype=decay.dtype)
        kernel = (self.B * self.C).unsqueeze(-1) * decay.unsqueeze(-1).pow(powers)
        kernel = kernel.unsqueeze(1)

        u_states = u.transpose(1, 2)
        y = F.conv1d(u_states, kernel.flip(-1), padding=seq_len - 1, groups=self.state_size)
        y = y[:, :, :seq_len] + u_states * self.D.view(1, -1, 1)
        y = self.out_proj(y.transpose(1, 2))
        return residual + self.dropout(y)


class StateSpaceLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.block_size = cfg.block_size
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.block_type in {"mamba", "s4d"}:
            use_gate = cfg.use_gate if cfg.block_type == "mamba" else False
            self.blocks = nn.ModuleList(
                [
                    DiagonalSSMLayer(cfg.n_embd, cfg.state_size, cfg.conv_kernel, cfg.dropout, use_gate)
                    for _ in range(cfg.n_layer)
                ]
            )
        elif cfg.block_type == "s5":
            self.blocks = nn.ModuleList([S5Layer(cfg.n_embd, cfg.s5_state_size, cfg.dropout) for _ in range(cfg.n_layer)])
        else:
            raise ValueError(f"unknown block_type: {cfg.block_type}")
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, vocab_size)

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
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            next_token_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                values, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = next_token_logits.masked_fill(next_token_logits < values[:, [-1]], float("-inf"))
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


@torch.no_grad()
def estimate_loss(
    model: StateSpaceLM,
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


def learning_rate_for_step(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.num_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def train(cfg: Config) -> tuple[StateSpaceLM, object]:
    device = resolve_device(cfg.device)
    train_dataset, test_dataset, vocab = build_datasets(
        block_size=cfg.block_size,
        train_fraction=cfg.train_fraction,
    )

    generator = torch.Generator()
    generator.manual_seed(1337)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        generator=generator,
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    train_iter = iter(train_loader)

    print(f"device: {device}", flush=True)
    print(f"vocab size: {vocab.size}", flush=True)
    print(f"train batches: {len(train_loader)}", flush=True)
    print(f"test batches: {len(test_loader)}", flush=True)

    torch.manual_seed(1337)
    model = StateSpaceLM(vocab.size, cfg).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {num_params:,}", flush=True)
    if num_params >= 20_000:
        raise RuntimeError(f"parameter count must be under 20k, got {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    model.train()
    for step in range(cfg.num_steps):
        lr = learning_rate_for_step(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

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

    return model, vocab


def main() -> None:
    cfg = Config()
    device = resolve_device(cfg.device)
    model, vocab = train(cfg)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = model.generate(
        context,
        max_new_tokens=cfg.generation_tokens,
        temperature=cfg.generation_temperature,
        top_k=cfg.generation_top_k,
    )[0].tolist()
    print(vocab.decode(generated))


if __name__ == "__main__":
    main()
