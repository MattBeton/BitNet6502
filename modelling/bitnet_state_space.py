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
    n_embd: int = 82
    n_layer: int = 3
    state_size: int = 8
    conv_kernel: int = 4
    use_gate: bool = True
    act_bound: float = 2.0
    dropout: float = 0.02
    learning_rate: float = 2.0e-2
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


def ternary_quantize_ste(w: torch.Tensor) -> torch.Tensor:
    """Absmean ternary quantization with straight-through estimator.

    Uses the BitNet b1.58 approach: scale by mean(|W|), round to {-1, 0, +1},
    then reapply scale. Gradients pass through unchanged (STE).
    """
    alpha = w.abs().mean().clamp(min=1e-8)
    w_q = torch.clamp(torch.round(w / alpha), -1.0, 1.0)
    # STE: forward computes w_q*alpha, backward treats it as identity on w
    return w + (w_q * alpha - w).detach()


class TernaryLinear(nn.Module):
    """Linear layer whose weight matrix is ternary {-1, 0, +1} at inference time.

    Full-precision weights are kept for training; quantization is applied in the
    forward pass via a straight-through estimator.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(in_features) # it seems like this bias is not quantized? The bias could be quantized in int16, given that we are doing int16 accumulate to int16 after activation fn.
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, ternary_quantize_ste(self.weight), self.bias)

    def ternary_count(self) -> int:
        return self.weight.numel()


class DiagonalSSMLayer(nn.Module):
    """S4D-style diagonal SSM block with BitNet ternary linear projections.

    When use_gate=True, in_proj outputs 2*n_embd: the first half feeds the SSM
    and the second half is a SiLU-gated multiplicative path (Mamba-style).
    """

    def __init__(
        self, n_embd: int, state_size: int, conv_kernel: int, dropout: float,
        use_gate: bool = False, act_bound: float = 1.0
    ) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.state_size = state_size
        self.use_gate = use_gate
        self.act_bound = act_bound

        self.in_proj = TernaryLinear(n_embd, 2 * n_embd if use_gate else n_embd)
        self.conv = nn.Conv1d(
            n_embd, n_embd, kernel_size=conv_kernel, groups=n_embd, padding=conv_kernel - 1
        )
        self.log_a = nn.Parameter(torch.empty(n_embd, state_size))
        self.B = nn.Parameter(torch.empty(n_embd, state_size))
        self.C = nn.Parameter(torch.empty(n_embd, state_size))
        self.D = nn.Parameter(torch.ones(n_embd))
        self.out_proj = TernaryLinear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
        self._init_ssm()

    def _init_ssm(self) -> None:
        nn.init.normal_(self.log_a, mean=-2.0, std=0.35)
        nn.init.normal_(self.B, mean=0.0, std=0.12)
        nn.init.normal_(self.C, mean=0.0, std=0.12)
        nn.init.ones_(self.D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        projected = self.in_proj(x)
        if self.use_gate:
            u, gate = projected.chunk(2, dim=-1)
        else:
            u, gate = projected, None
        u = self.conv(u.transpose(1, 2))[:, :, : x.size(1)].transpose(1, 2)
        u = F.hardtanh(u, min_val=-self.act_bound, max_val=self.act_bound)
        y = self._ssm_conv(u)
        if gate is not None:
            y = y * F.hardtanh(gate, min_val=-self.act_bound, max_val=self.act_bound)
        y = self.out_proj(y)
        return residual + self.dropout(y)

    def _ssm_conv(self, u: torch.Tensor) -> torch.Tensor:
        seq_len = u.size(1)
        decay = torch.exp(-F.softplus(self.log_a)) # we can't exponentiate on the 6502. We can only use this if it's hardcoded.
        powers = torch.arange(seq_len, device=u.device, dtype=decay.dtype)
        kernel = (self.B * self.C).unsqueeze(-1) * decay.unsqueeze(-1).pow(powers)
        kernel = kernel.sum(dim=1).unsqueeze(1)
        u_ch = u.transpose(1, 2)
        y = F.conv1d(u_ch, kernel.flip(-1), padding=seq_len - 1, groups=self.n_embd)
        y = y[:, :, :seq_len] + u_ch * self.D.view(1, -1, 1)
        return y.transpose(1, 2)


class BitNetStateSpaceLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.block_size = cfg.block_size
        self.token_embedding = nn.Embedding(vocab_size, cfg.n_embd)
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            [DiagonalSSMLayer(cfg.n_embd, cfg.state_size, cfg.conv_kernel, cfg.dropout, cfg.use_gate, cfg.act_bound)
             for _ in range(cfg.n_layer)]
        )
        self.head = TernaryLinear(cfg.n_embd, vocab_size, bias=False)

    def ternary_param_count(self) -> int:
        return sum(m.ternary_count() for m in self.modules() if isinstance(m, TernaryLinear))

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        _, seq_len = idx.shape
        if seq_len > self.block_size:
            raise ValueError("sequence length exceeds block size")
        pos = torch.arange(seq_len, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self, idx: torch.Tensor, max_new_tokens: int, temperature: float, top_k: int
    ) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size :]
            logits, _ = self(idx_cond)
            next_logits = logits[:, -1, :] / temperature
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], float("-inf"))
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_tok), dim=1)
        return idx


def next_train_batch(
    train_loader: DataLoader, train_iter: object, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, object]:
    try:
        xb, yb = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        xb, yb = next(train_iter)
    return xb.to(device), yb.to(device), train_iter


@torch.no_grad()
def estimate_loss(
    model: BitNetStateSpaceLM,
    train_loader: DataLoader,
    test_loader: DataLoader,
    cfg: Config,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    losses = {}
    for split_name, loader in (("train", train_loader), ("test", test_loader)):
        split_losses = []
        for xb, yb in loader:
            if len(split_losses) >= cfg.eval_batches:
                break
            _, loss = model(xb.to(device), yb.to(device))
            if loss is not None:
                split_losses.append(loss.item())
        losses[split_name] = sum(split_losses) / max(len(split_losses), 1)
    model.train()
    return losses


def learning_rate_for_step(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.num_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def train(cfg: Config) -> tuple[BitNetStateSpaceLM, object]:
    device = resolve_device(cfg.device)
    train_dataset, test_dataset, vocab = build_datasets(
        block_size=cfg.block_size, train_fraction=cfg.train_fraction
    )
    generator = torch.Generator()
    generator.manual_seed(1337)
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=generator
    )
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    train_iter = iter(train_loader)

    print(f"device: {device}", flush=True)
    print(f"vocab size: {vocab.size}", flush=True)

    torch.manual_seed(1337)
    model = BitNetStateSpaceLM(vocab.size, cfg).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    ternary_params = model.ternary_param_count()
    print(f"total parameters: {total_params:,}", flush=True)
    print(f"ternary parameters: {ternary_params:,}", flush=True)
    if total_params >= 80_000:
        raise RuntimeError(f"parameter count must be under 80k, got {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    model.train()
    for step in range(cfg.num_steps):
        lr = learning_rate_for_step(step, cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr

        xb, yb, train_iter = next_train_batch(train_loader, train_iter, device)
        _, loss = model(xb, yb)
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
