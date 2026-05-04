"""BitNet quantized state-space LM with strict int8/int16 simulation.

Single-file train + inference. Every intermediate value is fake-quantized
during training to match the integer arithmetic that will run on a 6502.
Forward output is bit-exact to what an int-only implementation would produce.

Datatype contract:
  - Activations & residual stream: int8 ([-128, 127])
  - Matmul / accumulator scratch:  int16 ([-32768, 32767])
  - Weights:                       ternary {-1, 0, +1}
  - Biases:                        int16
  - SSM decay:                     int8 in [0, 127]; effective decay = a / 128
  - SSM B, C:                      ternary
  - SSM D:                         int8
  - Per-layer activation scale:    learned non-negative integer shift,
                                   applied as (acc >> shift) → saturate to int8
  - Token / position embeddings:   int8

Operations on the 6502 inference path:
  - signed add, ternary signed add (== conditional add/sub), shift, clip
  - argmax for sampling (no softmax / exp / division anywhere)
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import sys
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

try:
    from modelling.shakespeare import build_datasets
except ModuleNotFoundError:
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from shakespeare import build_datasets


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass
class Config:
    block_size: int = 64
    batch_size: int = 128
    n_embd: int = 80           # multiple of 4 (bit-pack constraint)
    n_layer: int = 3
    state_size: int = 8
    conv_kernel: int = 4
    use_gate: bool = True
    use_pos_embed: bool = True
    dropout: float = 0.02
    learning_rate: float = 2.0e-3
    min_learning_rate: float = 1e-4
    weight_decay: float = 0.04
    grad_clip: float = 1.0
    train_fraction: float = 0.9
    num_steps: int = 6000
    warmup_steps: int = 300
    eval_interval: int = 600
    eval_batches: int = 40
    generation_tokens: int = 400
    generation_temperature: float = 0.9
    generation_top_k: int = 8
    device: str | None = None
    # Initial shift values (per-layer power-of-two activation scale)
    # Shifts chosen so each stage's output stays small at init, leaving the
    # residual stream room to grow without saturating. All shifts are learned
    # thereafter via STE on a continuous proxy.
    init_shift_in_proj: float = 3.0    # 80-wide ternary*int8 → std ≈ 175 → /8 ≈ 22
    init_shift_conv: float = 1.0       # 4-wide ternary*int8 → std ≈ 40 → /2 ≈ 20
    init_shift_ssm_out: float = 3.0    # 8-wide ternary*int8 → std ≈ 200 → /8 ≈ 25
    init_shift_d: float = 8.0          # int8 * int8 → /256
    init_shift_gate: float = 7.0       # int8 * int8 → /128
    init_shift_out_proj: float = 5.0   # block output small → residual stays in range
    init_shift_head: float = 5.0       # logits ~ int16 / 32 → softmax-friendly
    embed_init: float = 16.0           # token_embedding U[-16, 16]
    pos_embed_init: float = 8.0
    # Ablation flags for diagnosing the gap to the unquantized baseline.
    # All False = full int8/int16/ternary simulation (the deployable model).
    ablate_float_weights: bool = False     # weights stay float (no ternary round)
    ablate_float_acts: bool = False        # activations stay float (no int8 fake-quant)
    ablate_no_saturate: bool = False       # remove the [-128,127] clip (still shift)
    ablate_no_shift: bool = False          # remove the per-layer shift entirely


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# -----------------------------------------------------------------------------
# Quantization primitives — fake-quants whose forward equals the integer
# operation exactly and whose backward passes a smooth gradient (STE).
# -----------------------------------------------------------------------------

# Module-level ablation toggles, set from cfg before training. Defaults to
# full quantization (the deployable mode). When toggled to float, the quantize
# functions become identity, which lets us measure the cost of each constraint
# in isolation.
_ABLATION = {
    "float_weights": False,
    "float_acts":    False,
    "no_saturate":   False,
    "no_shift":      False,
}


def set_ablation_from_cfg(cfg: "Config") -> None:
    _ABLATION["float_weights"] = cfg.ablate_float_weights
    _ABLATION["float_acts"]    = cfg.ablate_float_acts
    _ABLATION["no_saturate"]   = cfg.ablate_no_saturate
    _ABLATION["no_shift"]      = cfg.ablate_no_shift


def fake_quant_int8(x: torch.Tensor) -> torch.Tensor:
    if _ABLATION["float_acts"]:
        return x
    q = torch.clamp(torch.round(x), -128.0, 127.0)
    return x + (q - x).detach()


def fake_quant_int16(x: torch.Tensor) -> torch.Tensor:
    if _ABLATION["float_acts"]:
        return x
    q = torch.clamp(torch.round(x), -32768.0, 32767.0)
    return x + (q - x).detach()


def ternary_quantize(w: torch.Tensor) -> torch.Tensor:
    """Round to {-1, 0, +1}. No per-tensor alpha; the per-layer activation
    shift absorbs all scaling, leaving weights as pure ternary."""
    if _ABLATION["float_weights"]:
        return w
    q = torch.clamp(torch.round(w), -1.0, 1.0)
    return w + (q - w).detach()


def shift_round(s: torch.Tensor, max_shift: int = 14) -> torch.Tensor:
    q = torch.clamp(torch.round(s), 0.0, float(max_shift))
    return s + (q - s).detach()


def saturating_shift_int8(acc: torch.Tensor, shift_param: torch.Tensor) -> torch.Tensor:
    """Apply learned right-shift then saturate to int8.

    Forward: floor(acc / 2^round(shift)), clamped to [-128, 127] — bit-exact
    to `acc >> shift; saturate` on a signed integer machine.
    Backward: gradient flows through smooth `acc / 2^shift_continuous`, so
    both `acc` and `shift` receive informative gradients.
    """
    if _ABLATION["no_shift"]:
        # No scaling at all — raw matmul output passes through.
        return acc
    s_int = shift_round(shift_param)
    div = torch.pow(2.0, s_int)
    if _ABLATION["no_saturate"]:
        # Shift but don't clip — used to isolate the cost of int8 saturation.
        smooth = acc / div
        hard = torch.floor(acc / div)
        return smooth + (hard - smooth).detach()
    smooth = acc / div
    hard = torch.clamp(torch.floor(acc / div), -128.0, 127.0)
    return smooth + (hard - smooth).detach()


def floor_div_pow2(x: torch.Tensor, shift_const: int) -> torch.Tensor:
    """Forward: floor(x / 2^shift). Backward: identity-scaled. Used for
    fixed (non-learned) shifts like the >>7 inside the SSM decay step."""
    div = float(1 << shift_const)
    smooth = x / div
    hard = torch.floor(x / div)
    return smooth + (hard - smooth).detach()


def learned_shift_no_sat(acc: torch.Tensor, shift_param: torch.Tensor) -> torch.Tensor:
    """Learned right-shift, no saturation. For the head: argmax is invariant
    to a positive scale, so the shift is purely a training-time scaling knob.
    On 6502 inference the head can ignore it entirely."""
    s_int = shift_round(shift_param)
    div = torch.pow(2.0, s_int)
    smooth = acc / div
    hard = torch.floor(acc / div)
    return smooth + (hard - smooth).detach()


# -----------------------------------------------------------------------------
# Modules
# -----------------------------------------------------------------------------

class QuantTernaryLinear(nn.Module):
    """int8 input · ternary weight + int16 bias → int16 acc → shift → int8 out."""

    def __init__(self, in_f: int, out_f: int, bias: bool = True, init_shift: float = 2.0) -> None:
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
        self.shift = nn.Parameter(torch.tensor(init_shift))
        # Init wide enough that ~50% of weights survive the {-1,0,+1} rounding.
        # Uniform [-1, 1] → P(±1) = 0.25 each, P(0) = 0.5.
        nn.init.uniform_(self.weight, -1.0, 1.0)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -4.0, 4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = ternary_quantize(self.weight)
        acc = F.linear(x, w)              # int16-valued
        if self.bias is not None:
            acc = acc + fake_quant_int16(self.bias)
        return saturating_shift_int8(acc, self.shift)

    def ternary_count(self) -> int:
        return self.weight.numel()


class QuantHead(nn.Module):
    """Vocab head: int8 input · ternary weight → int16 logits.

    A learned right-shift scales the int16 logits into a softmax-friendly
    range during training. On 6502 inference, argmax is invariant to that
    monotonic scale — the shift is a no-op there.
    """

    def __init__(self, in_f: int, vocab_size: int, init_shift: float = 5.0) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, in_f))
        self.shift = nn.Parameter(torch.tensor(init_shift))
        nn.init.uniform_(self.weight, -1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = F.linear(x, ternary_quantize(self.weight))
        return learned_shift_no_sat(acc, self.shift)

    def ternary_count(self) -> int:
        return self.weight.numel()


class QuantSSMLayer(nn.Module):
    """Recurrent diagonal SSM with ternary projections and int8 state."""

    def __init__(self, n_embd: int, state_size: int, conv_kernel: int,
                 dropout: float, use_gate: bool, cfg: Config) -> None:
        super().__init__()
        self.n_embd = n_embd
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.use_gate = use_gate

        proj_out = 2 * n_embd if use_gate else n_embd
        self.in_proj = QuantTernaryLinear(n_embd, proj_out, init_shift=cfg.init_shift_in_proj)

        self.conv_weight = nn.Parameter(torch.empty(n_embd, conv_kernel))
        nn.init.uniform_(self.conv_weight, -1.0, 1.0)
        self.conv_shift = nn.Parameter(torch.tensor(cfg.init_shift_conv))

        self.decay = nn.Parameter(torch.empty(n_embd, state_size))
        self.B = nn.Parameter(torch.empty(n_embd, state_size))
        self.C = nn.Parameter(torch.empty(n_embd, state_size))
        self.D = nn.Parameter(torch.empty(n_embd))
        nn.init.uniform_(self.decay, 100.0, 124.0)        # ≈ 0.78–0.97
        nn.init.uniform_(self.B, -1.0, 1.0)
        nn.init.uniform_(self.C, -1.0, 1.0)
        nn.init.uniform_(self.D, 8.0, 32.0)

        self.ssm_out_shift = nn.Parameter(torch.tensor(cfg.init_shift_ssm_out))
        self.d_shift = nn.Parameter(torch.tensor(cfg.init_shift_d))
        if use_gate:
            self.gate_shift = nn.Parameter(torch.tensor(cfg.init_shift_gate))

        self.out_proj = QuantTernaryLinear(n_embd, n_embd, init_shift=cfg.init_shift_out_proj)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        projected = self.in_proj(x)
        if self.use_gate:
            u, gate = projected.chunk(2, dim=-1)
        else:
            u, gate = projected, None
        u = self._depthwise_conv(u)
        y, state = self._ssm_recurrent(u, state)
        if gate is not None:
            y = saturating_shift_int8(y * gate, self.gate_shift)
        y = self.out_proj(y)
        y = self.dropout(y)
        out = fake_quant_int8(residual + y)
        return out, state

    def _depthwise_conv(self, u: torch.Tensor) -> torch.Tensor:
        B, T, C = u.shape
        K = self.conv_kernel
        w = ternary_quantize(self.conv_weight)
        u_padded = F.pad(u, (0, 0, K - 1, 0))             # left-pad for causality
        u_ch = u_padded.transpose(1, 2)                   # (B, C, T+K-1)
        kernel = w.unsqueeze(1)                           # (C, 1, K)
        acc = F.conv1d(u_ch, kernel, groups=C).transpose(1, 2)
        return saturating_shift_int8(acc, self.conv_shift)

    def _ssm_recurrent(self, u: torch.Tensor, state: torch.Tensor | None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = u.shape
        S = self.state_size

        decay_q = fake_quant_int8(torch.clamp(self.decay, 0.0, 127.0))   # (C, S)
        B_t = ternary_quantize(self.B)                                   # (C, S)
        C_t = ternary_quantize(self.C)                                   # (C, S)
        D_q = fake_quant_int8(self.D)                                    # (C,)

        if state is None:
            state = torch.zeros(B, C, S, device=u.device, dtype=u.dtype)

        outs = []
        for t in range(T):
            u_t = u[:, t, :]                                             # (B, C)
            decayed = decay_q.unsqueeze(0) * state                       # (B, C, S) int16
            decayed_q = floor_div_pow2(decayed, 7)                       # back to int8 range
            b_u = B_t.unsqueeze(0) * u_t.unsqueeze(-1)                   # (B, C, S)
            state = fake_quant_int8(decayed_q + b_u)
            c_state = (C_t.unsqueeze(0) * state).sum(dim=-1)             # (B, C) int16
            c_part = saturating_shift_int8(c_state, self.ssm_out_shift)  # int8
            d_part = saturating_shift_int8(D_q.unsqueeze(0) * u_t, self.d_shift)
            y_t = fake_quant_int8(c_part + d_part)
            outs.append(y_t)
        y = torch.stack(outs, dim=1)
        return y, state


class QuantBitNetLM(nn.Module):
    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.vocab_size = vocab_size

        self.token_embedding = nn.Parameter(torch.empty(vocab_size, cfg.n_embd))
        nn.init.uniform_(self.token_embedding, -cfg.embed_init, cfg.embed_init)
        if cfg.use_pos_embed:
            self.position_embedding = nn.Parameter(torch.empty(cfg.block_size, cfg.n_embd))
            nn.init.uniform_(self.position_embedding, -cfg.pos_embed_init, cfg.pos_embed_init)
        else:
            self.register_parameter("position_embedding", None)

        self.dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            QuantSSMLayer(cfg.n_embd, cfg.state_size, cfg.conv_kernel,
                          cfg.dropout, cfg.use_gate, cfg)
            for _ in range(cfg.n_layer)
        ])
        self.head = QuantHead(cfg.n_embd, vocab_size, init_shift=cfg.init_shift_head)

    def ternary_param_count(self) -> int:
        n = 0
        for b in self.blocks:
            n += b.in_proj.weight.numel()
            n += b.out_proj.weight.numel()
            n += b.conv_weight.numel()
            n += b.B.numel() + b.C.numel()
        n += self.head.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                states: list[torch.Tensor] | None = None,
                pos_offset: int = 0
                ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("sequence length exceeds block size")

        tok = fake_quant_int8(self.token_embedding[idx])
        if self.position_embedding is not None:
            pos = fake_quant_int8(self.position_embedding[pos_offset:pos_offset + T])
            x = fake_quant_int8(tok + pos)
        else:
            x = tok
        x = self.dropout(x)

        new_states: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            s_in = states[i] if states is not None else None
            x, s_out = block(x, s_in)
            new_states.append(s_out)

        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_states

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 0,
                 greedy: bool = False) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _, _ = self(idx_cond)
            next_logits = logits[:, -1, :]
            if greedy:
                next_tok = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits = next_logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                    next_logits = next_logits.masked_fill(next_logits < v[:, [-1]], float("-inf"))
                probs = F.softmax(next_logits, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_tok], dim=1)
        return idx


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def next_train_batch(loader: DataLoader, it, device: torch.device):
    try:
        xb, yb = next(it)
    except StopIteration:
        it = iter(loader)
        xb, yb = next(it)
    return xb.to(device), yb.to(device), it


@torch.no_grad()
def estimate_loss(model: QuantBitNetLM, train_loader: DataLoader,
                  test_loader: DataLoader, cfg: Config, device: torch.device
                  ) -> dict[str, float]:
    model.eval()
    out: dict[str, float] = {}
    for split_name, loader in (("train", train_loader), ("test", test_loader)):
        losses: list[float] = []
        for xb, yb in loader:
            if len(losses) >= cfg.eval_batches:
                break
            _, loss, _ = model(xb.to(device), yb.to(device))
            if loss is not None:
                losses.append(loss.item())
        out[split_name] = sum(losses) / max(len(losses), 1)
    model.train()
    return out


def lr_for_step(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.num_steps - cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_learning_rate + cosine * (cfg.learning_rate - cfg.min_learning_rate)


def train(cfg: Config, verbose: bool = True) -> tuple[QuantBitNetLM, object]:
    set_ablation_from_cfg(cfg)
    device = resolve_device(cfg.device)
    train_ds, test_ds, vocab = build_datasets(
        block_size=cfg.block_size, train_fraction=cfg.train_fraction
    )
    g = torch.Generator()
    g.manual_seed(1337)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              drop_last=True, generator=g)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    train_iter = iter(train_loader)

    if verbose:
        print(f"device: {device}", flush=True)
        print(f"vocab size: {vocab.size}", flush=True)

    torch.manual_seed(1337)
    model = QuantBitNetLM(vocab.size, cfg).to(device)

    total = sum(p.numel() for p in model.parameters())
    ternary = model.ternary_param_count()
    if verbose:
        print(f"total parameters: {total:,}", flush=True)
        print(f"ternary parameters: {ternary:,}", flush=True)
    if total >= 80_000:
        raise RuntimeError(f"parameter count must be under 80k, got {total:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    model.train()
    t0 = time.time()
    for step in range(cfg.num_steps):
        lr = lr_for_step(step, cfg)
        for grp in opt.param_groups:
            grp["lr"] = lr

        xb, yb, train_iter = next_train_batch(train_loader, train_iter, device)
        _, loss, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.eval_interval == 0 or step == cfg.num_steps - 1:
            losses = estimate_loss(model, train_loader, test_loader, cfg, device)
            elapsed = time.time() - t0
            if verbose:
                print(
                    f"step {step:4d} | lr {lr:.2e} | "
                    f"train {losses['train']:.4f} | test {losses['test']:.4f} | "
                    f"{elapsed:5.1f}s",
                    flush=True,
                )

    return model, vocab


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------

@torch.no_grad()
def diagnose_ranges(model: QuantBitNetLM, sample_idx: torch.Tensor) -> None:
    """Print min/max of activations at every layer to sanity-check int8/int16 limits."""
    print("\n=== range diagnostics ===")
    model.eval()
    B, T = sample_idx.shape
    tok = fake_quant_int8(model.token_embedding[sample_idx])
    print(f"token_emb_quant     min={tok.min().item():8.1f} max={tok.max().item():8.1f}")
    if model.position_embedding is not None:
        pos = fake_quant_int8(model.position_embedding[:T])
        x = fake_quant_int8(tok + pos)
        print(f"after pos+sat       min={x.min().item():8.1f} max={x.max().item():8.1f}")
    else:
        x = tok
    state = None
    for i, block in enumerate(model.blocks):
        proj = block.in_proj(x)
        print(f"block {i} in_proj    min={proj.min().item():8.1f} max={proj.max().item():8.1f} shift={int(round(block.in_proj.shift.item()))}")
        x, _ = block(x, state)
        print(f"block {i} out (resid)min={x.min().item():8.1f} max={x.max().item():8.1f}")
    logits = model.head(x)
    print(f"head logits         min={logits.min().item():8.1f} max={logits.max().item():8.1f}")
    print()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    cfg = Config()
    device = resolve_device(cfg.device)
    model, vocab = train(cfg)

    print("\n=== sample (greedy) ===", flush=True)
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    greedy = model.generate(ctx, max_new_tokens=cfg.generation_tokens, greedy=True)[0].tolist()
    print(vocab.decode(greedy))

    print("\n=== sample (top-k) ===", flush=True)
    ctx = torch.zeros((1, 1), dtype=torch.long, device=device)
    sampled = model.generate(ctx, max_new_tokens=cfg.generation_tokens,
                             temperature=cfg.generation_temperature,
                             top_k=cfg.generation_top_k)[0].tolist()
    print(vocab.decode(sampled))


if __name__ == "__main__":
    main()
