"""Production BitNet language model.

This is the single architecture that ships — no toggles, no ablation flags.
The 8-bit / int4 / ternary mix below is what produced
`build/bitnet_quant_n56_full.pt` and what runs on the BBC Model B inside
`build/bitnet.uef`.

Architecture (3 stacked SSM blocks):

    token id ──► [emb int8] ─► residual stream ─┬─► (block 0) ─┐
                                                │              │
                                                └──────────────┴──► residual stream ─► ...

Per block:
                                          ┌─► [out_proj ternary] ─► [+residual] ──►
    x ─► [in_proj ternary] ─► split:      │
                              ├─ u  ─► [conv int4] ─► [SSM (B ternary, C int4)] ─┐
                              └─ gate ─────────────────────────────────────────► [*]
                                                                                  │
                                                                                  └ multiplicative gate

After last block:
    x ─► [Int4Head] ─► int16 logits ─► argmax / softmax sample
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

from modelling.quant import (
    ternary_quantize, int4_quantize, fake_quant_int8, fake_quant_int16,
    saturating_shift_int8, learned_shift_no_sat, floor_div_pow2,
)


@dataclass
class ModelConfig:
    """Architecture-only config — defines what the model looks like.

    Defaults match the production n_embd=56 model (`bitnet_quant_n56_full.pt`).
    Init shifts are starting values; they're learned thereafter via STE.
    """
    vocab_size: int = 27
    block_size: int = 64
    n_embd: int = 56
    n_layer: int = 3
    state_size: int = 8
    conv_kernel: int = 4

    # Initial per-stage activation-shift values (powers of two). These are
    # learned during training; the initial values are chosen so each stage's
    # output stays small at init, leaving the residual stream room to grow
    # without saturating.
    init_shift_in_proj: float = 3.0
    init_shift_conv: float = 4.0       # int4 conv has wider range → larger shift than ternary
    init_shift_ssm_out: float = 6.0    # int4 C has wider range → larger shift than ternary
    init_shift_d: float = 8.0
    init_shift_gate: float = 7.0
    init_shift_out_proj: float = 5.0
    init_shift_head: float = 8.0       # int4 head has wider range → larger shift than ternary
    embed_init: float = 16.0


# ----------------------------------------------------------------------------- #
# Layer modules
# ----------------------------------------------------------------------------- #


class TernaryLinear(nn.Module):
    """int8 input · ternary weight + int16 bias → int16 acc → shift → int8 out."""

    def __init__(self, in_f: int, out_f: int, init_shift: float) -> None:
        super().__init__()
        self.in_features = in_f
        self.out_features = out_f
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.shift = nn.Parameter(torch.tensor(init_shift))
        # Init range covers ~50% nonzero after ternary rounding.
        nn.init.uniform_(self.weight, -1.0, 1.0)
        nn.init.uniform_(self.bias, -4.0, 4.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = ternary_quantize(self.weight)
        acc = F.linear(x, w) + fake_quant_int16(self.bias)
        return saturating_shift_int8(acc, self.shift)


class Int4Head(nn.Module):
    """Output head: int8 input · int4 weight → int16 logits (no bias, no saturation).

    Argmax is scale-invariant, so we skip int8 saturation here. The shift
    is purely a training-time scaling knob.
    """

    def __init__(self, in_f: int, vocab_size: int, init_shift: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, in_f))
        self.shift = nn.Parameter(torch.tensor(init_shift))
        # Init wide enough that ~all int4 levels are used at init.
        nn.init.uniform_(self.weight, -7.0, 7.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = F.linear(x, int4_quantize(self.weight))
        return learned_shift_no_sat(acc, self.shift)


class BitNetBlock(nn.Module):
    """One SSM block: in_proj → conv1d → diagonal SSM → gate → out_proj → residual."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        C, S, K = cfg.n_embd, cfg.state_size, cfg.conv_kernel
        self.n_embd = C
        self.state_size = S
        self.conv_kernel = K

        # in_proj projects (B, T, C) → (B, T, 2C). We split the output into
        # `u` (fed to the SSM path) and `gate` (used as a multiplicative gate
        # on the SSM output).
        self.in_proj = TernaryLinear(C, 2 * C, cfg.init_shift_in_proj)

        # Depthwise conv kernel — int4 weights ([-7, +7]).
        self.conv_weight = nn.Parameter(torch.empty(C, K))
        nn.init.uniform_(self.conv_weight, -7.0, 7.0)
        self.conv_shift = nn.Parameter(torch.tensor(cfg.init_shift_conv))

        # Diagonal SSM parameters.
        self.decay = nn.Parameter(torch.empty(C, S))   # int8 in [0, 127], effective decay = a / 128
        self.B = nn.Parameter(torch.empty(C, S))       # ternary
        self.C = nn.Parameter(torch.empty(C, S))       # int4 (name kept for checkpoint compat)
        self.D = nn.Parameter(torch.empty(C))          # int8
        nn.init.uniform_(self.decay, 100.0, 124.0)     # ≈ 0.78–0.97 effective decay
        nn.init.uniform_(self.B, -1.0, 1.0)
        nn.init.uniform_(self.C, -7.0, 7.0)
        nn.init.uniform_(self.D, 8.0, 32.0)

        self.ssm_out_shift = nn.Parameter(torch.tensor(cfg.init_shift_ssm_out))
        self.d_shift = nn.Parameter(torch.tensor(cfg.init_shift_d))
        self.gate_shift = nn.Parameter(torch.tensor(cfg.init_shift_gate))

        # Output projection back to residual width.
        self.out_proj = TernaryLinear(C, C, cfg.init_shift_out_proj)

    def forward(self, x: torch.Tensor, state: torch.Tensor | None = None
                ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        projected = self.in_proj(x)
        u, gate = projected.chunk(2, dim=-1)
        u = self._conv(u)
        y, state = self._ssm(u, state)
        y = saturating_shift_int8(y * gate, self.gate_shift)
        y = self.out_proj(y)
        out = fake_quant_int8(residual + y)
        return out, state

    def _conv(self, u: torch.Tensor) -> torch.Tensor:
        B, T, C = u.shape
        K = self.conv_kernel
        w = int4_quantize(self.conv_weight)
        u_padded = F.pad(u, (0, 0, K - 1, 0))             # causal left-pad
        u_ch = u_padded.transpose(1, 2)                   # (B, C, T+K-1)
        kernel = w.unsqueeze(1)                           # (C, 1, K)
        acc = F.conv1d(u_ch, kernel, groups=C).transpose(1, 2)
        return saturating_shift_int8(acc, self.conv_shift)

    def _ssm(self, u: torch.Tensor, state: torch.Tensor | None
             ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, Cd = u.shape
        S = self.state_size

        decay_q = fake_quant_int8(torch.clamp(self.decay, 0.0, 127.0))
        B_t = ternary_quantize(self.B)
        C_t = int4_quantize(self.C)
        D_q = fake_quant_int8(self.D)

        if state is None:
            state = torch.zeros(B, Cd, S, device=u.device, dtype=u.dtype)

        outs = []
        for t in range(T):
            u_t = u[:, t, :]                                     # (B, C)
            decayed = decay_q.unsqueeze(0) * state               # (B, C, S) int16-valued
            decayed_q = floor_div_pow2(decayed, 7)               # back into int8 range
            b_u = B_t.unsqueeze(0) * u_t.unsqueeze(-1)           # (B, C, S)
            state = fake_quant_int8(decayed_q + b_u)
            c_state = (C_t.unsqueeze(0) * state).sum(dim=-1)     # (B, C) int16
            c_part = saturating_shift_int8(c_state, self.ssm_out_shift)
            d_part = saturating_shift_int8(D_q.unsqueeze(0) * u_t, self.d_shift)
            y_t = fake_quant_int8(c_part + d_part)
            outs.append(y_t)
        return torch.stack(outs, dim=1), state


# ----------------------------------------------------------------------------- #
# Top-level model
# ----------------------------------------------------------------------------- #


class BitNetLM(nn.Module):
    """Stacked-SSM character-level language model.

    No positional embedding (SSM is autoregressive by construction).
    No RMSNorm. No dropout. No tied embeddings.
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.block_size = cfg.block_size
        self.vocab_size = cfg.vocab_size

        self.token_embedding = nn.Parameter(torch.empty(cfg.vocab_size, cfg.n_embd))
        nn.init.uniform_(self.token_embedding, -cfg.embed_init, cfg.embed_init)

        self.blocks = nn.ModuleList([BitNetBlock(cfg) for _ in range(cfg.n_layer)])
        self.head = Int4Head(cfg.n_embd, cfg.vocab_size, cfg.init_shift_head)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                states: list[torch.Tensor] | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor | None, list[torch.Tensor]]:
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block size {self.block_size}")

        x = fake_quant_int8(self.token_embedding[idx])

        new_states: list[torch.Tensor] = []
        for i, block in enumerate(self.blocks):
            s_in = states[i] if states is not None else None
            x, s_out = block(x, s_in)
            new_states.append(s_out)

        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss, new_states

    def ternary_param_count(self) -> int:
        n = 0
        for b in self.blocks:
            n += b.in_proj.weight.numel()
            n += b.out_proj.weight.numel()
            n += b.B.numel()
        return n

    def int4_param_count(self) -> int:
        n = 0
        for b in self.blocks:
            n += b.conv_weight.numel()
            n += b.C.numel()
        n += self.head.weight.numel()
        return n


def freeze_shift_params(model: nn.Module, freeze: bool = True) -> int:
    """Freeze (or unfreeze) every learned right-shift parameter.

    Catches both `.shift` (TernaryLinear, Int4Head) and the per-block
    `conv_shift`, `ssm_out_shift`, `d_shift`, `gate_shift`. Called partway
    through training (typically at 50% of steps) to lock in the activation
    range and let the rest of training settle.
    """
    n = 0
    for name, p in model.named_parameters():
        leaf = name.split(".")[-1]
        if leaf == "shift" or leaf.endswith("_shift"):
            p.requires_grad = not freeze
            n += 1
    return n
