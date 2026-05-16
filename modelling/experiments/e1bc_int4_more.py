"""E1b / E1c — extend int4 to SSM C and conv kernel.

Hypothesis (from bitnet_quant_experiments.md):
    int4 on small-fan-in 'decision-point' tensors recovers some of the
    0.21-nat ternary cost. E1a tested head-only; E1b adds SSM C (84×8
    per layer); E1c adds the depthwise conv kernel (84×4 per layer).
    Each step doubles those tensors' bytes and shrinks n_embd to keep
    the total at 23,770.

Method:
    Reuse Int4Head from e1a. For E1b/E1c, additionally subclass
    QuantSSMLayer to replace `ternary_quantize(self.C)` and/or
    `ternary_quantize(self.conv_weight)` with int4_quantize, and widen
    the param init range to U[-7,7] so int4 levels are usable at init.

Defined locally: Int4SSMLayer, Int4MoreBitNetLM (extends the E1a
model). Only run E1b/E1c if E1a delivered ≥ -0.03 nats.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Config, train_loop
from budget import compute_total_bytes, solve_n_embd_for_budget
from e1a_int4_head import Int4Head, int4_quantize

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bitnet_quant import (  # noqa: E402
    QuantSSMLayer, QuantBitNetLM, QuantTernaryLinear,
    fake_quant_int8, ternary_quantize, saturating_shift_int8, floor_div_pow2,
)


class Int4SSMLayer(QuantSSMLayer):
    """SSM block with optional int4 on the conv kernel and/or SSM C matrix.

    Inherits everything from QuantSSMLayer; overrides only the matmul-time
    weight quantizer for whichever of (conv_weight, C) is flagged int4.
    Also widens the init range for those tensors so int4 levels are usable.
    """

    def __init__(self, n_embd, state_size, conv_kernel, dropout, use_gate, cfg,
                 int4_C: bool = False, int4_conv: bool = False) -> None:
        super().__init__(n_embd, state_size, conv_kernel, dropout, use_gate, cfg)
        self.int4_C = int4_C
        self.int4_conv = int4_conv
        # Re-initialise the int4 tensors with a wider range so the int4
        # round-trip uses more of the {-7..7} levels.
        if int4_conv:
            nn.init.uniform_(self.conv_weight, -7.0, 7.0)
            # Conv kernel std jumps from ~0.71 to ~4.6 (uniform on integers
            # in [-7,7]) — the layer's conv_shift was tuned for ternary, so
            # bump it by ~3 (log2(4.6/0.71) ≈ 2.7).
            with torch.no_grad():
                self.conv_shift.data = self.conv_shift.data + 3.0
        if int4_C:
            nn.init.uniform_(self.C, -7.0, 7.0)
            # Same correction for the C readout.
            with torch.no_grad():
                self.ssm_out_shift.data = self.ssm_out_shift.data + 3.0

    def _depthwise_conv(self, u: torch.Tensor) -> torch.Tensor:
        B, T, C = u.shape
        K = self.conv_kernel
        if self.int4_conv:
            w = int4_quantize(self.conv_weight)
        else:
            w = ternary_quantize(self.conv_weight)
        u_padded = F.pad(u, (0, 0, K - 1, 0))
        u_ch = u_padded.transpose(1, 2)
        kernel = w.unsqueeze(1)
        acc = F.conv1d(u_ch, kernel, groups=C).transpose(1, 2)
        return saturating_shift_int8(acc, self.conv_shift)

    def _ssm_recurrent(self, u: torch.Tensor, state):
        B, T, Cd = u.shape
        S = self.state_size

        decay_q = fake_quant_int8(torch.clamp(self.decay, 0.0, 127.0))
        B_t = ternary_quantize(self.B)
        if self.int4_C:
            C_t = int4_quantize(self.C)
        else:
            C_t = ternary_quantize(self.C)
        D_q = fake_quant_int8(self.D)

        if state is None:
            state = torch.zeros(B, Cd, S, device=u.device, dtype=u.dtype)

        outs = []
        for t in range(T):
            u_t = u[:, t, :]
            decayed = decay_q.unsqueeze(0) * state
            decayed_q = floor_div_pow2(decayed, 7)
            b_u = B_t.unsqueeze(0) * u_t.unsqueeze(-1)
            state = fake_quant_int8(decayed_q + b_u)
            c_state = (C_t.unsqueeze(0) * state).sum(dim=-1)
            c_part = saturating_shift_int8(c_state, self.ssm_out_shift)
            d_part = saturating_shift_int8(D_q.unsqueeze(0) * u_t, self.d_shift)
            y_t = fake_quant_int8(c_part + d_part)
            outs.append(y_t)
        y = torch.stack(outs, dim=1)
        return y, state


class Int4MoreBitNetLM(QuantBitNetLM):
    """E1b/E1c model: int4 head + int4 SSM C and/or conv."""

    def __init__(self, vocab_size: int, cfg: Config,
                 int4_C: bool, int4_conv: bool) -> None:
        super().__init__(vocab_size, cfg)
        # Replace blocks with Int4SSMLayer
        self.blocks = nn.ModuleList([
            Int4SSMLayer(cfg.n_embd, cfg.state_size, cfg.conv_kernel,
                         cfg.dropout, cfg.use_gate, cfg,
                         int4_C=int4_C, int4_conv=int4_conv)
            for _ in range(cfg.n_layer)
        ])
        # Replace head
        head_shift = cfg.init_shift_head if cfg.init_shift_head != 5.0 else 8.0
        self.head = Int4Head(cfg.n_embd, vocab_size, init_shift=head_shift)


def make_factory(int4_C: bool, int4_conv: bool):
    def factory(vocab_size, cfg):
        return Int4MoreBitNetLM(vocab_size, cfg, int4_C=int4_C, int4_conv=int4_conv)
    return factory


def run_cell(label: str, num_steps: int,
             int4_C: bool, int4_conv: bool) -> dict:
    target_bytes = 23_770
    n_embd = solve_n_embd_for_budget(target_bytes,
                                     int4_head=True, int4_ssm_C=int4_C, int4_conv=int4_conv)
    bytes_used = compute_total_bytes(n_embd,
                                     int4_head=True, int4_ssm_C=int4_C, int4_conv=int4_conv)
    print(f"\n=== {label}: n_embd={n_embd} ({bytes_used} / {target_bytes} bytes) ===")
    cfg = Config(use_pos_embed=False, n_embd=n_embd, num_steps=num_steps)
    _, _, losses = train_loop(cfg, model_factory=make_factory(int4_C, int4_conv))
    print(f"\n[{label}] final test_loss={losses['test']:.4f}  train={losses['train']:.4f}")
    return losses


def run(num_steps: int = 4000) -> dict[str, dict]:
    return {
        "E1b": run_cell("E1b: int4 head+C", num_steps, int4_C=True, int4_conv=False),
        "E1c": run_cell("E1c: int4 head+C+conv", num_steps, int4_C=True, int4_conv=True),
    }


if __name__ == "__main__":
    out = run(num_steps=4000)
    print("\n--- Summary ---")
    print(f"A0 baseline (4k steps): 1.8713")
    print(f"E1b (head+C int4):      {out['E1b']['test']:.4f}")
    print(f"E1c (head+C+conv int4): {out['E1c']['test']:.4f}")
