"""E1a — int4 weight on the head matrix only.

Hypothesis (from bitnet_quant_experiments.md):
    Of the 0.21 nat ternary cost, some is plausibly concentrated in the
    'decision-point' tensors with small fan-in: head (27 classes from
    84 channels), SSM C, conv. int4 ({-7..+7}) doubles their byte
    footprint but quadruples representational range. E1a is the
    cheapest cell — head only — and bounds the leverage from below.

Method:
    Replace QuantHead's ternary weight with an int4 weight (clamp to
    {-7,+7}). To keep the total weight blob at 23,770 bytes we shrink
    n_embd to the largest value that fits the new budget — int4 head
    costs +567 bytes vs ternary head, so n_embd drops from 84 → 82
    (saves ternary bytes elsewhere).

The change is local: an int4_quantize STE and an Int4Head module
defined in this file. Nothing in bitnet_quant.py changes.
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

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bitnet_quant import QuantBitNetLM, _ABLATION, learned_shift_no_sat  # noqa: E402


# -----------------------------------------------------------------------------
# int4 weight quantizer (forward = round-to-int in [-7, 7]; backward = STE)
# -----------------------------------------------------------------------------

def int4_quantize(w: torch.Tensor) -> torch.Tensor:
    if _ABLATION["float_weights"]:
        return w
    q = torch.clamp(torch.round(w), -7.0, 7.0)
    return w + (q - w).detach()


class Int4Head(nn.Module):
    """Head with int4 weight: int8 input · int4 weight → int16 logits.

    Weight init range is widened to U[-7, 7] so that ~all int4 levels
    survive the round at init (vs ternary's U[-1, 1] which gives ~50%
    nonzero ±1).
    """

    def __init__(self, in_f: int, vocab_size: int, init_shift: float = 8.0) -> None:
        super().__init__()
        # int4 weights have std ~9.5× larger than ternary U[-1,1] rounded,
        # so the pre-shift accumulator is ~9.5× larger; we add log2(9.5)≈3.2
        # to the original ternary head shift (5) to compensate.
        self.weight = nn.Parameter(torch.empty(vocab_size, in_f))
        self.shift = nn.Parameter(torch.tensor(init_shift))
        nn.init.uniform_(self.weight, -7.0, 7.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = F.linear(x, int4_quantize(self.weight))
        return learned_shift_no_sat(acc, self.shift)

    # Keep the same parameter-counting contract; head is no longer ternary
    # but the count function on QuantBitNetLM uses it as a slot.
    def ternary_count(self) -> int:
        return 0


class Int4HeadBitNetLM(QuantBitNetLM):
    """Drop-in: same architecture, head's ternary weight replaced with int4."""

    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__(vocab_size, cfg)
        # init_shift_head defaults to 5.0 (tuned for ternary). Int4Head's own
        # default (8.0) absorbs the int4 magnitude jump; only override if cfg
        # explicitly raised init_shift_head from its ternary default.
        head_shift = cfg.init_shift_head if cfg.init_shift_head != 5.0 else 8.0
        self.head = Int4Head(cfg.n_embd, vocab_size, init_shift=head_shift)

    def ternary_param_count(self) -> int:
        n = 0
        for b in self.blocks:
            n += b.in_proj.weight.numel()
            n += b.out_proj.weight.numel()
            n += b.conv_weight.numel()
            n += b.B.numel() + b.C.numel()
        return n  # head no longer counted


def run(num_steps: int = 4000) -> dict:
    target_bytes = 23_770
    n_embd = solve_n_embd_for_budget(target_bytes, int4_head=True)
    bytes_used = compute_total_bytes(n_embd, int4_head=True)
    print(f"\n=== E1a: int4 head, n_embd={n_embd} ({bytes_used} bytes / {target_bytes} budget) ===")

    cfg = Config(use_pos_embed=False, n_embd=n_embd, num_steps=num_steps)
    _, _, losses = train_loop(cfg, model_factory=Int4HeadBitNetLM)
    print(f"\n[E1a] final test_loss={losses['test']:.4f}  train={losses['train']:.4f}")
    return losses


if __name__ == "__main__":
    losses = run(num_steps=4000)
    print(f"\nA0 baseline (4k steps): 1.8713")
    print(f"E1a (this run):         {losses['test']:.4f}")
    print(f"Δ:                      {losses['test'] - 1.8713:+.4f} nats")
