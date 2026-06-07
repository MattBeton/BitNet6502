"""Quantisation primitives — fake-quants whose forward equals the integer
operation exactly and whose backward passes a smooth gradient (STE).

Every intermediate value in the BitNet stack is fake-quantised during training
to match the integer arithmetic that runs on the 6502. With no ablation
toggles in production, these functions are the entire integer-faithful runtime.

Datatype contract (mirrors C inference):
  - Activations / residual stream: int8  ([-128, 127])
  - Matmul / accumulator scratch:  int16 ([-32768, 32767])
  - Ternary weights:               {-1, 0, +1}
  - int4 weights:                  [-7, +7]
  - SSM decay:                     int8 in [0, 127]; effective decay = a / 128
"""
from __future__ import annotations

import torch


# --- Weight quantisers --------------------------------------------------------

def ternary_quantize(w: torch.Tensor) -> torch.Tensor:
    """Round to {-1, 0, +1} with straight-through estimator on the backward pass."""
    q = torch.clamp(torch.round(w), -1.0, 1.0)
    return w + (q - w).detach()


def int4_quantize(w: torch.Tensor) -> torch.Tensor:
    """Round to integers in [-7, +7] with straight-through estimator."""
    q = torch.clamp(torch.round(w), -7.0, 7.0)
    return w + (q - w).detach()


# --- Activation quantisers ----------------------------------------------------

def fake_quant_int8(x: torch.Tensor) -> torch.Tensor:
    """Saturating round to int8 range — matches `(signed char)clamp(x, -128, 127)`."""
    q = torch.clamp(torch.round(x), -128.0, 127.0)
    return x + (q - x).detach()


def fake_quant_int16(x: torch.Tensor) -> torch.Tensor:
    """Saturating round to int16 range — matches `(signed int)clamp(x, INT16_MIN, INT16_MAX)`."""
    q = torch.clamp(torch.round(x), -32768.0, 32767.0)
    return x + (q - x).detach()


# --- Shift operations ---------------------------------------------------------

def shift_round(s: torch.Tensor, max_shift: int = 14) -> torch.Tensor:
    """Round a learned shift parameter to an integer in [0, max_shift].

    The shift is encoded as a continuous float for AdamW; this STE makes the
    actual shift integer-valued so the forward path is bit-exact to a 6502
    `acc >> shift`.
    """
    q = torch.clamp(torch.round(s), 0.0, float(max_shift))
    return s + (q - s).detach()


def saturating_shift_int8(acc: torch.Tensor, shift_param: torch.Tensor) -> torch.Tensor:
    """Forward: `clamp(floor(acc / 2^round(shift)), -128, 127)`.

    Backward: gradient flows through smooth `acc / 2^shift_continuous` so both
    `acc` and `shift` receive informative gradients. Used wherever an int16
    accumulator needs to come back into the int8 range.
    """
    s_int = shift_round(shift_param)
    div = torch.pow(2.0, s_int)
    smooth = acc / div
    hard = torch.clamp(torch.floor(acc / div), -128.0, 127.0)
    return smooth + (hard - smooth).detach()


def learned_shift_no_sat(acc: torch.Tensor, shift_param: torch.Tensor) -> torch.Tensor:
    """Same as `saturating_shift_int8` but without the int8 clamp.

    Used for the head: argmax is invariant to a positive scale, and we want
    the full int16 dynamic range so that top-k masking with INT_MIN works.
    """
    s_int = shift_round(shift_param)
    div = torch.pow(2.0, s_int)
    smooth = acc / div
    hard = torch.floor(acc / div)
    return smooth + (hard - smooth).detach()


def floor_div_pow2(x: torch.Tensor, shift_const: int) -> torch.Tensor:
    """Forward: `floor(x / 2^shift)` for a compile-time-constant shift.

    Used inside the SSM for the >>7 that brings `decay * state` back into int8
    range after the int8 * int8 multiplication.
    """
    div = float(1 << shift_const)
    smooth = x / div
    hard = torch.floor(x / div)
    return smooth + (hard - smooth).detach()
