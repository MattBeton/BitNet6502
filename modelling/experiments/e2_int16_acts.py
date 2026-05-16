"""E2 — int16 activation residual stream.

Hypothesis (from bitnet_quant_experiments.md):
    8-bit activation precision costs 0.08 nats. With int16 acts and
    int32 accumulators, both rounding-grain noise and saturation
    collapse to near zero, recovering essentially the entire
    activation-side gap. Note: this is a RAM trade, not a ROM trade —
    the 23,770-byte weight blob is unchanged.

Method:
    Monkey-patch `fake_quant_int8` and `saturating_shift_int8` in
    `bitnet_quant`'s namespace to use int16 ranges (±32,767). Both
    functions still round to integer (so the rounding-grain stays the
    same) but the saturation now never triggers in practice. This
    cleanly isolates the saturation cost without touching weights or
    the model's structure.

Note:
    The model continues to use the same per-layer shifts as A0. With
    saturation removed, the shifts may need to retrain — they have
    8 bits of budget by default but the ceiling is now effectively
    16 bits, so the optimizer is free to use larger output magnitudes
    if helpful. This is left to the optimizer to discover (no init
    change), to make the comparison clean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Config, train_loop

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import bitnet_quant  # noqa: E402


def fake_quant_int16(x: torch.Tensor) -> torch.Tensor:
    if bitnet_quant._ABLATION["float_acts"]:
        return x
    q = torch.clamp(torch.round(x), -32768.0, 32767.0)
    return x + (q - x).detach()


def saturating_shift_int16(acc: torch.Tensor, shift_param: torch.Tensor) -> torch.Tensor:
    if bitnet_quant._ABLATION["no_shift"]:
        return acc
    s_int = bitnet_quant.shift_round(shift_param)
    div = torch.pow(2.0, s_int)
    if bitnet_quant._ABLATION["no_saturate"]:
        smooth = acc / div
        hard = torch.floor(acc / div)
        return smooth + (hard - smooth).detach()
    smooth = acc / div
    hard = torch.clamp(torch.floor(acc / div), -32768.0, 32767.0)
    return smooth + (hard - smooth).detach()


def run(num_steps: int = 4000) -> dict:
    print(f"\n=== E2: int16 activations, {num_steps} steps ===")
    cfg = Config(use_pos_embed=False, n_embd=84, num_steps=num_steps)

    # Monkey-patch the activation quantizers in bitnet_quant's namespace.
    orig_q8 = bitnet_quant.fake_quant_int8
    orig_ss8 = bitnet_quant.saturating_shift_int8
    bitnet_quant.fake_quant_int8 = fake_quant_int16
    bitnet_quant.saturating_shift_int8 = saturating_shift_int16
    try:
        _, _, losses = train_loop(cfg)
    finally:
        bitnet_quant.fake_quant_int8 = orig_q8
        bitnet_quant.saturating_shift_int8 = orig_ss8

    print(f"\n[E2] final test_loss={losses['test']:.4f}  train={losses['train']:.4f}")
    return losses


if __name__ == "__main__":
    losses = run(num_steps=4000)
    print(f"\nA0 baseline (4k steps): 1.8713")
    print(f"A3 no saturation:       1.8184")
    print(f"E2 (this run):          {losses['test']:.4f}")
    print(f"Δ vs A0:                {losses['test'] - 1.8713:+.4f} nats")
