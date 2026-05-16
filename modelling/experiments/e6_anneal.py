"""E6 — quantization annealing on the ternary weights.

Hypothesis (from bitnet_quant_experiments.md):
    From step 0, the optimizer is steering through a loss surface
    heavily distorted by the ternary STE; the network has to find
    ternary minima before forming useful representations. Annealing
    the quantisation in (float warmup → linear ramp → pure ternary
    tail) might let the optimizer find a good float minimum first
    that the ternary projection only has to slightly displace.

Method:
    Replace bitnet_quant.ternary_quantize with an annealed version:
        forward = α * round_clip(w, ±1) + (1−α) * w
        backward = identity (STE on w)
    α schedule: 0 for the first 15% of steps, linear ramp to 1 by
    50%, then 1 (pure ternary) for the remaining 50%. All other
    quant constraints (int8 acts, shifts, biases) stay on throughout.

Implementation: monkey-patch `ternary_quantize` in `bitnet_quant`'s
namespace. This affects every module that imports it from there
(in_proj, out_proj, conv, B, C, head). The patch is reverted at the
end of run() so multiple experiments can share the interpreter.
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


# Module-level alpha — set by the on_step_start hook.
_ALPHA = {"value": 1.0}


def annealed_ternary(w: torch.Tensor) -> torch.Tensor:
    """Forward = α·round_clip(w, ±1) + (1−α)·w. Backward = identity-on-w."""
    if bitnet_quant._ABLATION["float_weights"]:
        return w
    q = torch.clamp(torch.round(w), -1.0, 1.0)
    alpha = _ALPHA["value"]
    if alpha >= 1.0:
        return w + (q - w).detach()
    forward = alpha * q + (1.0 - alpha) * w
    return w + (forward - w).detach()


def alpha_for_step(step: int, num_steps: int,
                   warmup_frac: float = 0.15, ramp_end_frac: float = 0.50) -> float:
    """0 → 0 (warmup) → linear ramp → 1 (tail)."""
    warmup = warmup_frac * num_steps
    ramp_end = ramp_end_frac * num_steps
    if step < warmup:
        return 0.0
    if step >= ramp_end:
        return 1.0
    return (step - warmup) / max(1.0, (ramp_end - warmup))


def make_on_step_start(num_steps: int, warmup_frac: float, ramp_end_frac: float):
    last_logged = {"alpha": -1.0}

    def on_step_start(step, model, opt):
        a = alpha_for_step(step, num_steps, warmup_frac, ramp_end_frac)
        _ALPHA["value"] = a
        # Log alpha changes at coarse boundaries
        if abs(a - last_logged["alpha"]) > 0.05 or a in (0.0, 1.0):
            if a != last_logged["alpha"]:
                if step % 200 == 0 or a in (0.0, 1.0):
                    print(f"  [E6] step {step}: alpha={a:.3f}", flush=True)
                last_logged["alpha"] = a

    return on_step_start


def run(num_steps: int = 4000,
        warmup_frac: float = 0.15, ramp_end_frac: float = 0.50) -> dict:
    print(f"\n=== E6: anneal ternary, warmup={warmup_frac}, ramp_end={ramp_end_frac}, steps={num_steps} ===")
    cfg = Config(use_pos_embed=False, n_embd=84, num_steps=num_steps)

    # Monkey-patch the quantizer in bitnet_quant's namespace.
    original = bitnet_quant.ternary_quantize
    bitnet_quant.ternary_quantize = annealed_ternary
    try:
        _, _, losses = train_loop(
            cfg,
            on_step_start=make_on_step_start(num_steps, warmup_frac, ramp_end_frac),
        )
    finally:
        bitnet_quant.ternary_quantize = original
        _ALPHA["value"] = 1.0
    print(f"\n[E6] final test_loss={losses['test']:.4f}  train={losses['train']:.4f}")
    return losses


if __name__ == "__main__":
    losses = run(num_steps=4000)
    print(f"\nA0 baseline (4k steps): 1.8713")
    print(f"E6 (this run):          {losses['test']:.4f}")
    print(f"Δ:                      {losses['test'] - 1.8713:+.4f} nats")
