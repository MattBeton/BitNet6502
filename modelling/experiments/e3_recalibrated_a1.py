"""E3 — recalibrated A1 (methodology cleanup).

Hypothesis (from bitnet_quant_experiments.md):
    A1 in the original ablation hit 2.81 (worse than A0 at 1.87) — a
    calibration artefact, not signal. The shift inits and weight init
    U[-1, 1] are tuned so ternary rounding gives ~50% nonzero ±1
    weights. With float weights and the same init, magnitudes stay
    well below 1, the int16 accumulator output is much smaller than
    the shift expects, layers emit near-zero activations, and training
    stalls.

    Recalibrating: float-W with U[-1,1] has std ~0.577 vs ternary's
    ~0.707. So float-W's matmul std is 0.577/0.707 ≈ 0.82× of ternary.
    To match A0's pre-shift magnitude, reduce all the per-layer shifts
    by log2(1/0.82) ≈ 0.3 bits (negligible) — OR widen the float weight
    init range so the std matches ternary. We'll widen: U[-1.225, 1.225]
    has std = 1.225/sqrt(3) = 0.707, matching ternary.

    Expected: A1 lands near A2's 1.79.

Method:
    Run with `ablate_float_weights=True` plus a custom weight-init
    range. We monkey-patch the modules' init logic by re-initialising
    weights after model construction, then call train.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Config, train_loop

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bitnet_quant import QuantBitNetLM  # noqa: E402


# Width that makes U[-w, w] have the same std as ternary U[-1,1] rounded
# (which is ~0.707): w = sqrt(3) * 0.707 ≈ 1.225.
RECAL_WIDTH = 1.2247448713915890


class RecalibratedBitNetLM(QuantBitNetLM):
    """Same architecture as A1, but every ternary-weight tensor is
    re-initialised with the wider U[-1.225, 1.225] range so the
    pre-shift accumulator variance matches the ternary baseline.
    """

    def __init__(self, vocab_size: int, cfg: Config) -> None:
        super().__init__(vocab_size, cfg)
        with torch.no_grad():
            for blk in self.blocks:
                nn.init.uniform_(blk.in_proj.weight, -RECAL_WIDTH, RECAL_WIDTH)
                nn.init.uniform_(blk.out_proj.weight, -RECAL_WIDTH, RECAL_WIDTH)
                nn.init.uniform_(blk.conv_weight, -RECAL_WIDTH, RECAL_WIDTH)
                nn.init.uniform_(blk.B, -RECAL_WIDTH, RECAL_WIDTH)
                nn.init.uniform_(blk.C, -RECAL_WIDTH, RECAL_WIDTH)
            nn.init.uniform_(self.head.weight, -RECAL_WIDTH, RECAL_WIDTH)


def run(num_steps: int = 4000) -> dict:
    print(f"\n=== E3: recalibrated A1 (float weights, std-matched init), {num_steps} steps ===")
    cfg = Config(use_pos_embed=False, n_embd=84, num_steps=num_steps,
                 ablate_float_weights=True)
    _, _, losses = train_loop(cfg, model_factory=RecalibratedBitNetLM)
    print(f"\n[E3] final test_loss={losses['test']:.4f}  train={losses['train']:.4f}")
    return losses


if __name__ == "__main__":
    losses = run(num_steps=4000)
    print(f"\nA0 baseline (full quant, 4k):    1.8713")
    print(f"A1 original (uncalibrated, 4k):  2.8108  (calibration artefact)")
    print(f"A2 float acts (ternary W, 4k):   1.7941")
    print(f"E3 recalibrated A1 (this run):   {losses['test']:.4f}")
    print(f"  Δ vs A2: {losses['test'] - 1.7941:+.4f} nats  (close to 0 = ternary cost is real)")
