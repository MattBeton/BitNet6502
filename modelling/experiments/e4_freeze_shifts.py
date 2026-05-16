"""E4 — freeze integer shifts in late training.

Hypothesis (from bitnet_quant_experiments.md):
    Late-training loss oscillates by ~0.07 nats step-to-step because
    `round(shift_continuous)` flips between adjacent integers when the
    continuous proxy is near a half-integer boundary, causing a discrete
    2× change in output magnitude. Freezing the shifts after some
    fraction of training and only updating weights / biases / decay /
    gain afterwards should remove the oscillation source.

Method:
    Run the deployable architecture (n_embd=84, no pos embed) for 4,000
    steps, freezing every parameter whose name ends in "shift" once
    `step >= freeze_after_frac * num_steps`. Compare to A0 (1.8713).

The change is ONLY in the training loop — no new modules, no new flags
in bitnet_quant.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Config, train_loop, freeze_shift_params


FREEZE_AFTER_FRAC = 0.6


def make_on_step_start(num_steps: int, frac: float):
    freeze_step = int(num_steps * frac)
    state = {"frozen": False}

    def on_step_start(step, model, opt):
        if not state["frozen"] and step >= freeze_step:
            n = freeze_shift_params(model, freeze=True)
            print(f"  [E4] step {step}: froze {n} shift params", flush=True)
            state["frozen"] = True

    return on_step_start


def run(num_steps: int = 4000, frac: float = FREEZE_AFTER_FRAC) -> dict:
    cfg = Config(use_pos_embed=False, n_embd=84, num_steps=num_steps)
    print(f"\n=== E4: freeze shifts after frac={frac} ({int(num_steps*frac)} of {num_steps} steps) ===")
    _, _, losses = train_loop(cfg, on_step_start=make_on_step_start(num_steps, frac))
    print(f"\n[E4] final test_loss={losses['test']:.4f}  train={losses['train']:.4f}")
    return losses


if __name__ == "__main__":
    losses = run(num_steps=4000)
    # Compare to A0 = 1.8713 (from findings.md)
    print(f"\nA0 baseline (4k steps): 1.8713")
    print(f"E4 (this run):          {losses['test']:.4f}")
    print(f"Δ:                      {losses['test'] - 1.8713:+.4f} nats")
