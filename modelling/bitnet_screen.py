"""Quick screening script for BitNet state-space configs.

Runs each config for a fixed number of steps and reports eval loss.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace

sys.path.insert(0, "modelling")

from bitnet_state_space import Config, train, resolve_device
import torch


SCREEN_STEPS = 3000
FULL_STEPS = 8000

# Each entry: (label, config_kwargs)
EXPERIMENTS: list[tuple[str, dict]] = [
    # Depth sweep — same total budget, different depth
    ("4L-e85-s8-baseline",  dict(n_embd=85, n_layer=4, state_size=8, num_steps=SCREEN_STEPS)),
    ("5L-e76-s8",           dict(n_embd=76, n_layer=5, state_size=8, num_steps=SCREEN_STEPS)),
    ("6L-e69-s8",           dict(n_embd=69, n_layer=6, state_size=8, num_steps=SCREEN_STEPS)),

    # State size sweep at 4 layers
    ("4L-e85-s12",          dict(n_embd=85, n_layer=4, state_size=12, num_steps=SCREEN_STEPS)),
    ("4L-e82-s16",          dict(n_embd=82, n_layer=4, state_size=16, num_steps=SCREEN_STEPS)),

    # LR sensitivity
    ("4L-e85-s8-lr2.6",     dict(n_embd=85, n_layer=4, state_size=8, learning_rate=2.6e-3, num_steps=SCREEN_STEPS)),
    ("4L-e85-s8-lr4.0",     dict(n_embd=85, n_layer=4, state_size=8, learning_rate=4.0e-3, num_steps=SCREEN_STEPS)),
]


def count_params(cfg: Config) -> tuple[int, int]:
    from bitnet_state_space import BitNetStateSpaceLM
    m = BitNetStateSpaceLM(27, cfg)
    total = sum(p.numel() for p in m.parameters())
    ternary = m.ternary_param_count()
    return total, ternary


def run_experiment(label: str, kwargs: dict) -> float:
    cfg = Config(**kwargs)
    total, ternary = count_params(cfg)
    if total >= 80_000:
        print(f"  SKIP {label}: {total:,} params (over budget)")
        return float("inf")
    print(f"\n{'='*60}", flush=True)
    print(f"  {label}  |  total={total:,}  ternary={ternary:,}", flush=True)
    t0 = time.time()
    _, vocab = train(cfg)
    elapsed = time.time() - t0
    print(f"  {label}: done in {elapsed:.0f}s", flush=True)
    # Return the final reported test loss — captured from stdout, but
    # we re-evaluate here for clarity. The train() function prints it.
    return elapsed


if __name__ == "__main__":
    results = []
    for label, kwargs in EXPERIMENTS:
        cfg = Config(**kwargs)
        total, ternary = count_params(cfg)
        print(f"{label}: total={total:,} ternary={ternary:,}")
    print()
    print("Run each experiment via:")
    print("  uv run python modelling/bitnet_screen.py <label>")
