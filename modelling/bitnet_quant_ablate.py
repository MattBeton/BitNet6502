"""Ablation runner — attribute the gap between the quantized BitNet and the
unquantized state-space baseline (1.46 from state-space-findings.md).

Each row turns off one or more quantization constraints while keeping the
architecture, param count, and training schedule fixed. The deltas tell us
where the loss is actually coming from.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: F401
from bitnet_quant import Config, QuantBitNetLM, train, estimate_loss, resolve_device
from shakespeare import build_datasets
from torch.utils.data import DataLoader


# Each row: (label, overrides). Overrides are merged into Config.
# Steps and base shape match the deployable model; only the constraints flip.
BASE = dict(use_pos_embed=False, n_embd=84, num_steps=4000)

ABLATIONS: list[tuple[str, dict]] = [
    ("A0_full_quant",         dict()),                                  # all constraints on (deployable)
    ("A1_float_weights",      dict(ablate_float_weights=True)),         # ternary → float
    ("A2_float_acts",         dict(ablate_float_acts=True)),            # int8 acts/state → float
    ("A3_no_saturate",        dict(ablate_no_saturate=True)),           # keep shift, drop ±127 clip
    ("A4_no_shift_no_sat",    dict(ablate_no_shift=True)),              # raw matmul output
    ("A5_all_float",          dict(ablate_float_weights=True,
                                   ablate_float_acts=True,
                                   ablate_no_saturate=True)),           # float reference
]


def run_one(label: str, overrides: dict) -> float:
    cfg = Config(**{**BASE, **overrides})
    print(f"\n{'='*60}\n  {label}  overrides={overrides}\n{'='*60}", flush=True)
    t0 = time.time()
    model, _ = train(cfg)
    device = resolve_device(cfg.device)
    train_ds, test_ds, _ = build_datasets(block_size=cfg.block_size, train_fraction=cfg.train_fraction)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    losses = estimate_loss(model, train_loader, test_loader, cfg, device)
    elapsed = time.time() - t0
    print(f"[done] {label}: test_loss={losses['test']:.4f} train={losses['train']:.4f} ({elapsed:.0f}s)", flush=True)
    return losses["test"]


def main() -> None:
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        runs = [(l, o) for l, o in ABLATIONS if l in wanted]
    else:
        runs = ABLATIONS
    results = []
    for label, overrides in runs:
        try:
            test = run_one(label, overrides)
        except Exception as exc:
            print(f"[error] {label}: {exc}", flush=True)
            test = float("inf")
        results.append((label, test))
    print("\n=== ablation summary (lower test loss = constraint that hurts most) ===")
    for label, loss in results:
        print(f"  {label:25s}  test_loss={loss:.4f}")


if __name__ == "__main__":
    main()
