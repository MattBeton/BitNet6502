"""Screening harness for bitnet_quant configs.

Each entry is (label, config-overrides). Trains for `screen_steps`,
prints final eval loss, and writes a summary line per run.
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch  # noqa: F401  (ensure torch is importable before bitnet_quant)
from bitnet_quant import Config, QuantBitNetLM, train


SCREEN_STEPS = 2500


# (label, overrides). Each runs in isolation.
EXPERIMENTS: list[tuple[str, dict]] = [
    ("wider-84x3-s8",     dict(n_embd=84)),
    ("no-pos-embed-84",   dict(use_pos_embed=False, n_embd=84)),
    ("lower-lr",          dict(learning_rate=1e-3)),
]


def count_params(cfg: Config) -> tuple[int, int]:
    m = QuantBitNetLM(27, cfg)
    total = sum(p.numel() for p in m.parameters())
    ternary = m.ternary_param_count()
    return total, ternary


def run_one(label: str, overrides: dict) -> tuple[str, float, float]:
    cfg = Config(num_steps=SCREEN_STEPS, eval_interval=SCREEN_STEPS // 5,
                 warmup_steps=min(150, SCREEN_STEPS // 10),
                 **overrides)
    total, ternary = count_params(cfg)
    if total >= 80_000:
        print(f"[skip] {label}: {total:,} params over budget")
        return label, total, float("inf")
    print(f"\n{'='*60}\n  {label}  total={total:,} ternary={ternary:,}\n{'='*60}", flush=True)
    t0 = time.time()
    model, _ = train(cfg)
    elapsed = time.time() - t0
    # Final eval loss is reported on the last step by train(); re-run a quick eval here
    from bitnet_quant import estimate_loss, resolve_device
    from shakespeare import build_datasets
    from torch.utils.data import DataLoader
    device = resolve_device(cfg.device)
    train_ds, test_ds, _ = build_datasets(block_size=cfg.block_size, train_fraction=cfg.train_fraction)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True)
    losses = estimate_loss(model, train_loader, test_loader, cfg, device)
    print(f"[done] {label}: test_loss={losses['test']:.4f} in {elapsed:.0f}s", flush=True)
    return label, total, losses["test"]


def main() -> None:
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        runs = [(l, o) for l, o in EXPERIMENTS if l in wanted]
    else:
        runs = EXPERIMENTS
    results = []
    for label, overrides in runs:
        try:
            results.append(run_one(label, overrides))
        except Exception as exc:
            print(f"[error] {label}: {exc}", flush=True)
            results.append((label, 0, float("inf")))
    print("\n=== summary ===")
    for label, total, loss in sorted(results, key=lambda r: r[2]):
        print(f"  {label:30s}  params={total:6,d}  test_loss={loss:.4f}")


if __name__ == "__main__":
    main()
