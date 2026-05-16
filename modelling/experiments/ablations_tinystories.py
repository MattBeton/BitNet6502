"""Ablation grid for TinyStories — find what causes v1's divergence.

Cells (each 5,000 steps, ~22 min on MPS):
    A. anneal off, wd=0.04, lr=2e-3   ← rerun of v2 baseline at fixed length
    B. anneal on,  wd=0.04, lr=2e-3   ← does anneal alone diverge at original wd?
    C. anneal off, wd=0.01, lr=2e-3   ← does low wd alone diverge without anneal?
    D. anneal on,  wd=0.01, lr=2e-3   ← v1's diverging config — sanity-check
    E. anneal on,  wd=0.04, lr=1e-3   ← does lower lr stabilise anneal?

Train + valid loss + gradient norms are logged per eval; an after-the-fact
plot superimposes all cells so the divergence pattern is easy to read.
"""
from __future__ import annotations

import csv
import importlib
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


CELLS: list[tuple[str, dict]] = [
    ("A_no-anneal_wd0.04_lr2e-3", dict(num_steps=5000, learning_rate=2e-3, weight_decay=0.04, anneal=False)),
    ("B_anneal_wd0.04_lr2e-3",    dict(num_steps=5000, learning_rate=2e-3, weight_decay=0.04, anneal=True)),
    ("C_no-anneal_wd0.01_lr2e-3", dict(num_steps=5000, learning_rate=2e-3, weight_decay=0.01, anneal=False)),
    ("D_anneal_wd0.01_lr2e-3",    dict(num_steps=5000, learning_rate=2e-3, weight_decay=0.01, anneal=True)),
    ("E_anneal_wd0.04_lr1e-3",    dict(num_steps=5000, learning_rate=1e-3, weight_decay=0.04, anneal=True)),
]

RESULTS_CSV = HERE / "ablations_tinystories.csv"


def append_result(label, kwargs, losses, elapsed):
    new = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["cell", "num_steps", "anneal", "wd", "lr",
                        "train_loss", "valid_loss",
                        "gn_avg", "gn_max", "elapsed_sec"])
        w.writerow([
            label, kwargs["num_steps"], kwargs["anneal"],
            kwargs["weight_decay"], kwargs["learning_rate"],
            f"{losses.get('train', float('nan')):.4f}",
            f"{losses.get('valid', float('nan')):.4f}",
            f"{losses.get('gn_avg', float('nan')):.2f}",
            f"{losses.get('gn_max', float('nan')):.2f}",
            f"{elapsed:.0f}",
        ])


def run_cell(label, kwargs):
    log_path = HERE / f"abl_{label}_log.txt"
    print(f"\n{'#'*70}\n# {label}: {kwargs}\n{'#'*70}", flush=True)
    t0 = time.time()

    class Tee:
        def __init__(self, *streams): self.streams = streams
        def write(self, s):
            for s_ in self.streams: s_.write(s)
        def flush(self):
            for s_ in self.streams: s_.flush()
    f = open(log_path, "w")
    saved = sys.stdout
    sys.stdout = Tee(saved, f)
    try:
        # Reload to reset any monkey-patched globals between runs.
        if "tinystories_run" in sys.modules:
            mod = importlib.reload(sys.modules["tinystories_run"])
        else:
            mod = importlib.import_module("tinystories_run")
        losses = mod.run(verbose=True, save_path=None, **kwargs)
        elapsed = time.time() - t0
        append_result(label, kwargs, losses, elapsed)
        print(f"  [done] {label} in {elapsed:.0f}s  valid={losses['valid']:.4f}", flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout = saved
        f.close()


def main():
    cells = CELLS
    if len(sys.argv) > 1:
        wanted = set(sys.argv[1:])
        cells = [c for c in cells if c[0] in wanted]
    for label, kwargs in cells:
        run_cell(label, kwargs)
    print("\nablation summary in", RESULTS_CSV)


if __name__ == "__main__":
    main()
