"""Re-run E5a alone with the frozen-head-shift fix."""
from __future__ import annotations
import sys, time, csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from e5_tied_embeddings import run_e5a_only

t0 = time.time()
losses = run_e5a_only(num_steps=4000)
elapsed = time.time() - t0

results = HERE / "results.csv"
with results.open("a", newline="") as f:
    w = csv.writer(f)
    w.writerow(["E5", "E5a_rerun_frozen_shift",
                f"{losses['test']:.4f}", f"{losses['train']:.4f}",
                f"{elapsed:.0f}"])
print(f"\nE5a rerun: test={losses['test']:.4f} train={losses['train']:.4f} ({elapsed:.0f}s)")
