"""Run all remaining experiments back-to-back and log to per-experiment files.

Independent experiments are run sequentially (single-GPU constraint). Each
writes its own console output to modelling/experiments/<name>_log.txt and
appends a result row to modelling/experiments/results.csv.
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


# (label, module, callable, kwargs, log_file). callable returns either a
# dict {"test": ..., "train": ...} or a dict-of-dicts (E5 / E1bc).
EXPERIMENTS = [
    ("E1a", "e1a_int4_head",        "run", {"num_steps": 4000}),
    ("E5",  "e5_tied_embeddings",   "run", {"num_steps": 4000}),
    ("E6",  "e6_anneal",            "run", {"num_steps": 4000}),
    ("E1bc","e1bc_int4_more",       "run", {"num_steps": 4000}),
    ("E3",  "e3_recalibrated_a1",   "run", {"num_steps": 4000}),
    ("E2",  "e2_int16_acts",        "run", {"num_steps": 4000}),
]

RESULTS_CSV = HERE / "results.csv"


def append_result(label: str, sub_label: str, test_loss: float, train_loss: float, elapsed: float):
    new_file = not RESULTS_CSV.exists()
    with RESULTS_CSV.open("a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["label", "sub_label", "test_loss", "train_loss", "elapsed_sec"])
        w.writerow([label, sub_label, f"{test_loss:.4f}", f"{train_loss:.4f}", f"{elapsed:.0f}"])


def run_one(label: str, module_name: str, fn_name: str, kwargs: dict):
    log_path = HERE / f"{module_name}_log.txt"
    print(f"\n{'#'*70}\n# {label}: {module_name}.{fn_name}({kwargs})\n{'#'*70}", flush=True)
    t0 = time.time()
    # Tee stdout to a per-experiment log file
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
        mod = importlib.import_module(module_name)
        # Some modules import bitnet_quant indirectly; reload to reset any
        # monkey-patched globals between runs.
        importlib.reload(mod)
        fn = getattr(mod, fn_name)
        out = fn(**kwargs)
        elapsed = time.time() - t0
        if isinstance(out, dict) and "test" in out and isinstance(out["test"], (int, float)):
            append_result(label, "", out["test"], out["train"], elapsed)
        elif isinstance(out, dict):
            for k, v in out.items():
                append_result(label, k, v["test"], v["train"], elapsed)
        else:
            print(f"  [warn] unexpected return shape: {type(out)}")
        print(f"  [done] {label} in {elapsed:.0f}s", flush=True)
    except Exception:
        traceback.print_exc()
    finally:
        sys.stdout = saved
        f.close()


def main():
    if RESULTS_CSV.exists():
        # Keep prior runs (e.g. E4 already in there); appended results stack.
        pass
    for label, mod, fn, kwargs in EXPERIMENTS:
        run_one(label, mod, fn, kwargs)
    print("\nAll experiments done. See results.csv and *_log.txt files.")


if __name__ == "__main__":
    main()
