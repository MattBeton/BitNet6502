"""Parse a training log file and plot train/valid loss vs step.

Usage:
    .venv/bin/python -u modelling/experiments/plot_loss.py \
        modelling/experiments/tinystories_v1_log.txt \
        --out modelling/experiments/tinystories_v1_loss.png
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Match either of:
#   step      0 | lr 2.00e-06 | train 4.1537 | valid 4.1565 |   31.3s
#   step      0 | lr 6.67e-06 | train 4.8414 | test 4.8584 |   3.9s
LINE_RE = re.compile(
    r"step\s+(\d+)\s*\|\s*lr\s+([\d.eE+-]+)\s*\|\s*train\s+([\d.]+)\s*\|\s*"
    r"(?:valid|test)\s+([\d.]+)\s*\|"
)


def parse(log_path: Path):
    steps, trains, evals = [], [], []
    for line in log_path.read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        steps.append(int(m.group(1)))
        trains.append(float(m.group(3)))
        evals.append(float(m.group(4)))
    return steps, trains, evals


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log", type=Path)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--ymin", type=float, default=None)
    p.add_argument("--ymax", type=float, default=None)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    steps, trains, evals = parse(args.log)
    if not steps:
        print(f"no data found in {args.log}", file=sys.stderr)
        return 1

    print(f"parsed {len(steps)} eval points from {args.log}")
    print(f"  step range: {steps[0]} .. {steps[-1]}")
    print(f"  train: {trains[0]:.4f} -> {trains[-1]:.4f}  (min {min(trains):.4f})")
    print(f"  valid: {evals[0]:.4f} -> {evals[-1]:.4f}  (min {min(evals):.4f})")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, trains, label="train", marker="o", markersize=3, linewidth=1.5)
    ax.plot(steps, evals,  label="valid", marker="s", markersize=3, linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats)")
    title = args.title or f"loss curve: {args.log.stem}"
    ax.set_title(title)
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(bottom=args.ymin, top=args.ymax)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()

    out = args.out or args.log.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
