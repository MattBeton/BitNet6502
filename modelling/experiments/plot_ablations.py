"""Overlay ablation cells from a folder of `abl_*_log.txt` files."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

LINE_RE = re.compile(
    r"step\s+(\d+)\s*\|\s*lr\s+([\d.eE+-]+)\s*\|\s*train\s+([\d.]+)\s*\|\s*"
    r"valid\s+([\d.]+)\s*\|\s*gn_avg\s+([\d.eE+-]+)\s+gn_max\s+([\d.eE+-]+)"
)


def parse(p: Path):
    s, t, v, gn = [], [], [], []
    for line in p.read_text().splitlines():
        m = LINE_RE.search(line)
        if not m:
            continue
        s.append(int(m.group(1)))
        t.append(float(m.group(3)))
        v.append(float(m.group(4)))
        gn.append(float(m.group(6)))
    return s, t, v, gn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs-glob", default="modelling/experiments/abl_*_log.txt")
    p.add_argument("--out-loss", default="modelling/experiments/ablations_loss.png")
    p.add_argument("--out-gn", default="modelling/experiments/ablations_gn.png")
    p.add_argument("--ymax", type=float, default=4.5)
    args = p.parse_args()

    paths = sorted(Path().glob(args.logs_glob))
    if not paths:
        print(f"no logs match {args.logs_glob}")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig1, ax1 = plt.subplots(figsize=(9, 5))
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    for path in paths:
        label = path.stem.removeprefix("abl_").removesuffix("_log")
        s, t, v, gn = parse(path)
        if not s:
            continue
        ax1.plot(s, v, label=f"{label} (final {v[-1]:.3f})", linewidth=1.5, marker="o", markersize=3)
        ax2.plot(s, gn, label=label, linewidth=1.5, marker="o", markersize=3)

    ax1.set_xlabel("step"); ax1.set_ylabel("valid loss (nats)")
    ax1.set_ylim(0.9, args.ymax)
    ax1.set_title("TinyStories ablations — valid loss")
    ax1.grid(True, alpha=0.3); ax1.legend(loc="upper right", fontsize=9)
    fig1.tight_layout(); fig1.savefig(args.out_loss, dpi=120)

    ax2.set_xlabel("step"); ax2.set_ylabel("max grad-norm in interval")
    ax2.set_yscale("log")
    ax2.set_title("TinyStories ablations — peak gradient norm (log)")
    ax2.grid(True, alpha=0.3, which="both"); ax2.legend(loc="upper left", fontsize=9)
    fig2.tight_layout(); fig2.savefig(args.out_gn, dpi=120)
    print(f"saved {args.out_loss} and {args.out_gn}")


if __name__ == "__main__":
    raise SystemExit(main())
