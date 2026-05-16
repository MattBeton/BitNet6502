"""Overlay curriculum vs control training curves (high-fidelity train EMA + sparser valid)."""
from __future__ import annotations

import csv
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def read_steps(path: Path):
    steps, ema, gn, bs = [], [], [], []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            ema.append(float(row["train_ema"]))
            gn.append(float(row["gn"]))
            bs.append(int(row["block_size"]))
    return steps, ema, gn, bs


def read_eval(path: Path):
    steps, valid = [], []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            valid.append(float(row["valid_loss"]))
    return steps, valid


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cells = [
        ("C1 control (bs=128)",       "curriculum_C1_control_steps.csv",    "curriculum_C1_control_eval.csv",    "tab:blue"),
        ("C2 curriculum (bs=32→128)", "curriculum_C2_curriculum_steps.csv", "curriculum_C2_curriculum_eval.csv", "tab:orange"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax_loss, ax_gn = axes

    for label, steps_csv, eval_csv, color in cells:
        s, ema, gn, bs = read_steps(HERE / steps_csv)
        es, ev = read_eval(HERE / eval_csv)
        ax_loss.plot(s, ema, color=color, alpha=0.85, linewidth=1.0,
                     label=f"{label}  train EMA")
        ax_loss.plot(es, ev, color=color, alpha=1.0, linewidth=2.0,
                     marker="o", markersize=4, linestyle="--",
                     label=f"{label}  valid (final {ev[-1]:.4f})")
        ax_gn.plot(s, gn, color=color, alpha=0.6, linewidth=0.8, label=label)

        # Mark phase change for curriculum
        if "curriculum" in label.lower():
            for i in range(1, len(bs)):
                if bs[i] != bs[i - 1]:
                    ax_loss.axvline(s[i], color=color, linestyle=":", linewidth=1, alpha=0.7)
                    ax_loss.text(s[i], ax_loss.get_ylim()[1] * 0.95,
                                 f"  bs {bs[i-1]} → {bs[i]}",
                                 color=color, fontsize=9, alpha=0.85)
                    break

    ax_loss.set_ylabel("loss (nats)")
    ax_loss.set_title("TinyStories curriculum experiment — control vs short→long")
    ax_loss.set_ylim(0.85, 4.5)
    ax_loss.grid(True, alpha=0.3)
    ax_loss.legend(loc="upper right", fontsize=9)

    ax_gn.set_xlabel("step")
    ax_gn.set_ylabel("grad norm (per step, log)")
    ax_gn.set_yscale("log")
    ax_gn.grid(True, alpha=0.3, which="both")
    ax_gn.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    out = HERE / "curriculum_curves.png"
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    sys.exit(main())
