"""Curriculum experiment: train on short sentences first, then extend.

Two cells, each at the winning TinyStories config (no anneal, wd=0.04,
freeze shifts at 50%, int4 head/SSM-C/conv):

    C1 control:    block_size=128 for all `total_steps` (~3 sentences/window).
    C2 curriculum: block_size= 32 for the first half (~1 sentence/window),
                   then block_size=128 for the second half.

Eval is fixed at block_size=128 in both cells so the curves measure the
same thing — long-context loss — regardless of training-time window.

In addition to the per-eval valid loss, this run logs **per-step train
loss** as an exponentially-weighted moving average (alpha=0.05, ~20-step
window) so we get a smooth, high-fidelity training curve to compare
against the ~80-point eval curve.

Outputs:
    modelling/experiments/curriculum_C1_steps.csv  (per-step EMA train)
    modelling/experiments/curriculum_C1_eval.csv   (per-eval valid)
    modelling/experiments/curriculum_C2_steps.csv
    modelling/experiments/curriculum_C2_eval.csv
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import Config, freeze_shift_params  # noqa: E402
from final_run import StackedBitNetLM, make_factory  # noqa: E402
import bitnet_quant
from bitnet_quant import lr_for_step, resolve_device, set_ablation_from_cfg  # noqa: E402

import tinystories  # noqa: E402
from tinystories_run import VOCAB_PATH, eval_loss  # noqa: E402


def make_loaders(cfg: Config):
    train_ds, valid_ds, vocab = tinystories.build_datasets(
        block_size=cfg.block_size,
        vocab_path=VOCAB_PATH,
    )
    g = torch.Generator(); g.manual_seed(1337)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                   drop_last=True, generator=g),
        DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True),
        vocab,
    )


def run_phased(
    cell_label: str,
    phases: list[tuple[int, int]],   # [(block_size, steps), ...]
    *,
    eval_block_size: int = 128,
    eval_interval: int = 200,
    eval_n_batches: int = 40,
    train_ema_alpha: float = 0.05,
    freeze_shift_after_step: int | None = None,
    int4_head: bool = True, int4_ssm_C: bool = True, int4_conv: bool = True,
    learning_rate: float = 2e-3,
    weight_decay: float = 0.04,
    n_embd: int = 81,
    save_path: str | None = None,
    csv_dir: Path = HERE,
):
    total_steps = sum(s for _, s in phases)
    print(f"\n{'='*70}")
    print(f"  {cell_label}: phases={phases}  total_steps={total_steps}")
    print(f"  eval_block_size={eval_block_size}  eval_interval={eval_interval}")
    print(f"  freeze_shift_after_step={freeze_shift_after_step}")
    print(f"{'='*70}", flush=True)

    # Build model with the LARGEST block_size used (training or eval) so the
    # forward pass never hits the `T > self.block_size` guard.
    max_block = max(eval_block_size, max(bs for bs, _ in phases))
    base_cfg = Config(
        use_pos_embed=False, n_embd=n_embd,
        num_steps=total_steps,
        block_size=max_block,
        warmup_steps=max(300, total_steps // 30),
        eval_interval=eval_interval,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=0.0,
    )
    set_ablation_from_cfg(base_cfg)
    device = resolve_device(base_cfg.device)

    # ----- model -----
    torch.manual_seed(1337)
    factory = make_factory(int4_head, int4_ssm_C, int4_conv, False)
    model = factory(27, base_cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    print(f"params: {total:,}  ternary: {model.ternary_param_count():,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=base_cfg.learning_rate,
                            weight_decay=base_cfg.weight_decay)

    # Pre-build eval loader (constant block_size)
    eval_cfg = dataclasses.replace(base_cfg, block_size=eval_block_size)
    _, valid_loader, vocab = make_loaders(eval_cfg)

    # CSV writers
    steps_csv = csv_dir / f"curriculum_{cell_label}_steps.csv"
    eval_csv  = csv_dir / f"curriculum_{cell_label}_eval.csv"
    sf = open(steps_csv, "w", newline=""); sw = csv.writer(sf)
    sw.writerow(["step", "block_size", "lr", "train_loss", "train_ema", "gn"])
    ef = open(eval_csv, "w", newline=""); ew = csv.writer(ef)
    ew.writerow(["step", "block_size_train", "valid_loss"])

    train_ema = None
    step = 0
    frozen = False
    t0 = time.time()
    final = {"valid": math.nan}

    for phase_idx, (bs, n_steps) in enumerate(phases):
        phase_cfg = dataclasses.replace(base_cfg, block_size=bs)
        train_loader, _, _ = make_loaders(phase_cfg)
        train_iter = iter(train_loader)
        print(f"\n[{cell_label}] phase {phase_idx}: bs={bs}, "
              f"steps {step}..{step + n_steps}", flush=True)
        model.train()

        for _ in range(n_steps):
            lr = lr_for_step(step, base_cfg)
            for grp in opt.param_groups:
                grp["lr"] = lr
            if (freeze_shift_after_step is not None and not frozen
                    and step >= freeze_shift_after_step):
                n = freeze_shift_params(model, freeze=True)
                print(f"  [{cell_label}] step {step}: froze {n} shift params", flush=True)
                frozen = True

            try:
                xb, yb = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                xb, yb = next(train_iter)
            xb, yb = xb.to(device), yb.to(device)
            _, loss, _ = model(xb, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), base_cfg.grad_clip)
            gn_val = float(gn) if torch.isfinite(gn) else float("inf")
            opt.step()

            tl = float(loss.item())
            train_ema = tl if train_ema is None else (
                train_ema_alpha * tl + (1 - train_ema_alpha) * train_ema
            )
            sw.writerow([step, bs, f"{lr:.6e}", f"{tl:.4f}", f"{train_ema:.4f}", f"{gn_val:.3f}"])

            if step % eval_interval == 0 or step == total_steps - 1:
                v_loss = eval_loss(model, valid_loader, eval_n_batches, device)
                ew.writerow([step, bs, f"{v_loss:.4f}"])
                ef.flush()
                final = {"valid": v_loss}
                elapsed = time.time() - t0
                print(
                    f"step {step:6d} | bs {bs:3d} | lr {lr:.2e} | "
                    f"train_ema {train_ema:.4f} | valid {v_loss:.4f} | "
                    f"{elapsed:6.1f}s",
                    flush=True,
                )
                model.train()

            step += 1

    sf.close(); ef.close()
    elapsed = time.time() - t0
    print(f"\n[{cell_label}] DONE  final valid={final['valid']:.4f}  ({elapsed:.0f}s)", flush=True)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "cfg": dataclasses.replace(base_cfg, block_size=eval_block_size),
            "vocab": vocab,
            "stack": dict(int4_head=int4_head, int4_ssm_C=int4_ssm_C,
                          int4_conv=int4_conv, tie_embeddings=False,
                          freeze_shift_after_frac=None,
                          freeze_shift_after_step=freeze_shift_after_step,
                          phases=phases),
            "losses": final,
            "dataset": "tinystories_top500",
        }, save_path)
        print(f"  saved → {save_path}")
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=8000,
                   help="total gradient steps per cell")
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-block", type=int, default=128)
    p.add_argument("--cells", default="control,curriculum",
                   help="comma-sep subset of {control,curriculum}")
    args = p.parse_args()

    cells = set(args.cells.split(","))
    half = args.steps // 2
    freeze_at = args.steps // 2  # freeze at 50% in both
    plans = {
        "control":    ("C1_control",    [(args.eval_block, args.steps)]),
        "curriculum": ("C2_curriculum", [(32, half), (args.eval_block, args.steps - half)]),
    }
    for key in ("control", "curriculum"):
        if key not in cells:
            continue
        label, phases = plans[key]
        run_phased(
            label, phases,
            eval_block_size=args.eval_block,
            eval_interval=args.eval_interval,
            freeze_shift_after_step=freeze_at,
            save_path=f"build/bitnet_quant_{label}.pt",
        )


if __name__ == "__main__":
    main()
