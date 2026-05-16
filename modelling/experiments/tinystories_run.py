"""Train the v3 stack on TinyStories instead of Shakespeare.

Adjustments vs `final_run.py` (which used Shakespeare):
  - Loaders point at the filtered TinyStories token cache.
  - dropout = 0 (overfitting risk is essentially zero on a 360M-token
    corpus for a 71k-param model).
  - weight_decay = 0.01 (was 0.04 — same reasoning).
  - Schedule extended (30k+ steps); the Shakespeare run plateaued
    around 12-20k mainly because the data was exhausted, not because
    the model converged.
  - Eval uses the actual valid file (no train_fraction split).

The model architecture is identical to `bitnet_quant_final_v3.pt`
(n_embd=81, int4 head + SSM C + conv, ternary in_proj/out_proj/B,
int8 acts) — same 27-char vocab on both sides means we can train from
scratch on TinyStories with no embedding-shape changes.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import Config, freeze_shift_params
from budget import compute_total_bytes, solve_n_embd_for_budget
from final_run import StackedBitNetLM, make_factory, make_hooks

import bitnet_quant
from bitnet_quant import lr_for_step, resolve_device, set_ablation_from_cfg
from e6_anneal import _ALPHA, annealed_ternary

import tinystories


VOCAB_PATH = HERE.parent / "data" / "tinystories_vocab_top500.txt"

# Set by run() at the top of the script via the --vocab-path CLI arg
ACTIVE_VOCAB_PATH = VOCAB_PATH


def make_ts_loaders(cfg: Config):
    train_ds, valid_ds, vocab = tinystories.build_datasets(
        block_size=cfg.block_size,
        vocab_path=ACTIVE_VOCAB_PATH,
    )
    g = torch.Generator(); g.manual_seed(1337)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        drop_last=True, generator=g, num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True,
    )
    return train_loader, valid_loader, vocab


@torch.no_grad()
def eval_loss(model, loader, n_batches: int, device) -> float:
    model.eval()
    losses = []
    it = iter(loader)
    for _ in range(n_batches):
        try:
            xb, yb = next(it)
        except StopIteration:
            break
        _, loss, _ = model(xb.to(device), yb.to(device))
        if loss is not None:
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1) if losses else float("nan")


def train_loop_ts(cfg: Config, *, model_factory, on_step_start=None,
                  eval_n_batches: int = 40, verbose: bool = True,
                  checkpoint_path: str | None = None,
                  checkpoint_every_eval: bool = True):
    set_ablation_from_cfg(cfg)
    device = resolve_device(cfg.device)
    train_loader, valid_loader, vocab = make_ts_loaders(cfg)
    train_iter = iter(train_loader)

    if verbose:
        print(f"device: {device}", flush=True)
        print(f"vocab size: {vocab.size}", flush=True)
        print(f"train windows: {len(train_loader.dataset):,}", flush=True)
        print(f"valid windows: {len(valid_loader.dataset):,}", flush=True)

    torch.manual_seed(1337)
    model = model_factory(vocab.size, cfg).to(device)
    total = sum(p.numel() for p in model.parameters())
    if hasattr(model, "ternary_param_count"):
        ternary = model.ternary_param_count()
        if verbose:
            print(f"total parameters: {total:,}  ternary: {ternary:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    model.train()
    t0 = time.time()
    final = {"train": math.nan, "valid": math.nan}
    # Track running grad-norm stats since the last eval so we can see
    # gradient explosions even if they only happen briefly.
    gn_max = 0.0
    gn_sum = 0.0
    gn_count = 0
    for step in range(cfg.num_steps):
        lr = lr_for_step(step, cfg)
        for grp in opt.param_groups:
            grp["lr"] = lr
        if on_step_start is not None:
            on_step_start(step, model, opt)

        try:
            xb, yb = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xb, yb = next(train_iter)
        xb, yb = xb.to(device), yb.to(device)

        _, loss, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Detect non-finite gradients BEFORE clipping so we can skip the
        # optimizer step entirely — calling Adam.step() with any NaN/Inf
        # in the gradient corrupts m/v running stats and the model never
        # recovers. clip_grad_norm with NaN total_norm produces NaN-scaled
        # gradients, so we must catch this here.
        any_bad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                any_bad = True
                break
        # clip_grad_norm_ returns the total norm BEFORE clipping; we log
        # this so we can detect explosions even when clipping hides them
        # from the optimizer.
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        gn_val = float(gn) if torch.isfinite(gn) else float("inf")
        gn_max = max(gn_max, gn_val)
        gn_sum += gn_val
        gn_count += 1
        if not any_bad and torch.isfinite(gn):
            opt.step()
        # else: silently skip — Adam state untouched, params unchanged.
        # The eval-time logging will reveal any sustained skip windows
        # via stuck loss / huge gn.

        if step % cfg.eval_interval == 0 or step == cfg.num_steps - 1:
            v_loss = eval_loss(model, valid_loader, eval_n_batches, device)
            elapsed = time.time() - t0
            gn_avg = gn_sum / max(gn_count, 1)
            final = {"train": loss.item(), "valid": v_loss,
                     "gn_max": gn_max, "gn_avg": gn_avg}
            if verbose:
                print(
                    f"step {step:6d} | lr {lr:.2e} | "
                    f"train {loss.item():.4f} | valid {v_loss:.4f} | "
                    f"gn_avg {gn_avg:6.2f} gn_max {gn_max:7.2f} | "
                    f"{elapsed:6.1f}s",
                    flush=True,
                )
            gn_max = 0.0; gn_sum = 0.0; gn_count = 0
            # Mid-training checkpoint: overwrite a sibling _ckpt.pt so the user
            # can sample any time without waiting for the final save.
            if checkpoint_path is not None and checkpoint_every_eval:
                ckpt_dir = Path(checkpoint_path).parent
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                mid_path = Path(checkpoint_path).with_name(
                    Path(checkpoint_path).stem + "_ckpt.pt"
                )
                torch.save({
                    "state_dict": model.state_dict(),
                    "cfg": cfg,
                    "vocab": vocab,
                    "step": step,
                    "losses": final,
                }, mid_path)
    return model, vocab, final


def run(num_steps: int = 30_000,
        freeze_shift_after_frac: float = 0.5,
        anneal: bool = True,
        anneal_warmup_frac: float = 0.15,
        anneal_ramp_end_frac: float = 0.50,
        int4_head: bool = True,
        int4_ssm_C: bool = True,
        int4_conv: bool = True,
        tie_embeddings: bool = False,
        target_bytes: int = 23_770,
        n_embd: int | None = None,
        batch_size: int = 128,
        block_size: int = 64,
        learning_rate: float = 2e-3,
        weight_decay: float = 0.01,
        dropout: float = 0.0,
        warmup_steps: int | None = None,
        eval_interval: int | None = None,
        save_path: str | None = None,
        verbose: bool = True) -> dict:
    if n_embd is None:
        n_embd = solve_n_embd_for_budget(
            target_bytes,
            int4_head=int4_head, int4_ssm_C=int4_ssm_C, int4_conv=int4_conv,
            tie_embeddings=tie_embeddings,
        )
    bytes_used = compute_total_bytes(
        n_embd,
        int4_head=int4_head, int4_ssm_C=int4_ssm_C, int4_conv=int4_conv,
        tie_embeddings=tie_embeddings,
    )
    print(f"\n{'='*70}")
    print(f"TinyStories run: n_embd={n_embd}  bytes={bytes_used} / {target_bytes}")
    print(f"  steps={num_steps}  bs={batch_size}  block={block_size}")
    print(f"  freeze_shift_after_frac={freeze_shift_after_frac}  anneal={anneal}")
    print(f"  int4_head={int4_head}  int4_ssm_C={int4_ssm_C}  int4_conv={int4_conv}")
    print(f"  tie_embeddings={tie_embeddings}")
    print(f"  weight_decay={weight_decay}  dropout={dropout}")
    print(f"{'='*70}")

    if anneal:
        orig_q = bitnet_quant.ternary_quantize
        bitnet_quant.ternary_quantize = annealed_ternary

    cfg = Config(
        use_pos_embed=False,
        n_embd=n_embd,
        num_steps=num_steps,
        batch_size=batch_size,
        block_size=block_size,
        warmup_steps=warmup_steps if warmup_steps is not None else max(300, num_steps // 30),
        eval_interval=eval_interval if eval_interval is not None else max(num_steps // 30, 200),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        dropout=dropout,
    )
    factory = make_factory(int4_head, int4_ssm_C, int4_conv, tie_embeddings)
    hooks = make_hooks(num_steps, freeze_shift_after_frac, anneal,
                       anneal_warmup_frac, anneal_ramp_end_frac)

    t0 = time.time()
    try:
        model, vocab, losses = train_loop_ts(
            cfg, model_factory=factory, on_step_start=hooks, verbose=verbose,
            checkpoint_path=save_path,
        )
    finally:
        if anneal:
            bitnet_quant.ternary_quantize = orig_q
            _ALPHA["value"] = 1.0

    elapsed = time.time() - t0
    print(f"\n[final-ts] train={losses['train']:.4f}  valid={losses['valid']:.4f}  ({elapsed:.0f}s)")

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "vocab": vocab,
            "stack": dict(int4_head=int4_head, int4_ssm_C=int4_ssm_C,
                          int4_conv=int4_conv, tie_embeddings=tie_embeddings,
                          freeze_shift_after_frac=freeze_shift_after_frac,
                          anneal=anneal),
            "losses": losses,
            "dataset": "tinystories_top500",
        }, save_path)
        print(f"  saved checkpoint → {save_path}")
    return losses


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--freeze-frac", type=float, default=0.5)
    p.add_argument("--no-anneal", action="store_true")
    p.add_argument("--no-int4-head", action="store_true")
    p.add_argument("--no-int4-ssm-C", action="store_true")
    p.add_argument("--no-int4-conv", action="store_true")
    p.add_argument("--save", type=str, default=None)
    p.add_argument("--n-embd", type=int, default=None,
                   help="Override n_embd directly (skips the budget solver).")
    p.add_argument("--vocab-path", type=str, default=None,
                   help="Path to filter-vocab file (default: tinystories_vocab_top500.txt).")
    args = p.parse_args()
    if args.vocab_path is not None:
        ACTIVE_VOCAB_PATH = Path(args.vocab_path)
        # Globals via locals don't work directly; rebind through module
        import sys as _sys
        _sys.modules[__name__].ACTIVE_VOCAB_PATH = ACTIVE_VOCAB_PATH

    run(
        num_steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        learning_rate=args.lr,
        weight_decay=args.wd,
        dropout=args.dropout,
        freeze_shift_after_frac=args.freeze_frac,
        anneal=not args.no_anneal,
        int4_head=not args.no_int4_head,
        int4_ssm_C=not args.no_int4_ssm_C,
        int4_conv=not args.no_int4_conv,
        n_embd=args.n_embd,
        save_path=args.save,
    )
