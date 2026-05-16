"""Word-level training run on TinyStories.

Mirrors `tinystories_run.py` but uses `word_tinystories.build_word_datasets`
to produce word-token streams. Vocab size is tunable via --vocab-size; default
64 to fit on a BBC Model B.

Usage:
    python modelling/experiments/tinystories_word_run.py \\
        --steps 30000 --vocab-size 64 --n-embd 56 --no-anneal --wd 0.04 \\
        --save build/bitnet_quant_word64.pt
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

from common import Config
from final_run import StackedBitNetLM, make_factory, make_hooks
import bitnet_quant
from bitnet_quant import lr_for_step, resolve_device, set_ablation_from_cfg
from e6_anneal import _ALPHA, annealed_ternary

import word_tinystories


TOP_WORDS_PATH = HERE.parent / "data" / "tinystories_vocab_top500.txt"


def make_ts_word_loaders(cfg: Config, vocab_size: int):
    train_ds, valid_ds, vocab = word_tinystories.build_word_datasets(
        block_size=cfg.block_size,
        top_words_path=TOP_WORDS_PATH,
        vocab_size=vocab_size,
    )
    g = torch.Generator()
    g.manual_seed(1337)
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


def train_loop(cfg, *, model_factory, vocab_size, on_step_start=None,
               eval_n_batches=40, verbose=True, checkpoint_path=None):
    set_ablation_from_cfg(cfg)
    device = resolve_device(cfg.device)
    train_loader, valid_loader, vocab = make_ts_word_loaders(cfg, vocab_size)
    train_iter = iter(train_loader)

    if verbose:
        print(f"device: {device}", flush=True)
        print(f"vocab size: {vocab.size}", flush=True)
        print(f"vocab itos[:8]: {vocab.itos[:8]}", flush=True)
        print(f"train tokens: {len(train_loader.dataset):,}", flush=True)
        print(f"valid tokens: {len(valid_loader.dataset):,}", flush=True)

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
        any_bad = False
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                any_bad = True
                break
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        gn_val = float(gn) if torch.isfinite(gn) else float("inf")
        gn_max = max(gn_max, gn_val)
        gn_sum += gn_val
        gn_count += 1
        if not any_bad and torch.isfinite(gn):
            opt.step()

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
            gn_max = 0.0
            gn_sum = 0.0
            gn_count = 0
            if checkpoint_path is not None:
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--wd", type=float, default=0.04)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--freeze-frac", type=float, default=0.5)
    p.add_argument("--no-anneal", action="store_true", default=True,
                   help="Anneal is off by default for word-level (proven unstable on TinyStories)")
    p.add_argument("--anneal", dest="no_anneal", action="store_false")
    p.add_argument("--n-embd", type=int, default=56)
    p.add_argument("--vocab-size", type=int, default=64)
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    cfg = Config(
        use_pos_embed=False,
        n_embd=args.n_embd,
        num_steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        warmup_steps=max(300, args.steps // 30),
        eval_interval=max(args.steps // 30, 200),
        learning_rate=args.lr,
        weight_decay=args.wd,
        dropout=args.dropout,
    )

    int4_head = True
    int4_ssm_C = True
    int4_conv = True
    anneal = not args.no_anneal

    print(f"\n{'='*70}")
    print(f"Word-level TinyStories: vocab_size={args.vocab_size}  n_embd={args.n_embd}")
    print(f"  steps={args.steps}  wd={args.wd}  anneal={anneal}  freeze_frac={args.freeze_frac}")
    print(f"{'='*70}")

    if anneal:
        orig_q = bitnet_quant.ternary_quantize
        bitnet_quant.ternary_quantize = annealed_ternary

    factory = make_factory(int4_head, int4_ssm_C, int4_conv, tie_embeddings=False)
    hooks = make_hooks(args.steps, args.freeze_frac, anneal, 0.15, 0.50)

    t0 = time.time()
    try:
        model, vocab, losses = train_loop(
            cfg, model_factory=factory, vocab_size=args.vocab_size,
            on_step_start=hooks, verbose=True, checkpoint_path=args.save,
        )
    finally:
        if anneal:
            bitnet_quant.ternary_quantize = orig_q
            _ALPHA["value"] = 1.0

    elapsed = time.time() - t0
    print(f"\n[final-word] train={losses['train']:.4f}  valid={losses['valid']:.4f}  ({elapsed:.0f}s)")

    if args.save is not None:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": model.state_dict(),
            "cfg": cfg,
            "vocab": vocab,
            "stack": dict(int4_head=int4_head, int4_ssm_C=int4_ssm_C,
                          int4_conv=int4_conv, tie_embeddings=False,
                          freeze_shift_after_frac=args.freeze_frac, anneal=anneal),
            "losses": losses,
            "dataset": "tinystories_word",
            "vocab_size": args.vocab_size,
        }, args.save)
        print(f"  saved checkpoint → {args.save}")


if __name__ == "__main__":
    main()
