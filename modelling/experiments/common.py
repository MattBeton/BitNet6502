"""Shared boilerplate for experiment runners.

Each experiment file imports `train_loop` and supplies its own model factory
(possibly with subclassed modules) and any per-step hooks. This keeps the
experiment-specific changes localised to each file rather than scattering
flags across `bitnet_quant.py`.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader

# Make the modelling/ folder importable when running as `python modelling/experiments/...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bitnet_quant import (  # noqa: E402
    Config,
    QuantBitNetLM,
    estimate_loss,
    lr_for_step,
    next_train_batch,
    resolve_device,
    set_ablation_from_cfg,
)
from shakespeare import build_datasets  # noqa: E402


def make_loaders(cfg: Config):
    train_ds, test_ds, vocab = build_datasets(
        block_size=cfg.block_size, train_fraction=cfg.train_fraction
    )
    g = torch.Generator()
    g.manual_seed(1337)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=True
    )
    return train_loader, test_loader, vocab


def train_loop(
    cfg: Config,
    *,
    model_factory: Callable[[int, Config], torch.nn.Module] = QuantBitNetLM,
    on_step_start: Callable[[int, torch.nn.Module, torch.optim.Optimizer], None] | None = None,
    verbose: bool = True,
) -> tuple[torch.nn.Module, object, dict[str, float]]:
    """Standard training loop with optional per-step hooks.

    Returns (model, vocab, final_losses).
    """
    set_ablation_from_cfg(cfg)
    device = resolve_device(cfg.device)
    train_loader, test_loader, vocab = make_loaders(cfg)
    train_iter = iter(train_loader)

    if verbose:
        print(f"device: {device}", flush=True)
        print(f"vocab size: {vocab.size}", flush=True)

    torch.manual_seed(1337)
    model = model_factory(vocab.size, cfg).to(device)

    total = sum(p.numel() for p in model.parameters())
    if hasattr(model, "ternary_param_count"):
        ternary = model.ternary_param_count()
        if verbose:
            print(f"total parameters: {total:,}  ternary: {ternary:,}", flush=True)
    else:
        if verbose:
            print(f"total parameters: {total:,}", flush=True)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    model.train()
    t0 = time.time()
    final_losses = {"train": math.nan, "test": math.nan}
    for step in range(cfg.num_steps):
        lr = lr_for_step(step, cfg)
        for grp in opt.param_groups:
            grp["lr"] = lr

        if on_step_start is not None:
            on_step_start(step, model, opt)

        xb, yb, train_iter = next_train_batch(train_loader, train_iter, device)
        _, loss, _ = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % cfg.eval_interval == 0 or step == cfg.num_steps - 1:
            losses = estimate_loss(model, train_loader, test_loader, cfg, device)
            elapsed = time.time() - t0
            final_losses = losses
            if verbose:
                print(
                    f"step {step:5d} | lr {lr:.2e} | "
                    f"train {losses['train']:.4f} | test {losses['test']:.4f} | "
                    f"{elapsed:5.1f}s",
                    flush=True,
                )
    return model, vocab, final_losses


def freeze_shift_params(model: torch.nn.Module, freeze: bool = True) -> int:
    """Freeze (or unfreeze) every learned right-shift parameter.

    Catches both the inner `*.shift` (QuantTernaryLinear, QuantHead) and
    the per-block `conv_shift`, `ssm_out_shift`, `d_shift`, `gate_shift`
    in QuantSSMLayer.
    """
    n = 0
    for name, p in model.named_parameters():
        leaf = name.split(".")[-1]
        if leaf == "shift" or leaf.endswith("_shift"):
            p.requires_grad = not freeze
            n += 1
    return n
