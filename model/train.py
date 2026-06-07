"""Training entry point.

Run from the repo root:

    python -m model.train --n-embd 56 --steps 30000 \\
                          --save build/bitnet_quant_n56_full.pt

Default config matches the BBC-deployed `bitnet_quant_n56_full.pt`
(valid loss 1.03 after 30,000 steps).
"""
from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset.data import build_datasets, Vocabulary
from model.model import BitNetLM, ModelConfig, freeze_shift_params


# ----------------------------------------------------------------------------- #
# Configs
# ----------------------------------------------------------------------------- #


@dataclass
class TrainConfig:
    """Optimisation hyperparameters."""
    learning_rate: float = 2.0e-3
    min_learning_rate: float = 1.0e-4
    weight_decay: float = 0.04
    grad_clip: float = 1.0
    num_steps: int = 30_000
    warmup_steps: int = 1_000
    batch_size: int = 128
    eval_interval: int = 1_000
    eval_batches: int = 40
    freeze_shift_at_frac: float = 0.5


@dataclass
class SampleConfig:
    """Generation defaults — also baked into the C softmax LUT at export time."""
    temperature: float = 0.9
    top_k: int = 8
    n_tokens: int = 200


# ----------------------------------------------------------------------------- #
# Training loop
# ----------------------------------------------------------------------------- #


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_for_step(step: int, train_cfg: TrainConfig) -> float:
    if step < train_cfg.warmup_steps:
        return train_cfg.learning_rate * (step + 1) / train_cfg.warmup_steps
    progress = (step - train_cfg.warmup_steps) / max(1, train_cfg.num_steps - train_cfg.warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return train_cfg.min_learning_rate + cosine * (train_cfg.learning_rate - train_cfg.min_learning_rate)


def _load_legacy_friendly(path: str | Path, *, map_location) -> dict:
    """Load a checkpoint, registering shim modules so old pickles still work.

    Older checkpoints (bitnet_quant_n56_full.pt, bitnet_quant_tinystories_final.pt)
    were saved when `Config` / `Vocabulary` lived in modules that have since been
    refactored away. We register lightweight placeholder classes under the old
    module names so the unpickler can rehydrate the non-tensor metadata
    fields; the state_dict (just tensors) is what we actually need to use.
    """
    import sys as _sys
    import types as _types

    class _Compat:
        def __setstate__(self, state): self.__dict__.update(state)
        def __init__(self, *args, **kwargs): pass

    for _mod in ("bitnet_quant", "shakespeare", "modelling.shakespeare",
                 "modelling.data", "modelling.model"):
        if _mod not in _sys.modules:
            _m = _types.ModuleType(_mod)
            _m.Config = type("Config", (_Compat,), {})
            _m.Vocabulary = type("Vocabulary", (_Compat,), {})
            _sys.modules[_mod] = _m
    return torch.load(str(path), map_location=map_location, weights_only=False)


@torch.no_grad()
def eval_loss(model: BitNetLM, loader: DataLoader, n_batches: int, device: torch.device) -> float:
    model.eval()
    losses: list[float] = []
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


def train(
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    save_path: str | Path | None = None,
    device_override: str | None = None,
    verbose: bool = True,
    vocab_path: str | Path | None = None,
    dedup_names: bool = False,
    strip_boilerplate: bool = False,
    init_from: str | Path | None = None,
) -> tuple[BitNetLM, Vocabulary, dict[str, float]]:
    """Train a BitNetLM on TinyStories and (optionally) save the checkpoint.

    Returns (model, vocab, final_losses). final_losses has keys
    {'train', 'valid', 'gn_max', 'gn_avg'}.

    `vocab_path` overrides the default top-500 word filter; `dedup_names`
    rewrites known character names to gendered canonicals (female→'lily',
    male→'tom'); `strip_boilerplate` removes 'once upon a time'/'the end'
    templating that would otherwise dominate the training distribution;
    `init_from` loads a previous checkpoint into the model (fine-tune mode).
    """
    device = resolve_device(device_override)

    ds_kwargs: dict = {"dedup_names": dedup_names, "strip_boilerplate": strip_boilerplate}
    if vocab_path is not None:
        ds_kwargs["vocab_path"] = vocab_path
    train_ds, valid_ds, vocab = build_datasets(block_size=model_cfg.block_size, **ds_kwargs)
    g = torch.Generator()
    g.manual_seed(1337)
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True,
        drop_last=True, generator=g, num_workers=0,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=train_cfg.batch_size, shuffle=False, drop_last=True,
    )
    train_iter = iter(train_loader)

    if verbose:
        print(f"device: {device}", flush=True)
        print(f"vocab size: {vocab.size}", flush=True)
        print(f"train windows: {len(train_ds):,}", flush=True)
        print(f"valid windows: {len(valid_ds):,}", flush=True)

    torch.manual_seed(1337)
    model = BitNetLM(model_cfg).to(device)
    if init_from is not None:
        if verbose:
            print(f"loading pretrained weights from {init_from}", flush=True)
        _ckpt = _load_legacy_friendly(init_from, map_location=device)
        model.load_state_dict(_ckpt["state_dict"])
    total = sum(p.numel() for p in model.parameters())
    if verbose:
        print(
            f"total parameters: {total:,}  ternary: {model.ternary_param_count():,}  int4: {model.int4_param_count():,}",
            flush=True,
        )

    opt = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay,
    )

    model.train()
    t0 = time.time()
    final = {"train": math.nan, "valid": math.nan, "gn_max": 0.0, "gn_avg": 0.0}
    gn_max = 0.0
    gn_sum = 0.0
    gn_count = 0
    freeze_step = (
        int(train_cfg.num_steps * train_cfg.freeze_shift_at_frac)
        if train_cfg.freeze_shift_at_frac is not None
        else None
    )
    frozen = False

    for step in range(train_cfg.num_steps):
        # LR schedule
        lr = lr_for_step(step, train_cfg)
        for grp in opt.param_groups:
            grp["lr"] = lr

        # Freeze activation-shift params partway through training.
        if freeze_step is not None and not frozen and step >= freeze_step:
            n = freeze_shift_params(model, freeze=True)
            print(f"  [step {step}] froze {n} shift params", flush=True)
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

        # Skip the optimizer step if any gradient is non-finite — calling
        # AdamW.step() with NaN/Inf corrupts the m/v running averages, and
        # the model never recovers. clip_grad_norm with NaN total_norm
        # produces NaN-scaled grads too, so we must detect this here.
        any_bad = any(
            (p.grad is not None) and (not torch.isfinite(p.grad).all())
            for p in model.parameters()
        )
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        gn_val = float(gn) if torch.isfinite(gn) else float("inf")
        gn_max = max(gn_max, gn_val)
        gn_sum += gn_val
        gn_count += 1
        if not any_bad and torch.isfinite(gn):
            opt.step()

        if step % train_cfg.eval_interval == 0 or step == train_cfg.num_steps - 1:
            v_loss = eval_loss(model, valid_loader, train_cfg.eval_batches, device)
            gn_avg = gn_sum / max(gn_count, 1)
            final = {"train": loss.item(), "valid": v_loss, "gn_max": gn_max, "gn_avg": gn_avg}
            if verbose:
                elapsed = time.time() - t0
                print(
                    f"step {step:6d} | lr {lr:.2e} | "
                    f"train {loss.item():.4f} | valid {v_loss:.4f} | "
                    f"gn_avg {gn_avg:7.2f} gn_max {gn_max:8.2f} | "
                    f"{elapsed:6.1f}s",
                    flush=True,
                )
            gn_max = 0.0
            gn_sum = 0.0
            gn_count = 0

            # Mid-training checkpoint: overwrite a sibling _ckpt.pt so we can
            # sample any time without waiting for the final save.
            if save_path is not None:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                mid_path = Path(save_path).with_name(Path(save_path).stem + "_ckpt.pt")
                torch.save(_checkpoint_dict(model, vocab, model_cfg, train_cfg, final, step), mid_path)

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(_checkpoint_dict(model, vocab, model_cfg, train_cfg, final, train_cfg.num_steps - 1), save_path)
        if verbose:
            print(f"  saved checkpoint → {save_path}", flush=True)

    return model, vocab, final


def _checkpoint_dict(
    model: BitNetLM,
    vocab: Vocabulary,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    losses: dict[str, float],
    step: int,
) -> dict:
    """Pure-dict checkpoint — no class dependencies, so the saved file can be
    loaded in any environment that has only torch + python."""
    return {
        "state_dict": model.state_dict(),
        "model_cfg": asdict(model_cfg),
        "train_cfg": asdict(train_cfg),
        "vocab": {"stoi": dict(vocab.stoi), "itos": list(vocab.itos)},
        "step": step,
        "losses": losses,
    }


# ----------------------------------------------------------------------------- #
# CLI
# ----------------------------------------------------------------------------- #


def main() -> None:
    p = argparse.ArgumentParser(description="Train the production BitNet LM on TinyStories.")
    p.add_argument("--n-embd", type=int, default=56,
                   help="Model width (default 56 — fits BBC Model B).")
    p.add_argument("--steps", type=int, default=30_000,
                   help="Number of training steps.")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2.0e-3)
    p.add_argument("--wd", type=float, default=0.04)
    p.add_argument("--freeze-frac", type=float, default=0.5,
                   help="Fraction of steps after which to freeze activation-shift params.")
    p.add_argument("--save", type=str, default=None,
                   help="Path to save the final checkpoint (also writes _ckpt.pt at every eval).")
    p.add_argument("--device", type=str, default=None,
                   help="Override device autodetect: 'mps' | 'cuda' | 'cpu'.")
    p.add_argument("--vocab-path", type=str, default=None,
                   help="Filter-vocab file (default: tinystories_vocab_top500.txt).")
    p.add_argument("--dedup-names", action="store_true",
                   help="Rewrite known character names to gendered canonicals "
                        "(female→'lily', male→'tom') before tokenising.")
    p.add_argument("--strip-boilerplate", action="store_true",
                   help="Drop TinyStories templating ('once upon a time', "
                        "'one day', 'the end') from each sentence.")
    p.add_argument("--init-from", type=str, default=None,
                   help="Pretrained checkpoint to load model weights from (fine-tune mode).")
    args = p.parse_args()

    model_cfg = ModelConfig(n_embd=args.n_embd, block_size=args.block_size)
    train_cfg = TrainConfig(
        num_steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.wd,
        warmup_steps=max(300, args.steps // 30),
        eval_interval=max(args.steps // 30, 200),
        freeze_shift_at_frac=args.freeze_frac,
    )

    print(f"\n{'='*70}")
    print(f"BitNet training: n_embd={model_cfg.n_embd}  steps={train_cfg.num_steps}")
    print(f"  lr={train_cfg.learning_rate}  wd={train_cfg.weight_decay}  "
          f"freeze_at={train_cfg.freeze_shift_at_frac}")
    if args.vocab_path:
        print(f"  vocab_path={args.vocab_path}")
    if args.dedup_names:
        print(f"  dedup_names=True")
    if args.strip_boilerplate:
        print(f"  strip_boilerplate=True")
    if args.init_from:
        print(f"  init_from={args.init_from}")
    print(f"{'='*70}")

    t0 = time.time()
    _, _, losses = train(model_cfg, train_cfg, save_path=args.save, device_override=args.device,
                         vocab_path=args.vocab_path, dedup_names=args.dedup_names,
                         strip_boilerplate=args.strip_boilerplate,
                         init_from=args.init_from)
    elapsed = time.time() - t0
    print(f"\ndone: train={losses['train']:.4f}  valid={losses['valid']:.4f}  ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
