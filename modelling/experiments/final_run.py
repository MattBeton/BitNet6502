"""Final full-scale run: stacks the winning experiments and trains long.

Configure the stack via the constants at the top of the file, then run.
Each technique is independently toggleable so this same script can do an
ablation-style comparison at full scale without rewriting.

Stacked techniques (selected after the 4k-step screen):
    - freeze_shifts:  E4 win — freeze shift params after `freeze_after_frac`
                      of training (training-only, free)
    - anneal:         E6 — alpha schedule on the ternary quantizer
                      (training-only, free)
    - int4_head:      E1a — int4 head weight (ROM cost: shrink n_embd)
    - int4_ssm_C:     E1b extension (ROM cost: shrink n_embd more)
    - int4_conv:      E1c extension (ROM cost: minor)
    - tie_embeddings: E5 — head shares token_embedding (frees 567 ROM bytes)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import Config, train_loop, freeze_shift_params
from budget import compute_total_bytes, solve_n_embd_for_budget
from e1a_int4_head import Int4Head, int4_quantize
from e1bc_int4_more import Int4SSMLayer
from e6_anneal import alpha_for_step, _ALPHA, annealed_ternary
from e5_tied_embeddings import TiedBitNetLM

import bitnet_quant
from bitnet_quant import QuantBitNetLM, fake_quant_int8, learned_shift_no_sat


# -----------------------------------------------------------------------------
# Combined model: any subset of int4_head / int4_ssm_C / int4_conv,
# optionally with tie_embeddings.
# -----------------------------------------------------------------------------

class StackedBitNetLM(QuantBitNetLM):
    def __init__(self, vocab_size: int, cfg: Config,
                 int4_head: bool = False, int4_ssm_C: bool = False,
                 int4_conv: bool = False, tie_embeddings: bool = False) -> None:
        super().__init__(vocab_size, cfg)
        self._tie = tie_embeddings
        # Swap blocks if any int4 SSM tensor is requested
        if int4_ssm_C or int4_conv:
            self.blocks = nn.ModuleList([
                Int4SSMLayer(cfg.n_embd, cfg.state_size, cfg.conv_kernel,
                             cfg.dropout, cfg.use_gate, cfg,
                             int4_C=int4_ssm_C, int4_conv=int4_conv)
                for _ in range(cfg.n_layer)
            ])
        # Head: tied takes priority over int4
        if tie_embeddings:
            del self.head
            head_shift_init = 11.0
            self.head_shift = nn.Parameter(torch.tensor(head_shift_init))
        elif int4_head:
            self.head = Int4Head(cfg.n_embd, vocab_size, init_shift=8.0)

    def forward(self, idx, targets=None, states=None, pos_offset=0):
        if not self._tie:
            return super().forward(idx, targets, states, pos_offset)
        # Tied head path
        from torch.nn import functional as F
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("sequence length exceeds block size")
        tok = fake_quant_int8(self.token_embedding[idx])
        if self.position_embedding is not None:
            pos = fake_quant_int8(self.position_embedding[pos_offset:pos_offset + T])
            x = fake_quant_int8(tok + pos)
        else:
            x = tok
        x = self.dropout(x)
        new_states = []
        for i, block in enumerate(self.blocks):
            s_in = states[i] if states is not None else None
            x, s_out = block(x, s_in)
            new_states.append(s_out)
        w = fake_quant_int8(self.token_embedding)
        acc = F.linear(x, w)
        logits = learned_shift_no_sat(acc, self.head_shift)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_states


def make_factory(int4_head, int4_ssm_C, int4_conv, tie_embeddings):
    def factory(vocab_size, cfg):
        return StackedBitNetLM(vocab_size, cfg,
                               int4_head=int4_head, int4_ssm_C=int4_ssm_C,
                               int4_conv=int4_conv, tie_embeddings=tie_embeddings)
    return factory


def make_hooks(num_steps, freeze_shift_after_frac, anneal,
               anneal_warmup_frac, anneal_ramp_end_frac):
    state = {"frozen": False}
    freeze_step = int(num_steps * freeze_shift_after_frac) if freeze_shift_after_frac else None

    def on_step_start(step, model, opt):
        if anneal:
            _ALPHA["value"] = alpha_for_step(step, num_steps,
                                             anneal_warmup_frac, anneal_ramp_end_frac)
        if freeze_step is not None and not state["frozen"] and step >= freeze_step:
            n = freeze_shift_params(model, freeze=True)
            print(f"  [final] step {step}: froze {n} shift params", flush=True)
            state["frozen"] = True

    return on_step_start


def run(num_steps: int = 12000,
        freeze_shift_after_frac: float | None = 0.6,
        anneal: bool = False,
        anneal_warmup_frac: float = 0.15,
        anneal_ramp_end_frac: float = 0.50,
        int4_head: bool = False,
        int4_ssm_C: bool = False,
        int4_conv: bool = False,
        tie_embeddings: bool = False,
        target_bytes: int = 23_770,
        save_path: str | None = None) -> dict:
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
    print(f"FINAL RUN: n_embd={n_embd}  bytes={bytes_used} / {target_bytes}")
    print(f"  freeze_shift_after_frac={freeze_shift_after_frac}  anneal={anneal}")
    print(f"  int4_head={int4_head}  int4_ssm_C={int4_ssm_C}  int4_conv={int4_conv}")
    print(f"  tie_embeddings={tie_embeddings}  num_steps={num_steps}")
    print(f"{'='*70}")

    # Optionally monkey-patch the ternary quantizer for annealing
    if anneal:
        orig = bitnet_quant.ternary_quantize
        bitnet_quant.ternary_quantize = annealed_ternary

    cfg = Config(
        use_pos_embed=False,
        n_embd=n_embd,
        num_steps=num_steps,
        warmup_steps=max(300, num_steps // 20),
        eval_interval=max(num_steps // 20, 200),
    )
    factory = make_factory(int4_head, int4_ssm_C, int4_conv, tie_embeddings)
    hooks = make_hooks(num_steps, freeze_shift_after_frac, anneal,
                       anneal_warmup_frac, anneal_ramp_end_frac)

    t0 = time.time()
    try:
        model, vocab, losses = train_loop(cfg, model_factory=factory,
                                          on_step_start=hooks)
    finally:
        if anneal:
            bitnet_quant.ternary_quantize = orig
            _ALPHA["value"] = 1.0

    elapsed = time.time() - t0
    print(f"\n[final] test_loss={losses['test']:.4f}  train={losses['train']:.4f}  ({elapsed:.0f}s)")

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
        }, save_path)
        print(f"  saved checkpoint → {save_path}")
    return losses


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=12000)
    p.add_argument("--freeze-frac", type=float, default=0.6)
    p.add_argument("--anneal", action="store_true")
    p.add_argument("--int4-head", action="store_true")
    p.add_argument("--int4-ssm-C", action="store_true")
    p.add_argument("--int4-conv", action="store_true")
    p.add_argument("--tie", action="store_true")
    p.add_argument("--save", type=str, default=None)
    args = p.parse_args()

    run(num_steps=args.steps,
        freeze_shift_after_frac=args.freeze_frac,
        anneal=args.anneal,
        int4_head=args.int4_head,
        int4_ssm_C=args.int4_ssm_C,
        int4_conv=args.int4_conv,
        tie_embeddings=args.tie,
        save_path=args.save)
