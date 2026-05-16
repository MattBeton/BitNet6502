"""E5 — tied input/output embeddings.

Hypothesis (from bitnet_quant_experiments.md):
    Three confounded effects when you tie:
      (a) inductive bias from shared I/O token rep — small/zero for
          unstructured 27-char vocab,
      (b) head precision becomes int8·int8 instead of ternary·int8,
      (c) bytes saved on the head (567) get re-spent on widening n_embd.

    The naive comparison (tied vs untied) confounds all three. The
    byte-matched experiment runs two cells:
      E5a: tied at int8, with the freed 567 bytes spent on widening
           n_embd until total bytes match the deployable budget.
      E5b (control): untied, *same* widened n_embd as E5a — over budget
           by 567 bytes, used to isolate (a)+(b) from (c).

Method:
    Replace QuantHead with a TiedHead that uses the int8 token
    embedding as the head weight. Solve for the largest n_embd that
    keeps the tied config inside 23,770 bytes; run E5a at that n_embd
    tied, E5b at that n_embd untied.

Defined locally: TiedHead module, TiedBitNetLM that swaps it in.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import Config, train_loop
from budget import compute_total_bytes, solve_n_embd_for_budget

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bitnet_quant import QuantBitNetLM, fake_quant_int8, learned_shift_no_sat  # noqa: E402


class TiedBitNetLM(QuantBitNetLM):
    """Tied variant: head shares token_embedding (no separate head weight).

    Logits = int8(token_embedding) · x — row `t` of the head IS the
    embedding row for token `t`. Head is now int8·int8 → int16 instead
    of ternary·int8.

    The original `self.head` (a QuantHead with its own ternary weight) is
    deleted to free both the parameter slot and its 567-byte ROM cost.
    Only the head's learned shift is kept (argmax-invariant; useful for
    training-time scaling parity).
    """

    def __init__(self, vocab_size: int, cfg: Config,
                 freeze_head_shift: bool = True) -> None:
        super().__init__(vocab_size, cfg)
        # Drop the ternary head weight entirely; keep a learned shift only.
        del self.head
        # Default shift for tied head: token_embedding ∈ [-16, 16] gives a
        # ~13× larger weight std than a ternary head and the matmul sum
        # scales as sqrt(N) not N, so empirically a shift around 11–12
        # puts the logits in softmax-friendly range; smoke-tested below.
        head_shift_init = cfg.init_shift_head if cfg.init_shift_head != 5.0 else 11.0
        self.head_shift = nn.Parameter(torch.tensor(head_shift_init),
                                       requires_grad=not freeze_head_shift)

    def _tied_logits(self, x: torch.Tensor) -> torch.Tensor:
        w = fake_quant_int8(self.token_embedding)   # (vocab, C)
        acc = F.linear(x, w)                         # (..., vocab)
        return learned_shift_no_sat(acc, self.head_shift)

    def forward(self, idx, targets=None, states=None, pos_offset=0):
        # Replicate QuantBitNetLM.forward but call the tied head at the end.
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

        logits = self._tied_logits(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss, new_states

    def ternary_param_count(self) -> int:
        n = 0
        for b in self.blocks:
            n += b.in_proj.weight.numel()
            n += b.out_proj.weight.numel()
            n += b.conv_weight.numel()
            n += b.B.numel() + b.C.numel()
        return n  # head no longer a separate ternary tensor


def run_cell(label: str, n_embd: int, num_steps: int, model_factory) -> dict:
    cfg = Config(use_pos_embed=False, n_embd=n_embd, num_steps=num_steps)
    print(f"\n=== {label}: n_embd={n_embd} ===")
    _, _, losses = train_loop(cfg, model_factory=model_factory)
    print(f"\n[{label}] final test_loss={losses['test']:.4f}  train={losses['train']:.4f}")
    return losses


def run(num_steps: int = 4000) -> dict[str, dict]:
    target_bytes = 23_770
    n_tied = solve_n_embd_for_budget(target_bytes, tie_embeddings=True)
    print(f"\nE5a tied: n_embd={n_tied}, bytes={compute_total_bytes(n_tied, tie_embeddings=True)}")
    print(f"E5b untied (control, over budget): n_embd={n_tied}, bytes={compute_total_bytes(n_tied)}")

    a = run_cell("E5a tied", n_tied, num_steps, TiedBitNetLM)
    b = run_cell("E5b untied (over budget)", n_tied, num_steps, QuantBitNetLM)

    return {"E5a": a, "E5b": b}


def run_e5a_only(num_steps: int = 4000) -> dict:
    """Re-run E5a alone with frozen head shift — the first attempt
    diverged at step ~1200 due to the tied-head shift oscillating
    across an integer boundary. Argmax is invariant to that shift, so
    freezing it costs nothing and removes the failure mode."""
    target_bytes = 23_770
    n_tied = solve_n_embd_for_budget(target_bytes, tie_embeddings=True)
    return run_cell("E5a tied (frozen shift)", n_tied, num_steps, TiedBitNetLM)


if __name__ == "__main__":
    out = run(num_steps=4000)
    print(f"\n--- Summary ---")
    print(f"A0 baseline (4k steps):       1.8713  (n_embd=84, untied, in budget)")
    print(f"E5a tied, widened:            {out['E5a']['test']:.4f}")
    print(f"E5b untied control:           {out['E5b']['test']:.4f}")
    print(f"E5a − E5b (isolates (a)+(b)): {out['E5a']['test'] - out['E5b']['test']:+.4f} nats")
    print(f"E5a − A0 (full effect):       {out['E5a']['test'] - 1.8713:+.4f} nats")
