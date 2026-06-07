"""Byte budget for the deployable BitNet weight blob.

The 6502 inference can hold ~14 KB of weights on a BBC Model B (after
code + BSS + stack overhead within the 27.5 KB free RAM). This module
maps a candidate `n_embd` to the total weight bytes, so we can pick the
largest model that fits a given byte budget.

The mix is fixed (no ablation toggles):
  - in_proj, out_proj, B           ternary  → 2 bits per value, 4 per byte
  - conv kernel, SSM C, head       int4     → 4 bits per value, 2 per byte
  - token embedding, decay, D      int8     → 1 byte per value
  - biases                         int16    → 2 bytes per value
  - learned shifts                 int8     → 1 byte each
"""
from __future__ import annotations


def _ternary_bytes(n: int) -> int:
    return (n + 3) // 4


def _int4_bytes(n: int) -> int:
    return (n + 1) // 2


def compute_total_bytes(
    n_embd: int,
    *,
    vocab_size: int = 27,
    n_layer: int = 3,
    state_size: int = 8,
    conv_kernel: int = 4,
) -> int:
    """Total weight blob size in bytes for the given `n_embd`."""
    C, S, K, L, V = n_embd, state_size, conv_kernel, n_layer, vocab_size
    proj_out = 2 * C  # gate enabled

    # Ternary: in_proj (proj_out × C), out_proj (C × C), B (C × S) per block.
    ternary_count = (proj_out * C + C * C + C * S) * L

    # int4: conv kernel (C × K), SSM C (C × S) per block; head (V × C) once.
    int4_count = (C * K + C * S) * L + V * C

    # int8: token embedding (V × C); SSM decay (C × S × L); SSM D (C × L).
    int8_bytes = V * C + C * S * L + C * L

    # int16 biases: in_proj bias (proj_out) + out_proj bias (C) per block.
    int16_count = (proj_out + C) * L

    # Shifts: one head shift + 6 per block (in_proj, conv, ssm_out, d, gate, out_proj).
    shift_count = 1 + L * 6

    return (
        _ternary_bytes(ternary_count)
        + _int4_bytes(int4_count)
        + int8_bytes
        + 2 * int16_count
        + shift_count
    )


def solve_n_embd_for_budget(
    target_bytes: int = 14_000,
    *,
    n_min: int = 16,
    n_max: int = 200,
    **kwargs,
) -> int:
    """Largest n_embd whose total weight bytes ≤ target_bytes.

    The budget grows monotonically with n_embd (quadratic via the projection
    matrices), so a simple ascending linear search is correct.
    """
    best: int | None = None
    for n in range(n_min, n_max + 1):
        if compute_total_bytes(n, **kwargs) <= target_bytes:
            best = n
        else:
            break
    if best is None:
        raise RuntimeError(f"no n_embd in [{n_min}, {n_max}] fits {target_bytes} bytes")
    return best


def describe(n_embd: int, **kwargs) -> str:
    return f"n_embd={n_embd}: {compute_total_bytes(n_embd, **kwargs)} bytes"


if __name__ == "__main__":
    # Sanity prints for common sizes.
    for n in (40, 48, 56, 64, 72, 80, 84):
        print(describe(n))
    print()
    print("largest n_embd fitting:")
    for target in (10_000, 14_000, 16_000, 23_770):
        print(f"  {target:>6,} bytes → n_embd={solve_n_embd_for_budget(target)}")
