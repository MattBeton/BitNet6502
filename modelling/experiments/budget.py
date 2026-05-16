"""Byte budget calculator for the deployable BitNet weight blob.

Used by the experiment runners to keep the total weight bytes constant
while flipping between ternary / int4 / tied / wider configurations.
The numbers here are exact (matches `bitnet_quant_experiments.md`'s
23,770-byte total at n_embd=84).
"""
from __future__ import annotations


def _ternary_bytes(n: int) -> int:
    """4 ternary values pack into 1 byte (2 bits each)."""
    return (n + 3) // 4


def _int4_bytes(n: int) -> int:
    """2 int4 values pack into 1 byte."""
    return (n + 1) // 2


def compute_total_bytes(
    n_embd: int,
    *,
    vocab_size: int = 27,
    n_layer: int = 3,
    state_size: int = 8,
    conv_kernel: int = 4,
    use_gate: bool = True,
    use_pos_embed: bool = False,
    block_size: int = 64,
    int4_head: bool = False,
    int4_ssm_C: bool = False,
    int4_conv: bool = False,
    tie_embeddings: bool = False,
) -> int:
    """Total weight blob size in bytes for the given config.

    Verified at n_embd=84, default flags: 23,770 bytes — matches the
    deployable target.
    """
    C = n_embd
    S = state_size
    K = conv_kernel
    L = n_layer
    V = vocab_size
    proj_out = 2 * C if use_gate else C

    # ----- ternary tensors -----
    # Per-layer ternary that always stays ternary
    per_layer_ternary = (proj_out * C) + (C * C) + (C * S)  # in_proj, out_proj, B
    ternary_count = per_layer_ternary * L

    # Per-layer SSM C (ternary or int4)
    if int4_ssm_C:
        int4_count_C = (C * S) * L
    else:
        ternary_count += (C * S) * L
        int4_count_C = 0

    # Per-layer conv (ternary or int4)
    if int4_conv:
        int4_count_conv = (C * K) * L
    else:
        ternary_count += (C * K) * L
        int4_count_conv = 0

    # Head: ternary, int4, or tied (no head storage)
    if tie_embeddings:
        head_int4 = 0
    elif int4_head:
        head_int4 = V * C
    else:
        ternary_count += V * C
        head_int4 = 0

    int4_count = int4_count_C + int4_count_conv + head_int4

    # ----- int8 tensors -----
    int8_bytes = V * C            # token embedding
    int8_bytes += C * S * L       # SSM decay
    int8_bytes += C * L           # SSM D
    if use_pos_embed:
        int8_bytes += block_size * C

    # ----- int16 biases -----
    int16_count = (proj_out + C) * L  # in_proj bias + out_proj bias

    # ----- shifts (1 byte each) -----
    # head shift, plus per-layer: in_proj, conv, ssm_out, d_shift, out_proj
    # plus gate_shift if gating
    shift_count = 1 + L * (5 + (1 if use_gate else 0))

    return (
        _ternary_bytes(ternary_count)
        + _int4_bytes(int4_count)
        + int8_bytes
        + 2 * int16_count
        + shift_count
    )


def solve_n_embd_for_budget(
    target_bytes: int = 23_770,
    *,
    n_min: int = 16,
    n_max: int = 200,
    step: int = 1,
    **kwargs,
) -> int:
    """Largest n_embd whose total weight bytes <= target_bytes.

    Default step=1 because experiments doc allows non-multiples-of-4
    via row padding on the C side. The budget grows monotonically in
    n_embd (quadratic via proj weights), so simple ascending search is
    correct.
    """
    best: int | None = None
    for n in range(n_min, n_max + 1, step):
        b = compute_total_bytes(n, **kwargs)
        if b <= target_bytes:
            best = n
        else:
            break
    if best is None:
        raise RuntimeError(f"No n_embd in [{n_min},{n_max}] fits {target_bytes} bytes")
    return best


def describe(n_embd: int, **kwargs) -> str:
    b = compute_total_bytes(n_embd, **kwargs)
    flags = ", ".join(f"{k}={v}" for k, v in kwargs.items() if v is not False and v is not None)
    return f"n_embd={n_embd}: {b} bytes ({flags})" if flags else f"n_embd={n_embd}: {b} bytes"


if __name__ == "__main__":
    # Sanity: deployable model at n_embd=84 should be 23,770 bytes.
    print(describe(84))
    print(describe(84, int4_head=True))
    print(describe(82, int4_head=True))
    print("\nbudget-matched n_embd for each cell:")
    print("  E1a (int4 head):                ", solve_n_embd_for_budget(int4_head=True))
    print("  E1b (int4 head + C):            ", solve_n_embd_for_budget(int4_head=True, int4_ssm_C=True))
    print("  E1c (int4 head + C + conv):     ", solve_n_embd_for_budget(int4_head=True, int4_ssm_C=True, int4_conv=True))
    print("  E5a (tied embedding):           ", solve_n_embd_for_budget(tie_embeddings=True))
