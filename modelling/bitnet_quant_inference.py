"""Pure-integer inference for the BitNet quant state-space LM.

Loads a checkpoint that was trained with fake-quantization in float32, casts every
parameter to its target on-device dtype, and runs generation using only the integer
operations that map to 6502 hardware. This file is the Python reference for the
C inference engine — every function here is a candidate for direct translation.

No fake-quant. No autograd. No softmax / exp / division. Sampling is greedy argmax.

Datatype contract
-----------------
  Activations & residual stream:    int8   [-128, 127]
  SSM state:                        int8
  Conv lookback ring buffer:        int8
  Accumulators / matmul scratch:    int16  [-32768, 32767]   (held in int32 in torch)
  Weight matrices (ternary):        int8 storing values in {-1, 0, +1}
  Weight matrices (int4):           int8 storing values in [-7, +7]   (head, conv, SSM C)
  Biases:                           int16
  SSM decay (effective a/128):      int8 in [0, 127]
  SSM D:                            int8
  Per-layer right-shift amount:     python int (range 0..14)
  Token / position embeddings:      int8

Architecture (summary of bitnet_quant.py with the final stack)
--------------------------------------------------------------
  token_emb[id]  →  clip int8
    │
    └─ × 3 blocks:
         in_proj(x)                            → (2*C,) int8        (ternary linear + bias + shift)
         u, gate = split(_, half)
         u = depthwise_causal_conv1d(u)        → (C,) int8           (int4, K=4)
         y, state = ssm_step(u, state)         → (C,) int8           (B ternary, C int4, decay/D int8)
         y = (y * gate) >> gate_shift, sat     → (C,) int8           (gating)
         y = out_proj(y)                       → (C,) int8           (ternary linear + bias + shift)
         x = clip_int8(x + y)                  → (C,) int8           (residual)
    │
    └─ head(x): int4 linear → int16 logits → argmax → next token id
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import torch


# =============================================================================
# Datatype primitives — every line below maps to a small piece of 6502 C
# =============================================================================

def sat_int8(x: torch.Tensor) -> torch.Tensor:
    """Clamp any int tensor to [-128, 127] and cast to int8.

    C analogue: `if (a > 127) out = 127; else if (a < -128) out = -128; else out = (signed char)a;`
    """
    return torch.clamp(x, -128, 127).to(torch.int8)


def shift_sat_int8(acc: torch.Tensor, shift: int) -> torch.Tensor:
    """Arithmetic right-shift an int16/int32 accumulator, then saturate to int8.

    C analogue: `out = saturate_int8(acc >> shift)` — both ops in one helper.
    """
    return sat_int8(acc >> shift)


# =============================================================================
# Layer operations — these are the work-horses; every block calls these.
# =============================================================================

def ternary_linear(
    x: torch.Tensor,            # int8 (..., in_f)
    weight: torch.Tensor,       # int8 ternary (out_f, in_f), values in {-1, 0, +1}
    bias: torch.Tensor,         # int16 (out_f,)
    shift: int,
) -> torch.Tensor:              # int8 (..., out_f)
    """int8 input · ternary weight + int16 bias → int16 acc → >>shift → sat int8.

    Accumulator semantics: int16. We use int32 in torch for the matmul kernel; values
    are guaranteed to fit in int16 by the model design (n_embd × 127 ≤ 84 × 127 ≪ 32767).
    """
    acc = torch.matmul(x.to(torch.int32), weight.to(torch.int32).t())  # (..., out_f) int32
    acc = acc + bias.to(torch.int32)
    return shift_sat_int8(acc, shift)


def int4_logits(
    x: torch.Tensor,            # int8 (..., in_f)
    weight: torch.Tensor,       # int8 int4 (out_f, in_f), values in [-7, +7]
    shift: int,
) -> torch.Tensor:              # int16 (..., out_f) — head output, no saturation
    """Head linear: int8 input · int4 weight → int32 acc → >>shift → int16 logits.

    No saturation: we keep the int16 dynamic range so argmax is exact. With
    n_embd × 127 × 7 ≈ 72k, the pre-shift accumulator can exceed int16, but the
    learned head_shift (≈6 in the trained model) brings it well within int16.
    On the 6502 the accumulator is signed long (32-bit), shifted at the end.
    """
    acc = torch.matmul(x.to(torch.int32), weight.to(torch.int32).t())  # (..., out_f) int32
    return (acc >> shift).to(torch.int16)


def depthwise_conv1d_step(
    window: torch.Tensor,       # int8 (K, C) — last K input timesteps; oldest at index 0, newest at K-1
    weight: torch.Tensor,       # int8 ternary (C, K)
    shift: int,
) -> torch.Tensor:              # int8 (C,) — one output timestep
    """Causal depthwise conv1d, single-step emission.

    Per channel: y[c] = sum_k weight[c, k] * window[k, c], then >>shift, then sat int8.
    Convention matches F.conv1d after left-padding by K-1 zeros.
    """
    # Multiply along K then sum: (K, C) elementwise → sum over K → (C,)
    acc = (weight.t().to(torch.int32) * window.to(torch.int32)).sum(dim=0)  # (C,) int32
    return shift_sat_int8(acc, shift)


def ssm_step(
    u_t: torch.Tensor,          # int8 (C,) — input at this timestep
    ssm_state: torch.Tensor,    # int8 (C, S) — recurrent state, MUTATED in place
    decay: torch.Tensor,        # int8 (C, S), values in [0, 127]; effective decay = decay / 128
    B: torch.Tensor,            # int8 ternary (C, S)
    C_mat: torch.Tensor,        # int8 ternary (C, S)
    D: torch.Tensor,            # int8 (C,)
    ssm_out_shift: int,
    d_shift: int,
) -> torch.Tensor:              # int8 (C,) — output at this timestep
    """Diagonal SSM update for one timestep. Mutates ssm_state.

    For each channel c, state index s:
        decayed[c,s] = (decay[c,s] * state[c,s]) >> 7         (int16 → int8 range)
        b_u[c,s]     = B[c,s] * u_t[c]                         (ternary mult)
        state[c,s]   = sat_int8(decayed[c,s] + b_u[c,s])
    Then per channel:
        c_state[c]   = sum_s C[c,s] * state[c,s]               (sum into int16)
        c_part[c]    = sat_int8(c_state[c] >> ssm_out_shift)
        d_part[c]    = sat_int8((D[c] * u_t[c]) >> d_shift)
        y[c]         = sat_int8(c_part[c] + d_part[c])
    """
    # State update
    decayed = decay.to(torch.int16) * ssm_state.to(torch.int16)             # (C, S) int16
    decayed = decayed >> 7                                                  # (C, S) int16, in int8 range
    b_u = B.to(torch.int16) * u_t.to(torch.int16).unsqueeze(-1)             # (C, S) int16, in int8 range
    new_state = sat_int8(decayed + b_u)                                     # (C, S) int8
    ssm_state.copy_(new_state)

    # Output
    c_state = (C_mat.to(torch.int16) * ssm_state.to(torch.int16)).sum(dim=-1)   # (C,) int16
    c_part = shift_sat_int8(c_state, ssm_out_shift)                              # (C,) int8
    d_acc = D.to(torch.int16) * u_t.to(torch.int16)                              # (C,) int16
    d_part = shift_sat_int8(d_acc, d_shift)                                      # (C,) int8
    return sat_int8(c_part.to(torch.int16) + d_part.to(torch.int16))             # (C,) int8


# =============================================================================
# Block-level structures and forward
# =============================================================================

@dataclass
class BlockWeights:
    in_proj_weight: torch.Tensor    # int8 ternary (2C, C)
    in_proj_bias: torch.Tensor      # int16 (2C,)
    in_proj_shift: int
    conv_weight: torch.Tensor       # int8 ternary or int4 (C, K)
    conv_shift: int
    decay: torch.Tensor             # int8 (C, S), [0, 127]
    B: torch.Tensor                 # int8 ternary (C, S)
    C_mat: torch.Tensor             # int8 ternary or int4 (C, S)
    D: torch.Tensor                 # int8 (C,)
    ssm_out_shift: int
    d_shift: int
    gate_shift: int
    out_proj_weight: torch.Tensor   # int8 ternary (C, C)
    out_proj_bias: torch.Tensor     # int16 (C,)
    out_proj_shift: int
    use_gate: bool


@dataclass
class BlockState:
    ssm_state: torch.Tensor         # int8 (C, S)
    conv_buffer: torch.Tensor       # int8 (K, C) — oldest at 0, newest at K-1


def make_block_state(c_dim: int, s_dim: int, k_dim: int) -> BlockState:
    return BlockState(
        ssm_state=torch.zeros(c_dim, s_dim, dtype=torch.int8),
        conv_buffer=torch.zeros(k_dim, c_dim, dtype=torch.int8),
    )


def block_step(
    x: torch.Tensor,                # int8 (C,) — input residual stream at this timestep
    w: BlockWeights,
    state: BlockState,
) -> torch.Tensor:                  # int8 (C,) — output residual stream
    """One timestep through one SSM block. Mutates state in place."""
    # 1) Input projection (and split into u, gate if gated)
    proj = ternary_linear(x, w.in_proj_weight, w.in_proj_bias, w.in_proj_shift)  # (2C,) int8
    if w.use_gate:
        c_dim = x.shape[0]
        u = proj[:c_dim]                                                          # (C,) int8
        gate = proj[c_dim:]                                                       # (C,) int8
    else:
        u = proj
        gate = None

    # 2) Depthwise conv (push u into the ring buffer, emit one output)
    state.conv_buffer = torch.cat([state.conv_buffer[1:], u.unsqueeze(0)], dim=0)  # int8 (K, C)
    u = depthwise_conv1d_step(state.conv_buffer, w.conv_weight, w.conv_shift)      # (C,) int8

    # 3) SSM recurrent update + output
    y = ssm_step(u, state.ssm_state, w.decay, w.B, w.C_mat, w.D,
                 w.ssm_out_shift, w.d_shift)                                       # (C,) int8

    # 4) Gating (element-wise int8 * int8 → int16, then >>shift, then sat int8)
    if gate is not None:
        y_acc = y.to(torch.int16) * gate.to(torch.int16)                           # (C,) int16
        y = shift_sat_int8(y_acc, w.gate_shift)                                    # (C,) int8

    # 5) Output projection
    y = ternary_linear(y, w.out_proj_weight, w.out_proj_bias, w.out_proj_shift)   # (C,) int8

    # 6) Residual: clip_int8(x + y)
    return sat_int8(x.to(torch.int16) + y.to(torch.int16))                         # (C,) int8


# =============================================================================
# Model-level structures and forward
# =============================================================================

@dataclass
class ModelConfig:
    block_size: int
    n_embd: int
    n_layer: int
    state_size: int
    conv_kernel: int
    use_gate: bool
    use_pos_embed: bool
    vocab_size: int


@dataclass
class ModelWeights:
    cfg: ModelConfig
    token_embedding: torch.Tensor       # int8 (vocab, C)
    pos_embedding: torch.Tensor | None  # int8 (block_size, C) or None
    blocks: list[BlockWeights]
    head_weight: torch.Tensor           # int8 ternary or int4 (vocab, C)
    head_shift: int                     # argmax-invariant; kept for fidelity to training
    # Per-tensor quantization flags. When False the tensor is ternary {-1,0,+1},
    # when True it is int4 [-7..+7]. The numeric kernels are identical for both;
    # the flags only matter for the on-device packed-byte layout (2-bit vs 4-bit).
    int4_head: bool = False
    int4_ssm_C: bool = False
    int4_conv: bool = False


def make_model_state(weights: ModelWeights) -> list[BlockState]:
    cfg = weights.cfg
    return [make_block_state(cfg.n_embd, cfg.state_size, cfg.conv_kernel)
            for _ in range(cfg.n_layer)]


def lm_step(
    token_id: int,
    pos: int,
    weights: ModelWeights,
    states: list[BlockState],
) -> torch.Tensor:                       # int16 (vocab,) — logits
    """One token in, logits out. Mutates block states."""
    # Embed
    x = weights.token_embedding[token_id].clone()                                  # (C,) int8
    if weights.pos_embedding is not None and pos < weights.cfg.block_size:
        x = sat_int8(x.to(torch.int16) + weights.pos_embedding[pos].to(torch.int16))

    # Stack of SSM blocks
    for w, s in zip(weights.blocks, states):
        x = block_step(x, w, s)                                                    # (C,) int8

    # Head: int8 · {ternary or int4} weight → int16 logits with learned shift.
    # No saturation: argmax needs the full int16 dynamic range.
    return int4_logits(x, weights.head_weight, weights.head_shift)


# =============================================================================
# Generation: greedy argmax + top-k with a tiny LCG (matches C exactly)
# =============================================================================

class LCG8:
    """8-bit LCG: state = state * 75 + 74 (mod 256). Mirrors C `rng_state` so
    that top-k sampling produces the same sequence on both sides."""
    def __init__(self, seed: int = 1) -> None:
        self.state = seed & 0xFF

    def next_u8(self) -> int:
        self.state = (self.state * 75 + 74) & 0xFF
        return self.state


def top_k_sample(logits: torch.Tensor, k: int, rng: LCG8) -> int:
    """Pick one of the top-k logit indices. Mutates `logits` (mask with INT_MIN
    after each pick) — matches the C implementation byte-for-byte."""
    if k > 16:
        k = 16
    logits = logits.clone()  # don't actually mutate caller's tensor in tests
    top_idx = []
    for _ in range(k):
        idx = int(logits.argmax().item())
        top_idx.append(idx)
        logits[idx] = -32768
    return top_idx[rng.next_u8() % k]


@torch.no_grad()
def generate_greedy(
    weights: ModelWeights,
    prompt_ids: list[int],
    max_new_tokens: int,
) -> list[int]:
    """Streaming generation: feed prompt one token at a time, then sample with argmax."""
    states = make_model_state(weights)
    out = list(prompt_ids)
    pos = 0
    logits = None
    # Prefill
    for tid in prompt_ids:
        logits = lm_step(tid, pos, weights, states)
        pos += 1
    # Decode
    for _ in range(max_new_tokens):
        next_id = int(logits.argmax().item())
        out.append(next_id)
        logits = lm_step(next_id, pos, weights, states)
        pos += 1
    return out


@torch.no_grad()
def generate_topk(
    weights: ModelWeights,
    prompt_ids: list[int],
    max_new_tokens: int,
    k: int = 4,
    rng_seed: int = 1,
) -> list[int]:
    """Streaming generation with top-k sampling via the same LCG the C engine uses."""
    rng = LCG8(rng_seed)
    states = make_model_state(weights)
    out = list(prompt_ids)
    pos = 0
    logits = None
    for tid in prompt_ids:
        logits = lm_step(tid, pos, weights, states)
        pos += 1
    for _ in range(max_new_tokens):
        next_id = top_k_sample(logits, k, rng)
        out.append(next_id)
        logits = lm_step(next_id, pos, weights, states)
        pos += 1
    return out


# =============================================================================
# Checkpoint loader — converts trained float32 fake-quants to integer dtypes
# =============================================================================

def _round_clip(t: torch.Tensor, dtype: torch.dtype, lo: int, hi: int) -> torch.Tensor:
    return torch.clamp(torch.round(t), lo, hi).to(dtype)


def _get(obj, key: str):
    """Access either dict[key] or obj.key — checkpoints differ in how cfg/vocab are stored."""
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def load_checkpoint(path: str | Path) -> tuple[ModelWeights, dict]:
    # Newer checkpoints pickle a `Config` from the top-level `bitnet_quant` module
    # and the experiment-stack subclasses (`StackedBitNetLM`, `Int4SSMLayer`, …)
    # from `modelling/experiments/`. Make both importable regardless of how this
    # function is called (pytest from repo root, sim65 driver, exporter, …).
    here = Path(__file__).resolve().parent
    for p in (here, here / "experiments"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    cfg_t = ck["cfg"]
    vocab = ck["vocab"]
    vocab_size = len(_get(vocab, "itos"))

    # Older checkpoints have no "stack" key — they're plain ternary everywhere.
    stack = ck.get("stack", {}) or {}
    int4_head    = bool(stack.get("int4_head", False))
    int4_ssm_C   = bool(stack.get("int4_ssm_C", False))
    int4_conv    = bool(stack.get("int4_conv", False))
    # Each int4 tensor uses the [-7, +7] alphabet (E1a/E1c convention).
    head_lo, head_hi = (-7, 7) if int4_head else (-1, 1)
    C_lo, C_hi       = (-7, 7) if int4_ssm_C else (-1, 1)
    conv_lo, conv_hi = (-7, 7) if int4_conv  else (-1, 1)

    cfg = ModelConfig(
        block_size=_get(cfg_t, "block_size"),
        n_embd=_get(cfg_t, "n_embd"),
        n_layer=_get(cfg_t, "n_layer"),
        state_size=_get(cfg_t, "state_size"),
        conv_kernel=_get(cfg_t, "conv_kernel"),
        use_gate=_get(cfg_t, "use_gate"),
        use_pos_embed=_get(cfg_t, "use_pos_embed"),
        vocab_size=vocab_size,
    )

    token_embedding = _round_clip(sd["token_embedding"], torch.int8, -128, 127)
    pos_embedding = (_round_clip(sd["position_embedding"], torch.int8, -128, 127)
                     if cfg.use_pos_embed else None)

    blocks: list[BlockWeights] = []
    for i in range(cfg.n_layer):
        p = f"blocks.{i}."
        blocks.append(BlockWeights(
            in_proj_weight  = _round_clip(sd[p + "in_proj.weight"], torch.int8, -1, 1),
            in_proj_bias    = _round_clip(sd[p + "in_proj.bias"], torch.int16, -32768, 32767),
            in_proj_shift   = int(round(sd[p + "in_proj.shift"].item())),
            conv_weight     = _round_clip(sd[p + "conv_weight"], torch.int8, conv_lo, conv_hi),
            conv_shift      = int(round(sd[p + "conv_shift"].item())),
            decay           = _round_clip(sd[p + "decay"], torch.int8, 0, 127),
            B               = _round_clip(sd[p + "B"], torch.int8, -1, 1),
            C_mat           = _round_clip(sd[p + "C"], torch.int8, C_lo, C_hi),
            D               = _round_clip(sd[p + "D"], torch.int8, -128, 127),
            ssm_out_shift   = int(round(sd[p + "ssm_out_shift"].item())),
            d_shift         = int(round(sd[p + "d_shift"].item())),
            gate_shift      = (int(round(sd[p + "gate_shift"].item()))
                               if cfg.use_gate else 0),
            out_proj_weight = _round_clip(sd[p + "out_proj.weight"], torch.int8, -1, 1),
            out_proj_bias   = _round_clip(sd[p + "out_proj.bias"], torch.int16, -32768, 32767),
            out_proj_shift  = int(round(sd[p + "out_proj.shift"].item())),
            use_gate        = cfg.use_gate,
        ))

    head_weight = _round_clip(sd["head.weight"], torch.int8, head_lo, head_hi)
    head_shift = int(round(sd["head.shift"].item()))

    return ModelWeights(
        cfg=cfg,
        token_embedding=token_embedding,
        pos_embedding=pos_embedding,
        blocks=blocks,
        head_weight=head_weight,
        head_shift=head_shift,
        int4_head=int4_head,
        int4_ssm_C=int4_ssm_C,
        int4_conv=int4_conv,
    ), vocab


# =============================================================================
# Main: load model, generate, print
# =============================================================================

def main() -> None:
    ckpt_path = Path(__file__).parent.parent / "build" / "bitnet_quant_final_v3.pt"
    weights, vocab = load_checkpoint(ckpt_path)

    cfg = weights.cfg
    print(f"loaded: vocab={cfg.vocab_size}  n_embd={cfg.n_embd}  n_layer={cfg.n_layer}  "
          f"state_size={cfg.state_size}  K={cfg.conv_kernel}  "
          f"gate={cfg.use_gate}  pos_embed={cfg.use_pos_embed}", flush=True)

    stoi = _get(vocab, "stoi")
    itos = _get(vocab, "itos")

    prompt = " "
    prompt_ids = [stoi[c] for c in prompt if c in stoi]
    if not prompt_ids:
        prompt_ids = [0]

    print(f"prompt: {prompt!r} (ids={prompt_ids})", flush=True)
    out_ids = generate_greedy(weights, prompt_ids, max_new_tokens=300)
    print("".join(itos[i] for i in out_ids))


if __name__ == "__main__":
    main()
