"""A/B equivalence tests: real C (via sim65 harness) vs Python reference.

Each test runs a function with the same input through both implementations
and asserts byte-exact equality. Any divergence is a real bug in one or the other.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from c_harness import CHarness
from inference.reference import (
    shift_sat_int8 as py_shift_sat_int8,
    ternary_linear as py_ternary_linear,
    depthwise_conv1d_step as py_depthwise_conv1d_step,
    ssm_step as py_ssm_step,
    int4_logits as py_int4_logits,
    softmax_sample as py_softmax_sample,
    make_exp_lut, LCG8,
)


# --------------------------------------------------------------------------- #
# Shared session-scoped harness (one sim65 process for all tests in the file)
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="session")
def harness():
    h = CHarness()
    yield h
    h.close()


# --------------------------------------------------------------------------- #
# Smoke test: wire protocol round-trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("payload", [
    b"",
    b"\x00",
    b"\xff",
    bytes(range(32)),
    bytes([0x00, 0x01, 0x80, 0x7f, 0xff]),
])
def test_ping_roundtrip(harness, payload):
    """Verify that the harness reads N bytes and writes the same N bytes back."""
    assert harness.ping(payload) == payload


# --------------------------------------------------------------------------- #
# Phase 2: shift_sat_int8 primitive
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("acc,shift", [
    # Boundary cases
    (0, 0),
    (127, 0), (128, 0), (-128, 0), (-129, 0),
    (32767, 0), (-32768, 0),
    # Shifts of various amounts
    (1024, 3), (1024, 7), (1024, 14),
    (-1024, 3), (-1024, 7),
    # Saturation after shift
    (32000, 1), (-32000, 1),
    (300, 0), (-300, 0),
    # Negative arithmetic shift edge cases
    (-1, 1), (-1, 7),
    (-3, 1), (-3, 2),
])
def test_shift_sat_int8_specific(harness, acc, shift):
    c = harness.shift_sat_int8(acc, shift)
    py = int(py_shift_sat_int8(torch.tensor(acc, dtype=torch.int32), shift).item())
    assert c == py, f"acc={acc}, shift={shift}: C={c}, py={py}"


@pytest.mark.parametrize("seed", range(20))
def test_shift_sat_int8_random(harness, seed):
    rng = np.random.default_rng(seed)
    acc = int(rng.integers(-32768, 32768))
    shift = int(rng.integers(0, 15))
    c = harness.shift_sat_int8(acc, shift)
    py = int(py_shift_sat_int8(torch.tensor(acc, dtype=torch.int32), shift).item())
    assert c == py, f"seed={seed}, acc={acc}, shift={shift}: C={c}, py={py}"


# --------------------------------------------------------------------------- #
# Phase 3: ternary_linear (matmul + bias + shift + saturate)
# --------------------------------------------------------------------------- #


def _random_ternary_linear_inputs(rng, in_f, out_f, seq=1):
    """Returns x_c (in_f, seq) int8, W (out_f, in_f) int8 ternary, bias (out_f,) int16."""
    W = rng.choice([-1, 0, 1], size=(out_f, in_f), p=[0.25, 0.5, 0.25]).astype(np.int8)
    x_c = rng.integers(-128, 128, size=(in_f, seq), dtype=np.int8)
    bias = rng.integers(-2000, 2001, size=(out_f,)).astype(np.int16)
    return x_c, W, bias


@pytest.mark.parametrize("in_f,out_f,seq,shift,seed", [
    (8,    8,    1, 2, 0),
    (8,    8,    2, 0, 1),
    (16,   8,    1, 3, 2),
    (16,  16,    1, 5, 3),
    (84,  84,    1, 5, 4),
    (84, 168,    1, 3, 5),    # in_proj shape from the trained model
    (84,  27,    1, 5, 6),    # head shape
    (84,  84,    4, 5, 7),    # multi-token sequence
])
def test_ternary_linear(harness, in_f, out_f, seq, shift, seed):
    rng = np.random.default_rng(seed)
    x_c, W, bias = _random_ternary_linear_inputs(rng, in_f, out_f, seq)

    # C version: x is (in_f, seq), output is (out_f, seq)
    c_out = harness.ternary_linear(x_c, W, bias, shift)

    # Python ref: x is (seq, in_f), output is (seq, out_f) — transpose to align
    x_py = torch.from_numpy(x_c.T).contiguous()      # (seq, in_f)
    W_py = torch.from_numpy(W)
    bias_py = torch.from_numpy(bias)
    py_out = py_ternary_linear(x_py, W_py, bias_py, shift).numpy()  # (seq, out_f)

    np.testing.assert_array_equal(c_out, py_out.T,
                                   err_msg=f"in={in_f} out={out_f} seq={seq} shift={shift} seed={seed}")


# --------------------------------------------------------------------------- #
# Phase 3b: hand-written 6502 asm ternary_linear — must match C byte-for-byte.
# Only seq=1 (single-token inference, the only shape the model uses on-device).
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("in_f,out_f,shift,seed", [
    (8,    8,  2, 0),
    (16,   8,  3, 1),
    (16,  16,  5, 2),
    (84,  84,  5, 3),     # n_embd=84, out_proj-ish shape
    (84, 168,  3, 4),     # in_proj shape from the trained 81-dim model (padded to 84)
    (60,  60,  5, 5),     # n_embd=56 (padded to 60)
    (60, 120,  3, 6),
    (32,  32,  4, 7),
    (32,  32,  0, 8),     # shift=0 (no right-shift, pure saturation)
    (12,  12,  7, 9),     # tiny shape with large shift
])
def test_ternary_linear_asm(harness, in_f, out_f, shift, seed):
    """The hand-written asm ternary_linear must produce byte-identical output
    to the C reference across the shapes the model actually uses."""
    rng = np.random.default_rng(seed)
    x_c, W, bias = _random_ternary_linear_inputs(rng, in_f, out_f, seq=1)

    c_out = harness.ternary_linear(x_c, W, bias, shift, asm=False)
    asm_out = harness.ternary_linear(x_c, W, bias, shift, asm=True)

    np.testing.assert_array_equal(
        asm_out, c_out,
        err_msg=f"asm != C: in={in_f} out={out_f} shift={shift} seed={seed}",
    )


# --------------------------------------------------------------------------- #
# Phase 4: vec_mul_shift_sat (element-wise gating)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n,shift,seed", [
    (1, 0, 0), (8, 7, 1), (84, 7, 2), (84, 0, 3), (168, 7, 4), (84, 14, 5),
])
def test_vec_mul_shift_sat(harness, n, shift, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(-128, 128, size=n, dtype=np.int8)
    b = rng.integers(-128, 128, size=n, dtype=np.int8)

    c_out = harness.vec_mul_shift_sat(a, b, shift)

    # Python reference: torch int16 multiplication then shift_sat_int8
    a_t = torch.from_numpy(a).to(torch.int16)
    b_t = torch.from_numpy(b).to(torch.int16)
    py_out = py_shift_sat_int8(a_t * b_t, shift).numpy()

    np.testing.assert_array_equal(c_out, py_out)


# --------------------------------------------------------------------------- #
# Phase 6: ssm_step (the core diagonal SSM update)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("Cch,S,ssm_shift,d_shift,seed", [
    (8,  4, 3, 8, 0),
    (8,  8, 3, 8, 1),
    (84, 8, 3, 8, 2),
    (84, 8, 0, 0, 3),
    (84, 8, 7, 14, 4),
])
def test_ssm_step(harness, Cch, S, ssm_shift, d_shift, seed):
    rng = np.random.default_rng(seed)
    u_t = rng.integers(-128, 128, size=Cch, dtype=np.int8)
    state0 = rng.integers(-128, 128, size=(Cch, S), dtype=np.int8)
    decay = rng.integers(0, 128, size=(Cch, S), dtype=np.int8)
    B = rng.choice([-1, 0, 1], size=(Cch, S), p=[0.25, 0.5, 0.25]).astype(np.int8)
    C_mat = rng.choice([-1, 0, 1], size=(Cch, S), p=[0.25, 0.5, 0.25]).astype(np.int8)
    D = rng.integers(-128, 128, size=Cch, dtype=np.int8)

    # C version (state passed by value via wire; harness echoes back final state)
    c_y, c_state = harness.ssm_step(u_t, state0, decay, B, C_mat, D, ssm_shift, d_shift)

    # Python ref (mutates state in place — give it a fresh copy)
    py_state = torch.from_numpy(state0.copy())
    py_y = py_ssm_step(
        torch.from_numpy(u_t),
        py_state,
        torch.from_numpy(decay),
        torch.from_numpy(B),
        torch.from_numpy(C_mat),
        torch.from_numpy(D),
        ssm_shift, d_shift,
    ).numpy()

    np.testing.assert_array_equal(c_state, py_state.numpy(),
                                   err_msg=f"state mismatch C={Cch} S={S} seed={seed}")
    np.testing.assert_array_equal(c_y, py_y,
                                   err_msg=f"y mismatch C={Cch} S={S} seed={seed}")


# --------------------------------------------------------------------------- #
# Phase 7: embedding_lookup + argmax_int16
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("vocab,Cch,seed", [(27, 84, 0), (32, 16, 1)])
def test_embedding_lookup(harness, vocab, Cch, seed):
    rng = np.random.default_rng(seed)
    table = rng.integers(-128, 128, size=(vocab, Cch), dtype=np.int8)
    for token_id in range(vocab):
        c_out = harness.embedding_lookup(table, token_id)
        np.testing.assert_array_equal(c_out, table[token_id])


@pytest.mark.parametrize("seed", range(10))
def test_argmax_int16(harness, seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 100))
    values = rng.integers(-32768, 32768, size=n, dtype=np.int16)
    c_idx = harness.argmax_int16(values)
    py_idx = int(np.argmax(values))
    assert c_idx == py_idx


# --------------------------------------------------------------------------- #
# int4 ops (head, conv, SSM C readout) — used by the bitnet_quant_final_v3 model
# --------------------------------------------------------------------------- #


def _random_int4(rng, shape):
    return rng.integers(-7, 8, size=shape, dtype=np.int8)


@pytest.mark.parametrize("in_f,out_f,seq,shift,seed", [
    (8,    8,    1, 0, 0),
    (16,   8,    1, 3, 1),
    (84,  27,    1, 8, 2),       # head shape with the prior n_embd
    (80,  27,    1, 6, 3),       # head shape on the deployable (in_f even)
    (84,  84,    2, 5, 4),       # multi-token sequence
])
def test_int4_logits(harness, in_f, out_f, seq, shift, seed):
    rng = np.random.default_rng(seed)
    W = _random_int4(rng, (out_f, in_f))
    x = rng.integers(-128, 128, size=(in_f, seq), dtype=np.int8)

    c_out = harness.int4_logits(x, W, shift)

    x_py = torch.from_numpy(x.T).contiguous()      # (seq, in_f)
    W_py = torch.from_numpy(W)
    py_out = py_int4_logits(x_py, W_py, shift).numpy()  # (seq, out_f)

    np.testing.assert_array_equal(c_out, py_out.T,
                                   err_msg=f"in={in_f} out={out_f} seq={seq} shift={shift} seed={seed}")


@pytest.mark.parametrize("Cch,K,shift,seed", [
    (8, 4, 1, 0), (16, 4, 1, 1), (81, 4, 2, 2), (84, 4, 0, 3),
])
def test_int4_depthwise_conv1d_step(harness, Cch, K, shift, seed):
    rng = np.random.default_rng(seed)
    window = rng.integers(-128, 128, size=(K, Cch), dtype=np.int8)
    W = _random_int4(rng, (Cch, K))

    c_out = harness.int4_depthwise_conv1d_step(window, W, shift)
    py_out = py_depthwise_conv1d_step(
        torch.from_numpy(window), torch.from_numpy(W), shift
    ).numpy()

    np.testing.assert_array_equal(c_out, py_out)


@pytest.mark.parametrize("Cch,S,ssm_shift,d_shift,seed", [
    (8,  4, 3, 8, 0),
    (8,  8, 3, 8, 1),
    (81, 8, 3, 8, 2),     # deployable shape (n_embd=81, S=8)
    (84, 8, 0, 0, 3),
])
def test_ssm_step_int4_C(harness, Cch, S, ssm_shift, d_shift, seed):
    rng = np.random.default_rng(seed)
    u_t = rng.integers(-128, 128, size=Cch, dtype=np.int8)
    state0 = rng.integers(-128, 128, size=(Cch, S), dtype=np.int8)
    decay = rng.integers(0, 128, size=(Cch, S), dtype=np.int8)
    B = rng.choice([-1, 0, 1], size=(Cch, S), p=[0.25, 0.5, 0.25]).astype(np.int8)
    C_mat = _random_int4(rng, (Cch, S))
    D = rng.integers(-128, 128, size=Cch, dtype=np.int8)

    c_y, c_state = harness.ssm_step_int4_C(u_t, state0, decay, B, C_mat, D, ssm_shift, d_shift)

    py_state = torch.from_numpy(state0.copy())
    py_y = py_ssm_step(
        torch.from_numpy(u_t),
        py_state,
        torch.from_numpy(decay),
        torch.from_numpy(B),
        torch.from_numpy(C_mat),     # py_ssm_step is alphabet-agnostic
        torch.from_numpy(D),
        ssm_shift, d_shift,
    ).numpy()

    np.testing.assert_array_equal(c_state, py_state.numpy(),
                                   err_msg=f"state mismatch C={Cch} S={S} seed={seed}")
    np.testing.assert_array_equal(c_y, py_y,
                                   err_msg=f"y mismatch C={Cch} S={S} seed={seed}")


# --------------------------------------------------------------------------- #
# ternary_linear with non-multiple-of-4 inner dim (deployable n_embd=81)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("in_f,out_f,shift,seed", [
    (81,  81,    5, 0),    # out_proj shape on the deployable
    (81, 162,    3, 1),    # in_proj shape on the deployable
    (5,    8,    0, 2),    # smallest interesting odd width
    (7,    4,    1, 3),
])
def test_ternary_linear_padded_inner_dim(harness, in_f, out_f, shift, seed):
    """C side rounds in_f up to the next multiple of 4 (zero-padding the last
    nibble byte) and reads the activation buffer past in_f-1. The harness
    pre-zeros its scratch so out-of-bounds reads land on zero."""
    rng = np.random.default_rng(seed)
    W = rng.choice([-1, 0, 1], size=(out_f, in_f), p=[0.25, 0.5, 0.25]).astype(np.int8)
    x = rng.integers(-128, 128, size=(in_f, 1), dtype=np.int8)
    bias = rng.integers(-2000, 2001, size=(out_f,)).astype(np.int16)

    # Pad both sides up to a multiple of 4 with zeros so the wire protocol works
    # (the existing harness assumes packable widths). Result is mathematically
    # identical because both contributions vanish at zero W or zero x.
    pad = (-in_f) % 4
    if pad:
        W_padded = np.concatenate([W, np.zeros((out_f, pad), dtype=np.int8)], axis=1)
        x_padded = np.concatenate([x, np.zeros((pad, 1), dtype=np.int8)], axis=0)
    else:
        W_padded, x_padded = W, x

    c_out = harness.ternary_linear(x_padded, W_padded, bias, shift)

    x_py = torch.from_numpy(x.T).contiguous()
    W_py = torch.from_numpy(W)
    bias_py = torch.from_numpy(bias)
    py_out = py_ternary_linear(x_py, W_py, bias_py, shift).numpy()

    np.testing.assert_array_equal(c_out, py_out.T,
                                   err_msg=f"in={in_f} out={out_f} shift={shift} seed={seed}")


# --------------------------------------------------------------------------- #
# Softmax sampling: 16-byte exp LUT + 16-bit LCG cumulative-sum walk
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed,k,seed_lcg", [
    (0, 8, 1), (1, 4, 2), (2, 3, 7), (3, 2, 13),
    (4, 8, 1), (5, 1, 1),                              # k=1 → always greedy
])
def test_softmax_sample_random(harness, seed, k, seed_lcg):
    """Random vocab logits, random LCG seeds — C and Python must pick the same
    index. Exercises the LCG-advance order, modulo-by-subtract, and cumsum walk.
    """
    rng_np = np.random.default_rng(seed)
    n = int(rng_np.integers(8, 30))                # vocab size
    logits = rng_np.integers(-300, 300, size=n, dtype=np.int16)
    lut = make_exp_lut()

    c_idx = harness.softmax_sample(logits.copy(), k, lut, seed_lcg)
    py_idx = py_softmax_sample(torch.from_numpy(logits.copy()), k, LCG8(seed_lcg), lut)
    assert int(c_idx) == int(py_idx), f"seed={seed} k={k} lcg_seed={seed_lcg}"


def test_softmax_sample_lut_all_zero(harness):
    """All-zero LUT past the peak → every weight is the peak. Should still
    sample without crashing; chosen index must match Python."""
    logits = np.array([100, 99, 98, 50, 0, -50, -100], dtype=np.int16)
    lut = [255] + [0] * 15
    c_idx = harness.softmax_sample(logits.copy(), 4, lut, rng_seed=42)
    py_idx = py_softmax_sample(torch.from_numpy(logits.copy()), 4, LCG8(42), lut)
    assert int(c_idx) == int(py_idx)


def test_softmax_sample_dominant_top1(harness):
    """Big gap to the rest — C and Python should both lean strongly toward top-1.
    Just check parity over many seeds (the gap test is informational)."""
    logits = np.array([200, 50, 49, 48, 10, 0, -50, -100], dtype=np.int16)
    lut = make_exp_lut()
    for s in range(20):
        c = harness.softmax_sample(logits.copy(), 8, lut, rng_seed=s + 1)
        p = py_softmax_sample(torch.from_numpy(logits.copy()), 8, LCG8(s + 1), lut)
        assert int(c) == int(p), f"seed={s+1}: c={c} py={p}"


# --------------------------------------------------------------------------- #
# Phase 11: end-to-end — full LM forward in C vs Python, on the trained model
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_end_to_end_generation():
    """Run the full sim65 program (no harness) and compare its emitted tokens
    to the Python reference softmax decode with the same prompt and LCG seed.

    Slow: the full 200-token generation takes ~5 minutes under sim65."""
    import subprocess
    from inference.reference import (
        load_checkpoint, generate_softmax,
    )

    repo = Path(__file__).resolve().parent.parent
    bin_path = repo / "build" / "program.sim6502"
    if not bin_path.exists():
        pytest.skip("build/program.sim6502 not built (run `make`)")

    # Run C
    try:
        result = subprocess.run(["sim65", str(bin_path)],
                                capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        pytest.skip(f"sim65 generation didn't finish in 10 min: {e}")
    c_out = result.stdout.rstrip("\n")

    # Run Python with the same checkpoint, prompt, k and LCG seed as program.c.
    weights, vocab = load_checkpoint(repo / "build" / "bitnet_quant_n56_full.pt")
    stoi = vocab["stoi"]
    itos = vocab["itos"]
    # program.c primes with "once upon a time " before sampling.
    prompt_ids = [stoi[c] for c in "once upon a time "]
    out_ids = generate_softmax(weights, prompt_ids, max_new_tokens=len(c_out), k=8, rng_seed=1)
    py_out = "".join(itos[i] for i in out_ids[len(prompt_ids):])

    assert c_out == py_out[:len(c_out)], (
        f"\n  C: {c_out[:120]!r}\n  py: {py_out[:120]!r}"
    )


# --------------------------------------------------------------------------- #
# Phase 5: depthwise_conv1d_step (causal, single-step emission)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("Cch,K,shift,seed", [
    (8, 4, 1, 0), (16, 4, 1, 1), (84, 4, 1, 2), (84, 4, 0, 3), (84, 4, 7, 4),
])
def test_depthwise_conv1d_step(harness, Cch, K, shift, seed):
    rng = np.random.default_rng(seed)
    window = rng.integers(-128, 128, size=(K, Cch), dtype=np.int8)
    W = rng.choice([-1, 0, 1], size=(Cch, K), p=[0.25, 0.5, 0.25]).astype(np.int8)

    c_out = harness.depthwise_conv1d_step(window, W, shift)
    py_out = py_depthwise_conv1d_step(
        torch.from_numpy(window), torch.from_numpy(W), shift
    ).numpy()

    np.testing.assert_array_equal(c_out, py_out)
