"""Python driver for the sim65-based C test harness.

Spawns the harness binary as a long-lived subprocess and exchanges binary
messages over stdin/stdout. Each method on CHarness corresponds to one OP_*
in tests/c_harness/harness.c.
"""
from __future__ import annotations

import struct
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS_BIN = REPO_ROOT / "build" / "test_harness.sim6502"


def pack_ternary(W: np.ndarray) -> bytes:
    """Pack a 2-D int8 matrix with values in {-1, 0, +1} into 2-bit-per-value bytes.
    Encoding: 0→0b00, +1→0b01, -1→0b10. LSB first, 4 values per byte.
    Width must be a multiple of 4. Matches inference/c/matrix.c packing.
    """
    H, W_ = W.shape
    if W_ % 4 != 0:
        raise ValueError(f"ternary matrix width must be multiple of 4, got {W_}")
    out = bytearray(H * (W_ // 4))
    for i in range(H):
        for k in range(W_ // 4):
            byte = 0
            for l in range(4):
                v = int(W[i, 4 * k + l])
                if v == 0:
                    bits = 0b00
                elif v == 1:
                    bits = 0b01
                elif v == -1:
                    bits = 0b10
                else:
                    raise ValueError(f"non-ternary value {v} at ({i},{4*k+l})")
                byte |= bits << (2 * l)
            out[i * (W_ // 4) + k] = byte
    return bytes(out)


def pack_int4(W: np.ndarray) -> bytes:
    """Pack a 2-D int8 matrix with values in [-7, +7] (or [-8, +7]) into nibble-
    packed bytes. Encoding: low nibble first, signed 4-bit (two's complement).
    Width must be a multiple of 2. Matches inference/c/matrix.c unpack_nibble.
    """
    H, W_ = W.shape
    if W_ % 2 != 0:
        raise ValueError(f"int4 matrix width must be multiple of 2, got {W_}")
    out = bytearray(H * (W_ // 2))
    for i in range(H):
        for k in range(W_ // 2):
            lo = int(W[i, 2 * k]) & 0x0F
            hi = int(W[i, 2 * k + 1]) & 0x0F
            out[i * (W_ // 2) + k] = lo | (hi << 4)
    return bytes(out)


class HarnessError(RuntimeError):
    pass


class CHarness:
    OP_PING            = 0x00
    OP_SHIFT_SAT_INT8  = 0x01
    OP_TERNARY_LINEAR  = 0x02
    OP_VEC_MUL_SHIFT   = 0x03
    OP_DW_CONV1D_STEP  = 0x04
    OP_SSM_STEP        = 0x05
    OP_EMBED_LOOKUP    = 0x06
    OP_ARGMAX_INT16    = 0x07
    OP_INT4_LOGITS     = 0x08
    OP_INT4_DW_CONV1D  = 0x09
    OP_SSM_STEP_INT4_C = 0x0A
    OP_SOFTMAX_SAMPLE  = 0x0B
    OP_TERNARY_LINEAR_ASM = 0x0C
    OP_QUIT            = 0xFF

    def __init__(self, binary: Path | str | None = None) -> None:
        binary = Path(binary) if binary else HARNESS_BIN
        if not binary.exists():
            raise FileNotFoundError(
                f"Harness binary not found at {binary}. Run `make harness` first."
            )
        self.p = subprocess.Popen(
            ["sim65", str(binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

    # ------------------------------------------------------------------ IO

    def _send(self, data: bytes) -> None:
        assert self.p.stdin is not None
        self.p.stdin.write(data)
        self.p.stdin.flush()

    def _recv(self, n: int) -> bytes:
        assert self.p.stdout is not None
        buf = bytearray()
        while len(buf) < n:
            chunk = self.p.stdout.read(n - len(buf))
            if not chunk:
                err = self.p.stderr.read().decode("utf-8", errors="replace") if self.p.stderr else ""
                raise HarnessError(
                    f"harness EOF after {len(buf)}/{n} bytes. stderr:\n{err}"
                )
            buf.extend(chunk)
        return bytes(buf)

    # ------------------------------------------------------------------ Ops

    def ping(self, data: bytes) -> bytes:
        if len(data) > 255:
            raise ValueError("ping payload max 255 bytes")
        self._send(bytes([self.OP_PING, len(data)]) + data)
        return self._recv(len(data))

    def shift_sat_int8(self, acc: int, shift: int) -> int:
        """Calls C shift_sat_int8(acc int16, shift u8) -> int8."""
        if not -32768 <= acc <= 32767:
            raise ValueError(f"acc {acc} out of int16 range")
        if not 0 <= shift <= 255:
            raise ValueError(f"shift {shift} out of u8 range")
        msg = bytes([self.OP_SHIFT_SAT_INT8]) + struct.pack("<h", acc) + bytes([shift])
        self._send(msg)
        (result,) = struct.unpack("<b", self._recv(1))
        return result

    def embedding_lookup(
        self,
        table: np.ndarray,    # int8 (vocab, C)
        token_id: int,
    ) -> np.ndarray:           # int8 (C,)
        vocab, Cch = table.shape
        if table.dtype != np.int8:
            raise ValueError("table must be int8")
        msg = bytes([self.OP_EMBED_LOOKUP, vocab, Cch, token_id]) + table.tobytes()
        self._send(msg)
        return np.frombuffer(self._recv(Cch), dtype=np.int8).copy()

    def argmax_int16(self, values: np.ndarray) -> int:
        if values.dtype != np.int16:
            raise ValueError("values must be int16")
        n = values.size
        if n > 255:
            raise ValueError("argmax capped at 255 values")
        msg = bytes([self.OP_ARGMAX_INT16, n]) + values.astype("<i2").tobytes()
        self._send(msg)
        return self._recv(1)[0]

    def ssm_step(
        self,
        u_t: np.ndarray,        # int8 (C,)
        state: np.ndarray,      # int8 (C, S) — initial; returned with new state
        decay: np.ndarray,      # int8 (C, S)
        B: np.ndarray,          # int8 ternary (C, S)
        C_mat: np.ndarray,      # int8 ternary (C, S)
        D: np.ndarray,          # int8 (C,)
        ssm_out_shift: int,
        d_shift: int,
    ) -> tuple[np.ndarray, np.ndarray]:  # (y_t, new_state)
        Cch, S = state.shape
        if S % 4 != 0:
            raise ValueError("S must be multiple of 4")
        for arr, name in [(u_t, "u_t"), (state, "state"), (decay, "decay"),
                          (B, "B"), (C_mat, "C_mat"), (D, "D")]:
            if arr.dtype != np.int8:
                raise ValueError(f"{name} dtype must be int8, got {arr.dtype}")
        msg = bytes([self.OP_SSM_STEP, Cch, S, ssm_out_shift, d_shift])
        msg += u_t.tobytes()
        msg += state.tobytes(order="C")
        msg += decay.tobytes(order="C")
        msg += pack_ternary(B)
        msg += pack_ternary(C_mat)
        msg += D.tobytes()
        self._send(msg)
        y = np.frombuffer(self._recv(Cch), dtype=np.int8).copy()
        new_state = np.frombuffer(self._recv(Cch * S), dtype=np.int8).reshape(Cch, S).copy()
        return y, new_state

    def depthwise_conv1d_step(
        self,
        window: np.ndarray,    # int8 (K, C) — oldest at row 0, newest at K-1
        W: np.ndarray,         # int8 ternary (C, K)
        shift: int,
    ) -> np.ndarray:           # int8 (C,)
        K, Cch = window.shape
        if W.shape != (Cch, K):
            raise ValueError(f"W shape {W.shape} != ({Cch},{K})")
        if K % 4 != 0:
            raise ValueError("K must be multiple of 4")
        if window.dtype != np.int8 or W.dtype != np.int8:
            raise ValueError("dtype mismatch")
        msg = bytes([self.OP_DW_CONV1D_STEP, Cch, K, shift])
        msg += window.tobytes(order="C")
        msg += pack_ternary(W)
        self._send(msg)
        return np.frombuffer(self._recv(Cch), dtype=np.int8).copy()

    def vec_mul_shift_sat(
        self,
        a: np.ndarray,    # int8 (n,)
        b: np.ndarray,    # int8 (n,)
        shift: int,
    ) -> np.ndarray:       # int8 (n,)
        if a.shape != b.shape or a.dtype != np.int8 or b.dtype != np.int8:
            raise ValueError("a and b must be matching int8 vectors")
        n = a.size
        if n > 168:
            raise ValueError("vec_mul_shift_sat capped at 168 elements")
        msg = bytes([self.OP_VEC_MUL_SHIFT, n, shift]) + a.tobytes() + b.tobytes()
        self._send(msg)
        return np.frombuffer(self._recv(n), dtype=np.int8).copy()

    def ternary_linear(
        self,
        x: np.ndarray,        # int8 (in_f, seq)
        W: np.ndarray,        # int8 (out_f, in_f), values in {-1, 0, +1}
        bias: np.ndarray,     # int16 (out_f,)
        shift: int,
        *,
        asm: bool = False,    # use the hand-written asm implementation
    ) -> np.ndarray:           # int8 (out_f, seq)
        in_f = W.shape[1]
        out_f = W.shape[0]
        seq = x.shape[1] if x.ndim == 2 else 1
        if in_f % 4 != 0:
            raise ValueError("in_features must be multiple of 4 (ternary packing)")
        if x.shape[0] != in_f:
            raise ValueError(f"x.shape[0]={x.shape[0]} != in_f={in_f}")
        if bias.shape != (out_f,):
            raise ValueError(f"bias shape {bias.shape} != ({out_f},)")
        if x.dtype != np.int8 or W.dtype != np.int8 or bias.dtype != np.int16:
            raise ValueError("dtype mismatch")
        if asm and seq != 1:
            raise ValueError("asm version only handles seq=1")

        op = self.OP_TERNARY_LINEAR_ASM if asm else self.OP_TERNARY_LINEAR
        msg = bytes([op, in_f, out_f, seq, shift])
        msg += pack_ternary(W)
        msg += x.tobytes(order="C")
        msg += bias.astype("<i2").tobytes()
        self._send(msg)
        out_bytes = self._recv(out_f * seq)
        out = np.frombuffer(out_bytes, dtype=np.int8).reshape(out_f, seq)
        return out.copy()

    def int4_logits(
        self,
        x: np.ndarray,        # int8 (in_f, seq)
        W: np.ndarray,        # int8 (out_f, in_f), values in [-7, +7]
        shift: int,
    ) -> np.ndarray:           # int16 (out_f, seq)
        in_f = W.shape[1]
        out_f = W.shape[0]
        seq = x.shape[1] if x.ndim == 2 else 1
        if in_f % 2 != 0:
            raise ValueError("in_features must be multiple of 2 (int4 packing)")
        if x.shape[0] != in_f:
            raise ValueError(f"x.shape[0]={x.shape[0]} != in_f={in_f}")
        if x.dtype != np.int8 or W.dtype != np.int8:
            raise ValueError("dtype mismatch")

        msg = bytes([self.OP_INT4_LOGITS, in_f, out_f, seq, shift])
        msg += pack_int4(W)
        msg += x.tobytes(order="C")
        self._send(msg)
        out_bytes = self._recv(out_f * seq * 2)
        out = np.frombuffer(out_bytes, dtype="<i2").reshape(out_f, seq)
        return out.copy()

    def int4_depthwise_conv1d_step(
        self,
        window: np.ndarray,    # int8 (K, C)
        W: np.ndarray,         # int8 (C, K), values in [-7, +7]
        shift: int,
    ) -> np.ndarray:           # int8 (C,)
        K, Cch = window.shape
        if W.shape != (Cch, K):
            raise ValueError(f"W shape {W.shape} != ({Cch},{K})")
        if K % 2 != 0:
            raise ValueError("K must be multiple of 2")
        if window.dtype != np.int8 or W.dtype != np.int8:
            raise ValueError("dtype mismatch")
        msg = bytes([self.OP_INT4_DW_CONV1D, Cch, K, shift])
        msg += window.tobytes(order="C")
        msg += pack_int4(W)
        self._send(msg)
        return np.frombuffer(self._recv(Cch), dtype=np.int8).copy()

    def softmax_sample(
        self,
        logits: np.ndarray,    # int16 (n,)
        k: int,
        lut: list[int],
        rng_seed: int,
    ) -> int:
        """Calls C softmax_sample after seeding the RNG to `rng_seed`."""
        if logits.dtype != np.int16:
            raise ValueError("logits must be int16")
        n = logits.size
        if n > 255:
            raise ValueError("vocab capped at 255 entries")
        lut_size = len(lut)
        if lut_size > 255:
            raise ValueError("LUT max 255 entries")
        msg = bytes([self.OP_SOFTMAX_SAMPLE, rng_seed, n, k, lut_size])
        msg += bytes(lut)
        msg += logits.astype("<i2").tobytes()
        self._send(msg)
        return self._recv(1)[0]

    def ssm_step_int4_C(
        self,
        u_t: np.ndarray,        # int8 (C,)
        state: np.ndarray,      # int8 (C, S)
        decay: np.ndarray,      # int8 (C, S)
        B: np.ndarray,          # int8 ternary (C, S)
        C_mat: np.ndarray,      # int8 int4 (C, S), values in [-7, +7]
        D: np.ndarray,          # int8 (C,)
        ssm_out_shift: int,
        d_shift: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        Cch, S = state.shape
        if S % 4 != 0:
            raise ValueError("S must be multiple of 4 (B is ternary)")
        for arr, name in [(u_t, "u_t"), (state, "state"), (decay, "decay"),
                          (B, "B"), (C_mat, "C_mat"), (D, "D")]:
            if arr.dtype != np.int8:
                raise ValueError(f"{name} dtype must be int8, got {arr.dtype}")
        msg = bytes([self.OP_SSM_STEP_INT4_C, Cch, S, ssm_out_shift, d_shift])
        msg += u_t.tobytes()
        msg += state.tobytes(order="C")
        msg += decay.tobytes(order="C")
        msg += pack_ternary(B)
        msg += pack_int4(C_mat)
        msg += D.tobytes()
        self._send(msg)
        y = np.frombuffer(self._recv(Cch), dtype=np.int8).copy()
        new_state = np.frombuffer(self._recv(Cch * S), dtype=np.int8).reshape(Cch, S).copy()
        return y, new_state

    # ------------------------------------------------------------------ Lifecycle

    def close(self) -> None:
        try:
            self._send(bytes([self.OP_QUIT]))
            self.p.wait(timeout=2)
        except Exception:
            self.p.kill()
        finally:
            for stream in (self.p.stdin, self.p.stdout, self.p.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def __enter__(self) -> "CHarness":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
