"""Convert a raw 6502 binary into a BBC Micro tape WAV file.

BBC Acorn tape format (1200 baud Kansas City):
  - Stream of bytes encoded as: 1 start bit (0), 8 data bits (LSB first), 1 stop bit (1).
  - '0' bit = 1 cycle of 1200 Hz; '1' bit = 2 cycles of 2400 Hz.
  - File is broken into blocks of up to 256 bytes. Each block:
        sync byte  : 0x2A ('*')
        filename   : NUL-terminated, max ~10 chars
        load addr  : 4 bytes little-endian
        exec addr  : 4 bytes little-endian
        block num  : 2 bytes little-endian
        data length: 2 bytes little-endian
        block flag : 1 byte (bit 7 set = last block)
        spare      : 4 bytes (zero)
        header CRC : 2 bytes big-endian, CRC-16-CCITT (poly 0x1021, init 0)
        data       : up to 256 bytes
        data CRC   : 2 bytes big-endian
  - Long carrier tone (~5 s) prefixes the file. Short carrier (~0.4 s)
    separates blocks. Trailing carrier (~1 s).

Usage:
  python tools/make_bbc_tape.py path/to/binary \\
       --load 0x0E00 --exec 0x0E00 --name HELLO --out hello.wav
"""
from __future__ import annotations

import argparse
import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 48000   # 1200 and 2400 Hz divide it evenly — integer samples per cycle
F_LOW = 1200    # encodes a '0' bit (one cycle per bit period)
F_HIGH = 2400   # encodes a '1' bit (two cycles per bit period)
BAUD = 1200     # one bit per 1/1200 s


def crc16_ccitt(data: bytes) -> int:
    """Acorn tape CRC-16: poly 0x1021, init 0x0000, MSB-first, no reflection."""
    crc = 0
    for b in data:
        crc ^= b << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


AMPLITUDE_HIGH = 32767   # 2400 Hz ('1' bit) — full scale
AMPLITUDE_LOW  = 32767   # 1200 Hz ('0' bit) — equal (pre-emphasis disabled; was overcorrecting)


class _PhaseTone:
    """Phase-continuous tone generator. Matches PlayUEF's approach: phase is
    threaded across every generateTone() call so there are no zero-crossing
    discontinuities at bit boundaries. Marginal real-hardware audio paths
    (headphone-jack → DIN with no level-matching) only decode reliably when
    the analog signal is clean across bit transitions."""
    def __init__(self) -> None:
        self.phase = 0.0   # radians, accumulated

    def tone(self, samples: list[int], freq: float, n_cycles: float) -> None:
        n_samples = int(round(SAMPLE_RATE * n_cycles / freq))
        # Per-frequency amplitude for cheap pre-emphasis.
        amp = AMPLITUDE_HIGH if freq >= F_HIGH else AMPLITUDE_LOW
        for _ in range(n_samples):
            samples.append(int(round(amp * math.sin(self.phase))))
            self.phase += 2 * math.pi * freq / SAMPLE_RATE
        # Wrap to keep `phase` from growing unboundedly (avoids FP precision loss).
        self.phase = math.fmod(self.phase, 2 * math.pi)


def _encode_bit(gen: _PhaseTone, samples: list[int], bit: int) -> None:
    if bit:
        gen.tone(samples, F_HIGH, 2)    # '1' = 2 cycles of 2400 Hz
    else:
        gen.tone(samples, F_LOW, 1)     # '0' = 1 cycle of 1200 Hz


def _encode_byte(gen: _PhaseTone, samples: list[int], byte: int) -> None:
    """1 start bit (0), 8 data bits LSB first, 1 stop bit (1)."""
    _encode_bit(gen, samples, 0)
    for k in range(8):
        _encode_bit(gen, samples, (byte >> k) & 1)
    _encode_bit(gen, samples, 1)


def _encode_carrier(gen: _PhaseTone, samples: list[int], seconds: float) -> None:
    """Continuous 2400 Hz tone (logical '1's), no framing — block lead-in."""
    gen.tone(samples, F_HIGH, seconds * F_HIGH)


def _build_block(*,
                 filename: bytes,
                 load: int,
                 exec_addr: int,
                 block_num: int,
                 data: bytes,
                 is_last: bool) -> bytes:
    """Construct a single tape block (header + data + CRCs), excluding carrier."""
    header = bytearray()
    header.append(0x2A)                            # sync byte '*'
    header.extend(filename + b"\x00")              # filename + null
    header.extend(struct.pack("<I", load))         # load address
    header.extend(struct.pack("<I", exec_addr))    # exec address
    header.extend(struct.pack("<H", block_num))    # block number
    header.extend(struct.pack("<H", len(data)))    # data length
    header.append(0x80 if is_last else 0x00)       # block flag
    header.extend(b"\x00\x00\x00\x00")             # spare next-file address
    hdr_crc = crc16_ccitt(bytes(header[1:]))       # CRC excludes the sync byte
    header.extend(struct.pack(">H", hdr_crc))      # CRC stored big-endian

    if not data:
        return bytes(header)

    data_crc = crc16_ccitt(data)
    return bytes(header) + data + struct.pack(">H", data_crc)


def encode_file(binary: bytes, name: str, load: int, exec_addr: int) -> bytes:
    """Split `binary` into ≤256-byte blocks and return the concatenated tape byte stream."""
    if len(name) > 10:
        raise ValueError(f"filename '{name}' is longer than 10 characters")
    filename = name.encode("ascii")

    if not binary:
        return _build_block(filename=filename, load=load, exec_addr=exec_addr,
                            block_num=0, data=b"", is_last=True)

    blocks = []
    for i in range(0, len(binary), 256):
        chunk = binary[i:i + 256]
        block_num = i // 256
        is_last = (i + 256) >= len(binary)
        blocks.append(_build_block(filename=filename, load=load, exec_addr=exec_addr,
                                   block_num=block_num, data=chunk, is_last=is_last))
    return b"".join(blocks)


def render_uef(out_path: Path, binary: bytes, name: str, load: int, exec_addr: int,
               lead_cycles: int = 12000, inter_block_cycles: int = 1000) -> int:
    """Write a UEF v0.10 file. Returns total chunk-payload bytes.

    UEF is the canonical emulator-side format (JSBeeb, BeebEm, b-em all consume
    it directly). The block byte stream is the same as for WAV — only the
    serialisation around it differs.

    Chunk types used:
      0x0000  origin info (string)
      0x0110  carrier tone (high tone), payload = u16 LE cycles of 2400 Hz
      0x0100  implicit-data block, payload = raw bytes (1 start + 8 data LSB + 1 stop framing assumed)
    """
    if len(name) > 10:
        raise ValueError(f"filename '{name}' too long")
    filename = name.encode("ascii")

    def chunk(type_: int, payload: bytes) -> bytes:
        return struct.pack("<HI", type_, len(payload)) + payload

    out = bytearray()
    out.extend(b"UEF File!\x00")          # signature, NUL-terminated
    out.extend(struct.pack("<H", 10))     # version 0.10, stored as u16 LE = 10

    out.extend(chunk(0x0000, b"BitNet6502 generated\x00"))
    out.extend(chunk(0x0110, struct.pack("<H", lead_cycles)))  # lead-in carrier

    blocks = []
    if binary:
        for i in range(0, len(binary), 256):
            blocks.append((binary[i:i + 256], i // 256, (i + 256) >= len(binary)))
    else:
        blocks.append((b"", 0, True))
    for idx, (data, block_num, is_last) in enumerate(blocks):
        blk = _build_block(filename=filename, load=load, exec_addr=exec_addr,
                           block_num=block_num, data=data, is_last=is_last)
        out.extend(chunk(0x0100, blk))
        if not is_last:
            out.extend(chunk(0x0110, struct.pack("<H", inter_block_cycles)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(out))
    return len(out)


def render_wav(out_path: Path, binary: bytes, name: str, load: int, exec_addr: int,
               lead_seconds: float = 8.0, inter_block_seconds: float = 1.5,
               trail_seconds: float = 1.0) -> int:
    """Write a mono signed-16 WAV at 44.1 kHz. Returns total sample count."""
    if len(name) > 10:
        raise ValueError(f"filename '{name}' too long")
    filename = name.encode("ascii")

    samples: list[int] = []
    gen = _PhaseTone()
    _encode_carrier(gen, samples, lead_seconds)
    if binary:
        for i in range(0, len(binary), 256):
            chunk = binary[i:i + 256]
            block_num = i // 256
            is_last = (i + 256) >= len(binary)
            block = _build_block(filename=filename, load=load, exec_addr=exec_addr,
                                 block_num=block_num, data=chunk, is_last=is_last)
            for byte in block:
                _encode_byte(gen, samples, byte)
            if not is_last:
                _encode_carrier(gen, samples, inter_block_seconds)
    else:
        block = _build_block(filename=filename, load=load, exec_addr=exec_addr,
                             block_num=0, data=b"", is_last=True)
        for byte in block:
            _encode_byte(gen, samples, byte)
    _encode_carrier(gen, samples, trail_seconds)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)            # 16-bit signed
        w.setframerate(SAMPLE_RATE)
        w.writeframes(b"".join(struct.pack("<h", s) for s in samples))
    return len(samples)


def main() -> None:
    p = argparse.ArgumentParser(description="BBC Micro tape WAV encoder")
    p.add_argument("binary", type=Path, help="raw 6502 binary to wrap")
    p.add_argument("--load", type=lambda s: int(s, 0), default=0x0E00,
                   help="load address (default: 0x0E00, cassette filing system start)")
    p.add_argument("--exec", dest="exec_addr", type=lambda s: int(s, 0), default=None,
                   help="exec address (default: same as --load)")
    p.add_argument("--name", type=str, default="PROG",
                   help="filename embedded in the tape header (max 10 chars)")
    p.add_argument("--out", type=Path, required=True,
                   help="output path; format inferred from extension (.wav or .uef)")
    args = p.parse_args()
    if args.exec_addr is None:
        args.exec_addr = args.load
    bin_bytes = args.binary.read_bytes()
    print(f"binary: {len(bin_bytes):,} bytes  load=&{args.load:04X}  exec=&{args.exec_addr:04X}  name={args.name!r}")
    if args.out.suffix.lower() == ".uef":
        size = render_uef(args.out, bin_bytes, args.name, args.load, args.exec_addr)
        print(f"wrote {args.out}: {size:,} bytes (UEF)")
    else:
        samples = render_wav(args.out, bin_bytes, args.name, args.load, args.exec_addr)
        duration = samples / SAMPLE_RATE
        out_size = args.out.stat().st_size
        print(f"wrote {args.out}: {samples:,} samples ({duration:.1f}s), {out_size:,} bytes")


if __name__ == "__main__":
    main()
