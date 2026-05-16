"""
Ternary encoding/decoding for BitNet weights.

Encoding scheme (2 bits per value, LSB first):
- 0b00 -> 0
- 0b01 -> +1
- 0b10 -> -1
- 0b11 -> invalid (should not occur)

4 ternary values are packed per byte, with the first value in the lowest 2 bits.
"""

import numpy as np
from typing import List, Union


def decode_ternary_byte(byte: int) -> List[int]:
    """Decode a single byte into 4 ternary values.

    Args:
        byte: Integer 0-255

    Returns:
        List of 4 ternary values (-1, 0, or 1)
    """
    values = []
    for _ in range(4):
        bits = byte & 0b11
        if bits == 0b00:
            values.append(0)
        elif bits == 0b01:
            values.append(1)
        elif bits == 0b10:
            values.append(-1)
        else:
            raise ValueError(f"Invalid ternary encoding: {bits:02b}")
        byte = byte >> 2
    return values


def encode_ternary(values: List[int]) -> bytes:
    """Encode a list of ternary values into packed bytes.

    Args:
        values: List of ternary values (-1, 0, or 1). Length must be multiple of 4.

    Returns:
        Packed bytes with 4 values per byte
    """
    if len(values) % 4 != 0:
        raise ValueError(f"Length must be multiple of 4, got {len(values)}")

    result = []
    for i in range(0, len(values), 4):
        byte = 0
        for j in range(4):
            v = values[i + j]
            if v == 0:
                bits = 0b00
            elif v == 1:
                bits = 0b01
            elif v == -1:
                bits = 0b10
            else:
                raise ValueError(f"Invalid ternary value: {v}")
            byte |= (bits << (j * 2))
        result.append(byte)
    return bytes(result)


def load_ternary_matrix(data: Union[bytes, List[int]], height: int, width: int) -> np.ndarray:
    """Load packed ternary bytes into a 2D numpy array.

    Args:
        data: Packed ternary bytes (4 values per byte)
        height: Number of rows
        width: Number of columns (must be multiple of 4)

    Returns:
        numpy array of shape (height, width) with values -1, 0, or 1
    """
    if width % 4 != 0:
        raise ValueError(f"Width must be multiple of 4, got {width}")

    bytes_per_row = width // 4
    expected_bytes = height * bytes_per_row

    if len(data) != expected_bytes:
        raise ValueError(f"Expected {expected_bytes} bytes, got {len(data)}")

    matrix = np.zeros((height, width), dtype=np.int8)

    for row in range(height):
        for byte_idx in range(bytes_per_row):
            byte = data[row * bytes_per_row + byte_idx]
            if isinstance(byte, bytes):
                byte = byte[0]
            values = decode_ternary_byte(byte)
            for k, v in enumerate(values):
                matrix[row, byte_idx * 4 + k] = v

    return matrix
