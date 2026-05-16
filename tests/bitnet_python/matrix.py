"""
Matrix operations for BitNet.

Implements ternary matrix multiplication with saturating arithmetic
to match the 6502 C implementation.
"""

import numpy as np
from typing import Optional
from .ternary import decode_ternary_byte


# Constants matching C's limits.h
SCHAR_MIN = -128
SCHAR_MAX = 127


def saturate_int8(value: int) -> int:
    """Saturate a value to signed 8-bit range [-128, 127]."""
    if value > SCHAR_MAX:
        return SCHAR_MAX
    elif value < SCHAR_MIN:
        return SCHAR_MIN
    return value


def ternary_matrix_multiply(
    x_data: bytes,
    x_height: int,
    x_width: int,
    y: np.ndarray
) -> np.ndarray:
    """Multiply ternary matrix x by int matrix y with saturation.

    This matches the C implementation in matrix.c:matrix_multiply()

    Args:
        x_data: Packed ternary bytes for weight matrix
        x_height: Height of ternary matrix
        x_width: Width of ternary matrix (must be multiple of 4)
        y: Input int8 matrix of shape (y_height, y_width) where y_height == x_width

    Returns:
        Result matrix of shape (x_height, y_width) with saturated int8 values
    """
    if x_width % 4 != 0:
        raise ValueError(f"Ternary matrix width must be multiple of 4, got {x_width}")

    if x_width != y.shape[0]:
        raise ValueError(
            f"Matrix dimensions must align: x is {x_height}x{x_width}, "
            f"y is {y.shape[0]}x{y.shape[1]}"
        )

    y_width = y.shape[1]
    z = np.zeros((x_height, y_width), dtype=np.int8)

    bytes_per_row = x_width // 4

    for i in range(x_height):
        for j in range(y_width):
            # Accumulator is signed int (32-bit in C)
            a = 0

            for k in range(bytes_per_row):
                # Get the packed ternary byte
                b = x_data[i * bytes_per_row + k]
                if isinstance(b, bytes):
                    b = b[0]

                # Decode and process 4 ternary values
                for l in range(4):
                    bits = b & 0b11
                    row_idx = 4 * k + l

                    if bits == 0b01:  # +1
                        # C code: a += y->data[j + y->width * (4 * k + l)]
                        a += int(y[row_idx, j])
                    elif bits == 0b10:  # -1
                        # C code: a -= y->data[j + y->width * (4 * k + l)]
                        a -= int(y[row_idx, j])
                    # bits == 0b00 means 0, no operation

                    b = b >> 2

            # Saturate to int8 range
            z[i, j] = saturate_int8(a)

    return z


def print_int_matrix(m: np.ndarray, width: Optional[int] = None) -> str:
    """Format an int matrix matching the C output format.

    Args:
        m: 2D numpy array
        width: Optional width override (for when matrix is stored flat)

    Returns:
        Formatted string matching C's printf("%4d", ...) format
    """
    lines = []
    height, actual_width = m.shape
    if width is not None:
        actual_width = width

    for i in range(height):
        row_str = ""
        for j in range(actual_width):
            row_str += f"{m[i, j]:4d}"
        lines.append(row_str)

    return "\n".join(lines)


def print_ternary_matrix(data: bytes, height: int, width: int) -> str:
    """Format a ternary matrix matching the C output format.

    Args:
        data: Packed ternary bytes
        height: Number of rows
        width: Number of columns (must be multiple of 4)

    Returns:
        Formatted string matching C's ternary print format
    """
    if width % 4 != 0:
        return "Error encountered: ternary matrix width must be a multiple of 4."

    lines = []
    bytes_per_row = width // 4

    for i in range(height):
        row_str = ""
        for j in range(bytes_per_row):
            b = data[i * bytes_per_row + j]
            if isinstance(b, bytes):
                b = b[0]

            for _ in range(4):
                bits = b & 0b11
                if bits == 0b00:
                    row_str += "0  "
                elif bits == 0b01:
                    row_str += "1  "
                elif bits == 0b10:
                    row_str += "-1 "
                b = b >> 2

        lines.append(row_str)

    return "\n".join(lines)
