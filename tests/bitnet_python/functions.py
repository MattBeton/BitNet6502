"""
Math functions for BitNet.

Implements ReLU, RMSNorm, integer sqrt, and other math operations
to match the 6502 C implementation.
"""

import numpy as np
from typing import Union


def abs_long(value: int) -> int:
    """Absolute value of a 32-bit signed integer.

    Matches C's abs_long() in F.c
    """
    if value < 0:
        return -value
    return value


def signed_unsigned_divide(a: int, b: int) -> int:
    """Divide signed char by unsigned char, preserving sign.

    Matches C's signed_unsigned_divide_char() in F.c

    Args:
        a: Signed 8-bit integer (-128 to 127)
        b: Unsigned 8-bit integer (0 to 255)

    Returns:
        Signed quotient with preserved sign
    """
    if b == 0:
        raise ZeroDivisionError("Division by zero")

    abs_a = -a if a < 0 else a
    quotient = abs_a // b
    return -quotient if a < 0 else quotient


def sqrt_long(num: int) -> int:
    """Integer square root using binary search.

    Matches C's sqrt_long() in F.c

    Args:
        num: Non-negative integer

    Returns:
        Integer square root (floor)
    """
    if num == 0 or num == 1:
        return num

    start = 1
    end = num
    result = 0

    while start <= end:
        mid = start + ((end - start) >> 1)

        if mid <= num // mid:
            result = mid
            start = mid + 1
        else:
            end = mid - 1

    return result


def relu(x: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """ReLU activation function.

    Matches C's ReLU() in F.c

    Args:
        x: Scalar or array of signed 8-bit integers

    Returns:
        max(0, x) for each element
    """
    if isinstance(x, np.ndarray):
        return np.maximum(0, x).astype(np.int8)
    return x if x > 0 else 0


def matrix_relu(a: np.ndarray) -> np.ndarray:
    """In-place ReLU on a matrix.

    Matches C's MatrixReLU() in F.c

    Note: The C version only applies ReLU to the first 'height' elements.
    This function applies it to all elements.

    Args:
        a: 2D numpy array

    Returns:
        Array with ReLU applied (also modifies in place)
    """
    np.maximum(a, 0, out=a)
    return a


def rms_norm(a: np.ndarray) -> np.ndarray:
    """RMS Normalization of a vector.

    Matches C's rms_norm() in F.c

    Note: The C implementation has issues (doesn't return result properly).
    This implementation follows the intended algorithm.

    Args:
        a: 1D or 2D numpy array (uses first dimension as height)

    Returns:
        RMS normalized array
    """
    # Flatten to 1D for computation
    height = a.shape[0]
    data = a.flatten()[:height] if a.ndim > 1 else a

    # Sum of squares
    acc = sum(int(v) * int(v) for v in data)

    # Mean
    acc = acc // height

    # Square root of absolute value
    acc = sqrt_long(abs_long(acc))

    # Cast to unsigned char (0-255)
    rms = acc & 0xFF

    if rms == 0:
        # Avoid division by zero
        return np.zeros_like(data, dtype=np.int8)

    # Normalize each element
    result = np.array(
        [signed_unsigned_divide(int(v), rms) for v in data],
        dtype=np.int8
    )

    return result
