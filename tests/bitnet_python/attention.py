"""
Attention mechanism for BitNet.

Implements the attention Q/K/V projections to match the 6502 C implementation.
"""

import numpy as np
from typing import Tuple
from .matrix import ternary_matrix_multiply


def attention(
    x: np.ndarray,
    wq_data: bytes,
    wq_height: int,
    wq_width: int,
    wk_data: bytes,
    wk_height: int,
    wk_width: int,
    wv_data: bytes,
    wv_height: int,
    wv_width: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute attention Q, K, V projections.

    Matches C's attention() in model.c

    Args:
        x: Input matrix of shape (hidden_dim, seq_len)
        wq_data, wq_height, wq_width: Query weight matrix (ternary)
        wk_data, wk_height, wk_width: Key weight matrix (ternary)
        wv_data, wv_height, wv_width: Value weight matrix (ternary)

    Returns:
        Tuple of (Q, K, V) matrices, each of shape (wq_height, x.shape[1])
    """
    # Compute Q = WQ @ x
    Q = ternary_matrix_multiply(wq_data, wq_height, wq_width, x)

    # Compute K = WK @ x
    K = ternary_matrix_multiply(wk_data, wk_height, wk_width, x)

    # Compute V = WV @ x
    V = ternary_matrix_multiply(wv_data, wv_height, wv_width, x)

    return Q, K, V


def attention_simple(
    x: np.ndarray,
    WQ: Tuple[bytes, int, int],
    WK: Tuple[bytes, int, int],
    WV: Tuple[bytes, int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplified attention interface.

    Args:
        x: Input matrix
        WQ: Tuple of (data, height, width) for query weights
        WK: Tuple of (data, height, width) for key weights
        WV: Tuple of (data, height, width) for value weights

    Returns:
        Tuple of (Q, K, V) matrices
    """
    return attention(
        x,
        WQ[0], WQ[1], WQ[2],
        WK[0], WK[1], WK[2],
        WV[0], WV[1], WV[2]
    )
