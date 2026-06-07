"""
Pytest fixtures for BitNet6502 testing.

Contains the test data matching inference/c/matrix_const.c
"""

import pytest
import numpy as np


# Constants from matrix_const.c
RESIDUAL_LEN = 8
HIDDEN_LEN = 8
MAX_SEQ_LEN = 256


# Weight data from matrix_const.c (in binary literal form)
# WQ_data: 8 rows x 8 columns = 16 bytes (2 bytes per row)
WQ_DATA = bytes([
    0b01010101, 0b00011010,
    0b01010101, 0b01011010,
    0b01010101, 0b10011010,
    0b01010101, 0b00011010,
    0b01010101, 0b10011010,
    0b01010101, 0b00011010,
    0b01010101, 0b00011010,
    0b01010101, 0b00011010,
])

WK_DATA = bytes([
    0b01010101, 0b00011010,
    0b01010101, 0b01011010,
    0b01010101, 0b10011010,
    0b01010101, 0b00011010,
    0b01010101, 0b10011010,
    0b01010101, 0b00011010,
    0b01010101, 0b00011010,
    0b01010101, 0b00011010,
])

WV_DATA = bytes([
    0b01010101, 0b00011010,
    0b01010101, 0b01011010,
    0b01010101, 0b10011010,
    0b01010101, 0b00011010,
    0b01010101, 0b10011010,
    0b01010101, 0b00011010,
    0b01010101, 0b00011010,
    0b01010101, 0b00011010,
])

# Test input data from matrix_const.c
# y_data is stored row-major: y_data[row * width + col]
# C array: { 125, 14, 26, -23, -1, 12, 14, -123, 1, 14, -26, -3, -1, 12, 112, -13 }
# With height=8, width=2:
#   Row 0: y_data[0], y_data[1] = 125, 14
#   Row 1: y_data[2], y_data[3] = 26, -23
#   ...
Y_DATA = np.array([
    [125, 14],
    [26, -23],
    [-1, 12],
    [14, -123],
    [1, 14],
    [-26, -3],
    [-1, 12],
    [112, -13],
], dtype=np.int8)


@pytest.fixture
def wq():
    """Query weight matrix."""
    return (WQ_DATA, RESIDUAL_LEN, HIDDEN_LEN)


@pytest.fixture
def wk():
    """Key weight matrix."""
    return (WK_DATA, RESIDUAL_LEN, HIDDEN_LEN)


@pytest.fixture
def wv():
    """Value weight matrix."""
    return (WV_DATA, RESIDUAL_LEN, HIDDEN_LEN)


@pytest.fixture
def test_input():
    """Test input matrix y (8x2)."""
    return Y_DATA.copy()


@pytest.fixture
def residual_len():
    """Residual/hidden dimension length."""
    return RESIDUAL_LEN


@pytest.fixture
def hidden_len():
    """Hidden layer length."""
    return HIDDEN_LEN


# Expected Q output from running the C code
# This will be filled in after running the C implementation
@pytest.fixture
def expected_q_output():
    """Expected Q matrix output from C implementation.

    Run `make run` and capture the output to get these values.
    Format: 8 rows x 2 columns matching the C printf output.
    """
    # These values are from running the C code
    # The C output shows:
    #  127-119
    #  127-128
    #   76-106
    #  127-119
    #   76-106
    #  127-119
    #  127-119
    #  127-119
    return np.array([
        [127, -119],
        [127, -128],
        [76, -106],
        [127, -119],
        [76, -106],
        [127, -119],
        [127, -119],
        [127, -119],
    ], dtype=np.int8)
