#!/usr/bin/env python3
"""
Test runner that produces output matching the C implementation.

Run this script to generate Python output for comparison with C output:
    python tests/test_runner.py

Compare with C output:
    make run > /tmp/c_output.txt
    python tests/test_runner.py > /tmp/py_output.txt
    diff /tmp/c_output.txt /tmp/py_output.txt
"""

import sys
import os

# Add tests directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from bitnet_python.attention import attention_simple
from bitnet_python.matrix import print_int_matrix
from conftest import WQ_DATA, WK_DATA, WV_DATA, RESIDUAL_LEN, HIDDEN_LEN, Y_DATA


def main():
    """Run attention and print Q matrix matching C output format."""
    # Set up weight matrices (tuple of data, height, width)
    WQ = (WQ_DATA, RESIDUAL_LEN, HIDDEN_LEN)
    WK = (WK_DATA, RESIDUAL_LEN, HIDDEN_LEN)
    WV = (WV_DATA, RESIDUAL_LEN, HIDDEN_LEN)

    # Get test input
    y = Y_DATA.copy()

    # Run attention
    Q, K, V = attention_simple(y, WQ, WK, WV)

    # Print Q matrix in same format as C
    print(print_int_matrix(Q))


if __name__ == "__main__":
    main()
