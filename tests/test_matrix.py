"""
Tests for matrix operations.
"""

import pytest
import numpy as np
from bitnet_python.ternary import decode_ternary_byte, encode_ternary, load_ternary_matrix
from bitnet_python.matrix import ternary_matrix_multiply, print_int_matrix, saturate_int8


class TestTernaryDecoding:
    """Tests for ternary encoding/decoding."""

    def test_decode_all_zeros(self):
        """0b00000000 should decode to [0, 0, 0, 0]."""
        assert decode_ternary_byte(0b00000000) == [0, 0, 0, 0]

    def test_decode_all_ones(self):
        """0b01010101 should decode to [1, 1, 1, 1]."""
        assert decode_ternary_byte(0b01010101) == [1, 1, 1, 1]

    def test_decode_all_negative_ones(self):
        """0b10101010 should decode to [-1, -1, -1, -1]."""
        assert decode_ternary_byte(0b10101010) == [-1, -1, -1, -1]

    def test_decode_mixed(self):
        """0b00011010 should decode to [0, -1, 1, 0] (LSB first)."""
        # Bits: 00 01 10 00 -> values [0, 1, -1, 0] reading from LSB
        # Wait, let's trace through:
        # byte = 0b00011010 = 26
        # iteration 0: bits = 10 = 2 -> -1, byte >> 2 = 0b000110
        # iteration 1: bits = 10 = 2 -> -1, byte >> 2 = 0b0001
        # iteration 2: bits = 01 = 1 -> +1, byte >> 2 = 0b00
        # iteration 3: bits = 00 = 0 -> 0
        # Wait, 0b00011010 & 0b11 = 0b10 = 2
        # Then 0b00011010 >> 2 = 0b000110 = 6
        # 6 & 0b11 = 0b10 = 2
        # 6 >> 2 = 1
        # 1 & 0b11 = 1
        # 1 >> 2 = 0
        # 0 & 0b11 = 0
        # So we get [-1, -1, 1, 0]
        assert decode_ternary_byte(0b00011010) == [-1, -1, 1, 0]

    def test_encode_decode_roundtrip(self):
        """Encoding then decoding should return original values."""
        original = [1, 0, -1, 1, -1, 0, 0, 1]
        encoded = encode_ternary(original)
        decoded = []
        for b in encoded:
            decoded.extend(decode_ternary_byte(b))
        assert decoded == original


class TestLoadTernaryMatrix:
    """Tests for loading ternary matrices."""

    def test_load_simple_matrix(self):
        """Load a 2x4 ternary matrix."""
        # 2 rows, 4 cols = 2 bytes
        data = bytes([0b01010101, 0b10101010])  # [1,1,1,1], [-1,-1,-1,-1]
        matrix = load_ternary_matrix(data, 2, 4)

        assert matrix.shape == (2, 4)
        np.testing.assert_array_equal(matrix[0], [1, 1, 1, 1])
        np.testing.assert_array_equal(matrix[1], [-1, -1, -1, -1])

    def test_load_wq_matrix(self, wq):
        """Load the WQ weight matrix from fixtures."""
        data, height, width = wq
        matrix = load_ternary_matrix(data, height, width)

        assert matrix.shape == (8, 8)
        # First row: 0b01010101, 0b00011010 -> [1,1,1,1], [-1,-1,1,0]
        np.testing.assert_array_equal(matrix[0], [1, 1, 1, 1, -1, -1, 1, 0])


class TestSaturation:
    """Tests for saturating arithmetic."""

    def test_saturate_positive_overflow(self):
        """Values > 127 should saturate to 127."""
        assert saturate_int8(200) == 127
        assert saturate_int8(1000) == 127

    def test_saturate_negative_overflow(self):
        """Values < -128 should saturate to -128."""
        assert saturate_int8(-200) == -128
        assert saturate_int8(-1000) == -128

    def test_saturate_in_range(self):
        """Values in [-128, 127] should pass through."""
        assert saturate_int8(0) == 0
        assert saturate_int8(127) == 127
        assert saturate_int8(-128) == -128
        assert saturate_int8(50) == 50
        assert saturate_int8(-50) == -50


class TestMatrixMultiply:
    """Tests for ternary matrix multiplication."""

    def test_simple_multiply(self):
        """Simple 2x4 @ 4x2 multiplication."""
        # Weight matrix: 2x4, all ones
        w_data = bytes([0b01010101, 0b01010101])  # [1,1,1,1], [1,1,1,1]

        # Input: 4x2
        y = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
            [7, 8],
        ], dtype=np.int8)

        result = ternary_matrix_multiply(w_data, 2, 4, y)

        # Each output row should be sum of all input rows
        # Row 0: [1+3+5+7, 2+4+6+8] = [16, 20]
        assert result.shape == (2, 2)
        np.testing.assert_array_equal(result[0], [16, 20])
        np.testing.assert_array_equal(result[1], [16, 20])

    def test_multiply_with_negatives(self):
        """Multiplication with negative ternary weights."""
        # Weight matrix: 1x4, all -1
        w_data = bytes([0b10101010])  # [-1,-1,-1,-1]

        # Input: 4x1
        y = np.array([[1], [2], [3], [4]], dtype=np.int8)

        result = ternary_matrix_multiply(w_data, 1, 4, y)

        # Result: -(1+2+3+4) = -10
        assert result.shape == (1, 1)
        assert result[0, 0] == -10

    def test_multiply_with_saturation(self):
        """Verify saturation on overflow."""
        # Weight matrix: 1x4, all ones
        w_data = bytes([0b01010101])

        # Input: 4x1, large values that will overflow
        y = np.array([[100], [100], [100], [100]], dtype=np.int8)

        result = ternary_matrix_multiply(w_data, 1, 4, y)

        # 100*4 = 400 > 127, should saturate to 127
        assert result[0, 0] == 127

    def test_multiply_full_wq(self, wq, test_input, expected_q_output):
        """Test full WQ @ y multiplication matches expected output."""
        data, height, width = wq

        result = ternary_matrix_multiply(data, height, width, test_input)

        np.testing.assert_array_equal(result, expected_q_output)


class TestPrintIntMatrix:
    """Tests for matrix printing."""

    def test_print_format(self):
        """Check output format matches C."""
        m = np.array([
            [1, -2],
            [100, -100],
        ], dtype=np.int8)

        output = print_int_matrix(m)
        lines = output.split('\n')

        assert len(lines) == 2
        # C format: %4d for each element
        assert "   1  -2" in lines[0] or lines[0].strip() == "1  -2"
