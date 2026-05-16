"""
Tests for math functions.
"""

import pytest
import numpy as np
from bitnet_python.functions import (
    abs_long,
    signed_unsigned_divide,
    sqrt_long,
    relu,
    matrix_relu,
    rms_norm,
)


class TestAbsLong:
    """Tests for abs_long function."""

    def test_positive_value(self):
        assert abs_long(42) == 42

    def test_negative_value(self):
        assert abs_long(-42) == 42

    def test_zero(self):
        assert abs_long(0) == 0

    def test_large_negative(self):
        assert abs_long(-2147483648) == 2147483648


class TestSignedUnsignedDivide:
    """Tests for signed_unsigned_divide function."""

    def test_positive_divided_by_positive(self):
        assert signed_unsigned_divide(10, 3) == 3  # 10 // 3 = 3

    def test_negative_divided_by_positive(self):
        assert signed_unsigned_divide(-10, 3) == -3  # -10 // 3 = -3 (preserves sign)

    def test_zero_dividend(self):
        assert signed_unsigned_divide(0, 5) == 0

    def test_larger_divisor(self):
        assert signed_unsigned_divide(5, 10) == 0  # 5 // 10 = 0

    def test_division_by_zero_raises(self):
        with pytest.raises(ZeroDivisionError):
            signed_unsigned_divide(10, 0)


class TestSqrtLong:
    """Tests for sqrt_long function."""

    def test_sqrt_zero(self):
        assert sqrt_long(0) == 0

    def test_sqrt_one(self):
        assert sqrt_long(1) == 1

    def test_sqrt_perfect_square(self):
        assert sqrt_long(4) == 2
        assert sqrt_long(9) == 3
        assert sqrt_long(16) == 4
        assert sqrt_long(25) == 5
        assert sqrt_long(100) == 10

    def test_sqrt_non_perfect_square(self):
        # Should return floor of sqrt
        assert sqrt_long(2) == 1
        assert sqrt_long(3) == 1
        assert sqrt_long(5) == 2
        assert sqrt_long(10) == 3
        assert sqrt_long(99) == 9

    def test_sqrt_large_number(self):
        assert sqrt_long(1000000) == 1000
        assert sqrt_long(999999) == 999


class TestReLU:
    """Tests for ReLU function."""

    def test_relu_positive(self):
        assert relu(5) == 5
        assert relu(127) == 127

    def test_relu_zero(self):
        assert relu(0) == 0

    def test_relu_negative(self):
        assert relu(-5) == 0
        assert relu(-128) == 0

    def test_relu_array(self):
        arr = np.array([-5, 0, 5, -10, 10], dtype=np.int8)
        result = relu(arr)
        np.testing.assert_array_equal(result, [0, 0, 5, 0, 10])


class TestMatrixReLU:
    """Tests for matrix ReLU function."""

    def test_matrix_relu(self):
        m = np.array([
            [-5, 0, 5],
            [-10, 10, -1],
        ], dtype=np.int8)

        result = matrix_relu(m)

        expected = np.array([
            [0, 0, 5],
            [0, 10, 0],
        ], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_matrix_relu_inplace(self):
        """Verify ReLU modifies array in place."""
        m = np.array([[-5, 5]], dtype=np.int8)
        matrix_relu(m)
        assert m[0, 0] == 0
        assert m[0, 1] == 5


class TestRMSNorm:
    """Tests for RMS normalization."""

    def test_rms_norm_simple(self):
        """Test RMS norm on simple vector."""
        # Vector: [10, 10, 10, 10]
        # Sum of squares: 400
        # Mean: 100
        # Sqrt: 10
        # Normalized: [1, 1, 1, 1]
        a = np.array([10, 10, 10, 10], dtype=np.int8)
        result = rms_norm(a)

        np.testing.assert_array_equal(result, [1, 1, 1, 1])

    def test_rms_norm_mixed(self):
        """Test RMS norm with mixed signs."""
        # Vector: [6, -6]
        # Sum of squares: 36 + 36 = 72
        # Mean: 36
        # Sqrt: 6
        # Normalized: [1, -1]
        a = np.array([6, -6], dtype=np.int8)
        result = rms_norm(a)

        np.testing.assert_array_equal(result, [1, -1])

    def test_rms_norm_preserves_sign(self):
        """Verify sign is preserved after normalization."""
        # Use larger values to avoid rounding to zero
        a = np.array([100, -100, 100, -100], dtype=np.int8)
        result = rms_norm(a)

        # Check signs are preserved (or zero due to integer division)
        assert result[0] >= 0
        assert result[1] <= 0
        assert result[2] >= 0
        assert result[3] <= 0
        # At least some values should be non-zero
        assert result[0] > 0 or result[1] < 0
