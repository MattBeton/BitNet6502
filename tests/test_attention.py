"""
Tests for attention mechanism.
"""

import pytest
import numpy as np
from bitnet_python.attention import attention, attention_simple


class TestAttention:
    """Tests for attention mechanism."""

    def test_attention_q_output(self, wq, wk, wv, test_input, expected_q_output):
        """Test that Q output matches C implementation."""
        Q, K, V = attention_simple(test_input, wq, wk, wv)

        # Q should match expected output
        np.testing.assert_array_equal(Q, expected_q_output)

    def test_attention_output_shapes(self, wq, wk, wv, test_input):
        """Test output shapes are correct."""
        Q, K, V = attention_simple(test_input, wq, wk, wv)

        # All outputs should be (hidden_len, seq_len) = (8, 2)
        assert Q.shape == (8, 2)
        assert K.shape == (8, 2)
        assert V.shape == (8, 2)

    def test_attention_with_same_weights(self, wq, test_input):
        """When all weights are the same, Q = K = V."""
        Q, K, V = attention_simple(test_input, wq, wq, wq)

        np.testing.assert_array_equal(Q, K)
        np.testing.assert_array_equal(K, V)

    def test_attention_saturation(self, wq, wk, wv):
        """Test that attention saturates large values."""
        # Create input with large values
        large_input = np.full((8, 2), 100, dtype=np.int8)

        Q, K, V = attention_simple(large_input, wq, wk, wv)

        # Values should be saturated to [-128, 127]
        assert np.all(Q >= -128)
        assert np.all(Q <= 127)
        assert np.all(K >= -128)
        assert np.all(K <= 127)
        assert np.all(V >= -128)
        assert np.all(V <= 127)


class TestAttentionVsC:
    """Tests comparing Python attention to C implementation output."""

    def test_full_attention_matches_c(self, wq, wk, wv, test_input, expected_q_output):
        """Full end-to-end test matching C output.

        The C code outputs:
         127-119
         127-128
          76-106
         127-119
          76-106
         127-119
         127-119
         127-119
        """
        Q, K, V = attention_simple(test_input, wq, wk, wv)

        # Row by row verification
        assert Q[0, 0] == 127 and Q[0, 1] == -119
        assert Q[1, 0] == 127 and Q[1, 1] == -128
        assert Q[2, 0] == 76 and Q[2, 1] == -106
        assert Q[3, 0] == 127 and Q[3, 1] == -119
        assert Q[4, 0] == 76 and Q[4, 1] == -106
        assert Q[5, 0] == 127 and Q[5, 1] == -119
        assert Q[6, 0] == 127 and Q[6, 1] == -119
        assert Q[7, 0] == 127 and Q[7, 1] == -119
