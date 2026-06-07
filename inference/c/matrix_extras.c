/* Legacy / pre-int4 matrix kernels.
 *
 * These functions have been superseded by their int4 counterparts in matrix.c
 * (ternary_linear, int4_logits, ssm_step_int4_C, int4_depthwise_conv1d_step,
 * softmax_sample). They're kept here only so the equivalence test harness can
 * keep testing them against the Python reference; the BBC / apple2 / sim6502
 * inference builds do not compile this file. */
#include <limits.h>

#include "matrix.h"

/* Globals defined in matrix.c — reused for tight inner loops. */
extern char i, j, k, l;
extern signed int a;
extern unsigned char b;

void matrix_multiply(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *z) {
    for (i = 0; i < W->height; i++) {
        for (j = 0; j < x->width; j++) {
            a = 0;
            for (k = 0; k < W->width / 4; k++) {
                b = W->data[i * (W->width / 4) + k];
                for (l = 0; l < 4; l++) {
                    switch (b & 0b11) {
                        case 0b0:
                            break;
                        case 0b1:
                            a += x->data[j + x->width * (4 * k + l)];
                            break;
                        case 0b10:
                            a -= x->data[j + x->width * (4 * k + l)];
                            break;
                    }
                    b = b >> 2;
                }
            }
            z->data[i * z->width + j] = a;
        }
    }
}

void hard_tanh(struct int_matrix *x, struct char_matrix *y, signed char scale) {
    for (i = 0; i < x->height; i++) {
        for (j = 0; j < x->width; j++) {
            a = x->data[i * x->width + j];
            if (a > scale) {
                y->data[i * y->width + j] = scale;
            } else if (a < -scale) {
                y->data[i * y->width + j] = -scale;
            } else {
                y->data[i * y->width + j] = (signed char)a;
            }
        }
    }
}

void linear(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *tmp, struct char_matrix *z) {
    tmp->height = W->height;
    tmp->width = x->width;
    matrix_multiply(W, x, tmp);
    hard_tanh(tmp, z, SCHAR_MAX);
}

/* Old uniform top-k sampler — replaced by softmax_sample.
 * Uses the same LCG state defined in matrix.c. */
extern unsigned char rng_state;
unsigned char top_k_sample(struct int_matrix *logits, unsigned char k) {
    unsigned char top_idx[16];
    unsigned char ii;
    unsigned char chosen;

    if (k > 16) k = 16;
    for (ii = 0; ii < k; ii++) {
        top_idx[ii] = argmax_int16(logits);
        logits->data[top_idx[ii]] = -32768;
    }
    rng_state = (unsigned char)(rng_state * 75 + 74);
    chosen = rng_state % k;
    return top_idx[chosen];
}

void ssm_step(struct char_matrix    *u_t,
              struct char_matrix    *ssm_state,
              struct char_matrix    *decay,
              struct ternary_matrix *B,
              struct ternary_matrix *C_mat,
              struct char_matrix    *D,
              unsigned char          ssm_out_shift,
              unsigned char          d_shift,
              struct char_matrix    *y_t) {
    char C_dim = ssm_state->height;
    char S = ssm_state->width;
    char S_packed = S / 4;
    signed int prod;
    signed int u_val;
    signed char d_part_v;

    for (i = 0; i < C_dim; i++) {
        u_val = (signed int)u_t->data[i];
        for (k = 0; k < S_packed; k++) {
            b = B->data[i * S_packed + k];
            for (l = 0; l < 4; l++) {
                char s_idx = 4 * k + l;
                signed int dec = (signed int)decay->data[i * S + s_idx]
                               * (signed int)ssm_state->data[i * S + s_idx];
                dec = dec >> 7;
                switch (b & 0b11) {
                    case 0b00: break;
                    case 0b01: dec += u_val; break;
                    case 0b10: dec -= u_val; break;
                }
                if (dec > SCHAR_MAX) dec = SCHAR_MAX;
                else if (dec < SCHAR_MIN) dec = SCHAR_MIN;
                ssm_state->data[i * S + s_idx] = (signed char)dec;
                b = b >> 2;
            }
        }
    }

    for (i = 0; i < C_dim; i++) {
        u_val = (signed int)u_t->data[i];
        a = 0;
        for (k = 0; k < S_packed; k++) {
            b = C_mat->data[i * S_packed + k];
            for (l = 0; l < 4; l++) {
                char s_idx = 4 * k + l;
                signed int sv = (signed int)ssm_state->data[i * S + s_idx];
                switch (b & 0b11) {
                    case 0b00: break;
                    case 0b01: a += sv; break;
                    case 0b10: a -= sv; break;
                }
                b = b >> 2;
            }
        }
        prod = (signed int)D->data[i] * u_val;
        d_part_v = shift_sat_int8(prod, d_shift);
        a = (signed int)shift_sat_int8(a, ssm_out_shift) + (signed int)d_part_v;
        if (a > SCHAR_MAX) a = SCHAR_MAX;
        else if (a < SCHAR_MIN) a = SCHAR_MIN;
        y_t->data[i] = (signed char)a;
    }
}

void depthwise_conv1d_step(struct char_matrix    *window,
                           struct ternary_matrix *W,
                           unsigned char          shift,
                           struct char_matrix    *out) {
    char K = W->width;
    char C_dim = W->height;

    for (i = 0; i < C_dim; i++) {
        a = 0;
        for (k = 0; k < K / 4; k++) {
            b = W->data[i * (K / 4) + k];
            for (l = 0; l < 4; l++) {
                signed int wv = (signed int)window->data[(4 * k + l) * C_dim + i];
                switch (b & 0b11) {
                    case 0b00: break;
                    case 0b01: a += wv; break;
                    case 0b10: a -= wv; break;
                }
                b = b >> 2;
            }
        }
        out->data[i] = shift_sat_int8(a, shift);
    }
}
