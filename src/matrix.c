#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "matrix.h"

char i, j, k, l;
signed int a;
unsigned char b;

/* Arithmetic right-shift then saturate to int8. cc65's `>>` on signed int is
 * arithmetic (sign-extending) on the 6502 target, matching Python semantics. */
signed char shift_sat_int8(signed int acc, unsigned char shift) {
    signed int shifted = acc >> shift;
    if (shifted > SCHAR_MAX) return SCHAR_MAX;
    if (shifted < SCHAR_MIN) return SCHAR_MIN;
    return (signed char)shifted;
}

void print_int_matrix(struct int_matrix *m) {
    for (i = 0; i < m->height; i++) {
        for (j = 0; j < m->width; j++) {
            printf("%6d", m->data[i * m->width + j]);
        }
        printf("\n");
    }
}

void print_char_matrix(struct char_matrix *m) {
    for (i = 0; i < m->height; i++) {
        for (j = 0; j < m->width; j++) {
            printf("%4d", m->data[i * m->width + j]);
        }
        printf("\n");
    }
}

void print_ternary_matrix(struct ternary_matrix *m) {
    if (m->width % 4 != 0) printf("Error encountered: ternary matrix width must be a multiple of 4.");

    for (i = 0; i < m->height; i++) {
        for (j = 0; j < m->width / 4; j++) {
            b = m->data[i * m->width / 4 + j];

            for (k = 0; k < 4; k++) {
                switch (b & 0b11) {
                    case 0b0:
                        printf("0  ");
                        break;
                    case 0b1:
                        printf("1  ");
                        break;
                    case 0b10:
                        printf("-1 ");
                        break;
                }
                b = b >> 2;
            }
        }
        printf("\n");
    }
}

// z := W @ x  (ternary weights, int8 input, int16 output, no activation)
void matrix_multiply(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *z) {
    if (W->width != x->height) {
        printf("Error encountered: matrix dimensions must align\n");
        printf("%dx%d, %dx%d \n", W->height, W->width, x->height, x->width);
    }

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
                        default:
                            printf("\nError encountered: unknown value in ternary");
                            printf("\ni: %d, j: %d, k: %d, l: %d\n", i, j, k, l);
                            break;
                    }

                    b = b >> 2;
                }
            }

            z->data[i * z->width + j] = a;
        }
    }
}

// Clips int16 values element-wise to [-scale, +scale] and writes to int8 output
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

// z := hard_tanh(W @ x, SCHAR_MAX): full linear layer, int8 in, int8 out
// tmp is a caller-supplied int16 scratch buffer; z must be pre-allocated to the correct shape
void linear(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *tmp, struct char_matrix *z) {
    if (z->height != W->height || z->width != x->width) {
        printf("linear: z shape %dx%d does not match expected %dx%d\n",
               z->height, z->width, W->height, x->width);
        return;
    }
    tmp->height = W->height;
    tmp->width = x->width;
    matrix_multiply(W, x, tmp);
    hard_tanh(tmp, z, SCHAR_MAX);
}

// Copies row token_id of `table` (vocab x C) into `out` (1 x C).
void embedding_lookup(struct char_matrix *table, unsigned char token_id,
                      struct char_matrix *out) {
    char C_dim = table->width;
    unsigned int row_offset = (unsigned int)token_id * C_dim;
    for (i = 0; i < C_dim; i++) {
        out->data[i] = table->data[row_offset + i];
    }
}

// Argmax over int16 vector. Returns index of largest element (first wins on ties).
unsigned char argmax_int16(struct int_matrix *logits) {
    unsigned int n = (unsigned int)logits->height * (unsigned int)logits->width;
    unsigned int idx;
    unsigned char best_idx = 0;
    signed int best_val = logits->data[0];
    for (idx = 1; idx < n; idx++) {
        if (logits->data[idx] > best_val) {
            best_val = logits->data[idx];
            best_idx = (unsigned char)idx;
        }
    }
    return best_idx;
}

/* 8-bit LCG: rng = rng * 75 + 74 (mod 256). Period is the full 256.
 * Constants give acceptable spectral properties for tiny generators.
 * Seed of 1 is conventional. */
static unsigned char rng_state = 1;

void rng_seed(unsigned char seed) {
    rng_state = seed;
}

unsigned char top_k_sample(struct int_matrix *logits, unsigned char k) {
    unsigned char top_idx[16];        /* k <= 16 for our use; we only need 8 */
    unsigned char i;
    unsigned char chosen;

    if (k > 16) k = 16;

    /* Repeated argmax with masking. After each pick we set the picked slot
     * to INT_MIN so the next argmax finds the next-largest. */
    for (i = 0; i < k; i++) {
        top_idx[i] = argmax_int16(logits);
        logits->data[top_idx[i]] = -32768;
    }

    /* Advance the LCG and pick one of the top-k. */
    rng_state = (unsigned char)(rng_state * 75 + 74);
    chosen = rng_state % k;
    return top_idx[chosen];
}

// Diagonal SSM update for one timestep. Mutates ssm_state in place.
// Per channel c, state index s:
//   decayed = (decay[c,s] * state[c,s]) >> 7
//   b_u     = ternary(B[c,s]) * u_t[c]
//   state[c,s] = sat_int8(decayed + b_u)
// Then per channel c:
//   c_state[c] = sum_s ternary(C[c,s]) * state[c,s]
//   c_part = sat_int8(c_state[c] >> ssm_out_shift)
//   d_part = sat_int8((D[c] * u_t[c]) >> d_shift)
//   y_t[c] = sat_int8(c_part + d_part)
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

    // Phase 1: state update — for each (c, s)
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

    // Phase 2: output — per channel c
    for (i = 0; i < C_dim; i++) {
        u_val = (signed int)u_t->data[i];

        // c_state[c] = sum_s ternary(C[c,s]) * state[c,s]
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
        // c_part = sat_int8(c_state >> ssm_out_shift)
        // d_part = sat_int8((D[c] * u_t[c]) >> d_shift)
        // y_t[c] = sat_int8(c_part + d_part)
        prod = (signed int)D->data[i] * u_val;
        d_part_v = shift_sat_int8(prod, d_shift);
        a = (signed int)shift_sat_int8(a, ssm_out_shift) + (signed int)d_part_v;
        if (a > SCHAR_MAX) a = SCHAR_MAX;
        else if (a < SCHAR_MIN) a = SCHAR_MIN;
        y_t->data[i] = (signed char)a;
    }
}

// Causal depthwise conv1d, one timestep emission.
//   window: int8 (K, C); newest input at row K-1
//   W:      ternary (C, K); K must be a multiple of 4 (one packed byte per row)
//   out:    int8 (1, C)
// Per channel: acc = sum_k W[c, k] * window[k, c]; out[c] = sat_int8(acc >> shift)
void depthwise_conv1d_step(struct char_matrix    *window,
                           struct ternary_matrix *W,
                           unsigned char          shift,
                           struct char_matrix    *out) {
    char K = W->width;
    char C_dim = W->height;

    /* window is shape-sensitive (we index by row); out is just C linear elements. */
    if (window->height != K || window->width != C_dim ||
        (unsigned int)out->height * (unsigned int)out->width != (unsigned int)C_dim) {
        printf("depthwise_conv1d_step: shape mismatch K=%d C=%d\n", K, C_dim);
        return;
    }

    // For each channel c
    for (i = 0; i < C_dim; i++) {
        a = 0;
        // Walk packed ternary bytes for this channel's row of W
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

// Roll window down: shift rows toward index 0, place new_row at row K-1.
// window: (K, C). new_row: (1, C) or (C,) (height treated as 1).
void push_conv_window(struct char_matrix *window, struct char_matrix *new_row) {
    char K = window->height;
    char C_dim = window->width;
    char r;
    for (r = 0; r < K - 1; r++) {
        for (i = 0; i < C_dim; i++) {
            window->data[r * C_dim + i] = window->data[(r + 1) * C_dim + i];
        }
    }
    for (i = 0; i < C_dim; i++) {
        window->data[(K - 1) * C_dim + i] = new_row->data[i];
    }
}

// out := sat_int8((a * b) >> shift), element-wise. Used for SSM gating.
// All three matrices share the same shape; iteration is over the flat array.
void vec_mul_shift_sat(struct char_matrix *a,
                       struct char_matrix *b,
                       unsigned char       shift,
                       struct char_matrix *out) {
    unsigned int n = (unsigned int)a->height * (unsigned int)a->width;
    unsigned int idx;

    if (a->height != b->height || a->width != b->width ||
        out->height != a->height || out->width != a->width) {
        printf("vec_mul_shift_sat: shape mismatch\n");
        return;
    }

    for (idx = 0; idx < n; idx++) {
        signed int prod = (signed int)a->data[idx] * (signed int)b->data[idx];
        out->data[idx] = shift_sat_int8(prod, shift);
    }
}

/* Sign-extend a 4-bit nibble (low 4 bits of byte) to a signed char in [-8, +7].
 * Trained model only uses [-7, +7] but the ALU handles -8 the same way. */
static signed char unpack_nibble(unsigned char nib) {
    signed char v = (signed char)(nib & 0x0F);
    if (v & 0x08) v = (signed char)(v | 0xF0);
    return v;
}

/* int8 input · int4 weight ([-7, +7]) → int16 logits >> shift, no bias, no sat.
 * Used for the head: fan-in is N_EMBD=81, weights up to ±7, activations up to
 * ±127, so |acc| can reach ~72k — too big for int16. Accumulator is therefore
 * signed long (32-bit on cc65). After the shift, results fit int16 comfortably
 * for the trained model (head_shift ≈ 6 → |out| ≲ 1125). We do not saturate so
 * argmax works correctly.
 *   W:    int4 (out_f, in_f).  Packed row = (in_f + 1) / 2 bytes, low nibble first.
 *   x:    int8 (in_f, seq_len). Caller must zero-pad to ((in_f + 1) & ~1) - 1.
 *   out:  int16 (out_f, seq_len) — pre-allocated.
 */
void int4_logits(struct int4_matrix *W,
                 struct char_matrix *x,
                 unsigned char       shift,
                 struct int_matrix  *out) {
    unsigned char w_packed = (W->width + 1) >> 1;   /* bytes per packed row */
    signed long   acc;
    signed int    xv;
    signed char   wv;
    unsigned char nib;

    if (W->width != x->height || out->height != W->height || out->width != x->width) {
        printf("int4_logits: shape mismatch W=%dx%d x=%dx%d out=%dx%d\n",
               W->height, W->width, x->height, x->width, out->height, out->width);
        return;
    }

    for (i = 0; i < W->height; i++) {
        for (j = 0; j < x->width; j++) {
            acc = 0;

            for (k = 0; k < w_packed; k++) {
                b = W->data[i * w_packed + k];

                /* Low nibble: column index 2*k */
                nib = b & 0x0F;
                wv = unpack_nibble(nib);
                if (wv) {
                    xv = (signed int)x->data[j + x->width * (2 * k)];
                    acc += (signed long)wv * (signed long)xv;
                }

                /* High nibble: column index 2*k + 1 (zero-padded if odd width) */
                nib = (b >> 4) & 0x0F;
                wv = unpack_nibble(nib);
                if (wv) {
                    xv = (signed int)x->data[j + x->width * (2 * k + 1)];
                    acc += (signed long)wv * (signed long)xv;
                }
            }

            out->data[i * out->width + j] = (signed int)(acc >> shift);
        }
    }
}

/* Causal depthwise conv1d, int4 kernel. Per channel:
 *   acc = sum_k W[c, k] * window[k, c];   out[c] = sat_int8(acc >> shift)
 * K must be a multiple of 2; the trained model has K=4 (one int4 byte = two
 * weights, two bytes per row). int16 accumulator fits comfortably:
 * 4 × 127 × 7 = 3556 ≪ 32767.
 */
void int4_depthwise_conv1d_step(struct char_matrix *window,
                                struct int4_matrix *W,
                                unsigned char       shift,
                                struct char_matrix *out) {
    char K     = W->width;
    char C_dim = W->height;
    unsigned char w_packed = (K + 1) >> 1;
    signed char wv;
    signed int  wval;

    if (window->height != K || window->width != C_dim ||
        (unsigned int)out->height * (unsigned int)out->width != (unsigned int)C_dim) {
        printf("int4_depthwise_conv1d_step: shape mismatch K=%d C=%d\n", K, C_dim);
        return;
    }

    for (i = 0; i < C_dim; i++) {
        a = 0;
        for (k = 0; k < w_packed; k++) {
            b = W->data[i * w_packed + k];

            wv = unpack_nibble(b & 0x0F);
            if (wv) {
                wval = (signed int)window->data[(2 * k) * C_dim + i];
                a += (signed int)wv * wval;
            }
            wv = unpack_nibble((b >> 4) & 0x0F);
            if (wv) {
                wval = (signed int)window->data[(2 * k + 1) * C_dim + i];
                a += (signed int)wv * wval;
            }
        }
        out->data[i] = shift_sat_int8(a, shift);
    }
}

/* Diagonal SSM with int4 C readout. Phase 1 (state update) is identical to
 * ssm_step — B is still ternary. Phase 2 reads C as int4 nibbles instead.
 * S must be a multiple of 2 (we have S=8). int16 accumulator fits:
 * 8 × 127 × 7 = 7112 ≪ 32767.
 */
void ssm_step_int4_C(struct char_matrix    *u_t,
                     struct char_matrix    *ssm_state,
                     struct char_matrix    *decay,
                     struct ternary_matrix *B,
                     struct int4_matrix    *C_mat,
                     struct char_matrix    *D,
                     unsigned char          ssm_out_shift,
                     unsigned char          d_shift,
                     struct char_matrix    *y_t) {
    char C_dim = ssm_state->height;
    char S = ssm_state->width;
    char S_pack_t = S / 4;          /* B is ternary: 4 vals/byte */
    char S_pack_4 = (S + 1) >> 1;   /* C is int4:    2 vals/byte */
    signed int prod;
    signed int u_val;
    signed char d_part_v;
    signed char wv;

    /* Phase 1: state update (same as ssm_step, B ternary) */
    for (i = 0; i < C_dim; i++) {
        u_val = (signed int)u_t->data[i];
        for (k = 0; k < S_pack_t; k++) {
            b = B->data[i * S_pack_t + k];
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

    /* Phase 2: output — c_state[c] = sum_s int4(C[c,s]) * state[c,s] */
    for (i = 0; i < C_dim; i++) {
        u_val = (signed int)u_t->data[i];

        a = 0;
        for (k = 0; k < S_pack_4; k++) {
            b = C_mat->data[i * S_pack_4 + k];

            wv = unpack_nibble(b & 0x0F);
            if (wv) {
                signed int sv = (signed int)ssm_state->data[i * S + (2 * k)];
                a += (signed int)wv * sv;
            }
            wv = unpack_nibble((b >> 4) & 0x0F);
            if (wv) {
                signed int sv = (signed int)ssm_state->data[i * S + (2 * k + 1)];
                a += (signed int)wv * sv;
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

// out := sat_int8((W @ x + bias) >> shift)
//   W:    ternary (out_f, in_f). in_f may be any positive integer; packed row
//         size is (in_f + 3) / 4 bytes with the trailing 1..3 nibbles zero.
//   x:    int8    (in_f, seq_len)  — column-by-column. Caller must zero-pad
//         the data buffer up to index ((in_f + 3) & ~3) - 1.
//   bias: int16   (out_f,)         — one int16 per output row, broadcast across seq_len
//   shift: per-layer learned right-shift amount
//   out:  int8    (out_f, seq_len) — pre-allocated by caller
//
// Inner loop accumulates into a signed int (int16 on cc65), tested below to fit.
void ternary_linear(struct ternary_matrix *W,
                    struct char_matrix    *x,
                    struct int_matrix     *bias,
                    unsigned char          shift,
                    struct char_matrix    *out) {
    unsigned char w_packed = (W->width + 3) >> 2;   /* bytes per packed row */

    if (W->width != x->height || out->height != W->height || out->width != x->width) {
        printf("ternary_linear: shape mismatch W=%dx%d x=%dx%d out=%dx%d\n",
               W->height, W->width, x->height, x->width, out->height, out->width);
        return;
    }

    for (i = 0; i < W->height; i++) {
        for (j = 0; j < x->width; j++) {
            // Initialise with bias (broadcast across the sequence dimension)
            a = bias->data[i];

            for (k = 0; k < w_packed; k++) {
                b = W->data[i * w_packed + k];

                for (l = 0; l < 4; l++) {
                    switch (b & 0b11) {
                        case 0b00:
                            break;
                        case 0b01:
                            a += x->data[j + x->width * (4 * k + l)];
                            break;
                        case 0b10:
                            a -= x->data[j + x->width * (4 * k + l)];
                            break;
                    }
                    b = b >> 2;
                }
            }

            out->data[i * out->width + j] = shift_sat_int8(a, shift);
        }
    }
}
