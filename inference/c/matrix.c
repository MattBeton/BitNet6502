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
unsigned char rng_state = 1;

void rng_seed(unsigned char seed) {
    rng_state = seed;
}

/* Softmax sample over the top-k logits via a precomputed 16-byte exp LUT.
 * See matrix.h for the wire-protocol description. */
unsigned char softmax_sample(struct int_matrix    *logits,
                             unsigned char         k,
                             const unsigned char  *exp_lut,
                             unsigned char         lut_size) {
    unsigned char top_idx[16];
    signed int    top_logit[16];      /* int16, the original logit before masking */
    unsigned char weights[16];
    unsigned int  total;              /* k * 255 ≤ 4080 fits in u16 */
    unsigned int  cumulative;
    unsigned int  r_lo, r_hi;
    unsigned int  r;
    signed int    delta;
    signed int    max_logit;
    unsigned char ii;
    unsigned char lut_max_idx;

    if (k > 16) k = 16;
    if (lut_size == 0) return argmax_int16(logits);
    lut_max_idx = lut_size - 1;

    /* 1) Top-k: argmax repeatedly, recording the original logit value before
     *    masking so we can compute deltas. */
    for (ii = 0; ii < k; ii++) {
        top_idx[ii] = argmax_int16(logits);
        top_logit[ii] = logits->data[top_idx[ii]];
        logits->data[top_idx[ii]] = -32768;
    }
    max_logit = top_logit[0];

    /* 2) weights[i] = exp_lut[clip(max_logit - top_logit[i], 0..lut_max_idx)] */
    total = 0;
    for (ii = 0; ii < k; ii++) {
        delta = max_logit - top_logit[ii];
        if (delta < 0) delta = 0;
        if (delta > (signed int)lut_max_idx) delta = (signed int)lut_max_idx;
        weights[ii] = exp_lut[delta];
        total += (unsigned int)weights[ii];
    }
    if (total == 0) return top_idx[0];   /* every weight rounded to 0 — fall back */

    /* 3) Draw 16-bit r via two LCG advances (low byte first, then high). */
    rng_state = (unsigned char)(rng_state * 75 + 74);
    r_lo = (unsigned int)rng_state;
    rng_state = (unsigned char)(rng_state * 75 + 74);
    r_hi = (unsigned int)rng_state;
    r = (r_hi << 8) | r_lo;

    /* 4) r %= total via while-subtract. With total ≤ 4080 the loop runs
     *    at most 65535 / 4080 ≈ 16 iterations. */
    while (r >= total) r -= total;

    /* 5) Cumulative-sum walk. */
    cumulative = 0;
    for (ii = 0; ii < k; ii++) {
        cumulative += (unsigned int)weights[ii];
        if (r < cumulative) return top_idx[ii];
    }
    return top_idx[k - 1];   /* safety; should not be reached */
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
