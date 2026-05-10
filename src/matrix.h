#ifndef MATRIX_H
#define MATRIX_H

struct int_matrix {
    signed int *data;
    char height;
    char width;
};

struct char_matrix {
    signed char *data;
    char height;
    char width;
};

struct ternary_matrix {
    char *data;
    char height;
    char width;
};

/* Same shape contract as ternary_matrix, but each value is a signed 4-bit
 * integer in the range [-7, +7]. Storage: 2 values per byte, low nibble first.
 * If `width` is odd, the high nibble of the last byte in each row is zero. */
struct int4_matrix {
    char *data;
    char height;
    char width;
};

void print_int_matrix(struct int_matrix *m);
void print_char_matrix(struct char_matrix *m);
void print_ternary_matrix(struct ternary_matrix *m);
void matrix_multiply(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *z);
void hard_tanh(struct int_matrix *x, struct char_matrix *y, signed char scale);
void linear(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *tmp, struct char_matrix *z);

/* Primitive: arithmetic right-shift then saturate to int8.
 * Matches Python sat_int8(acc >> shift) — used everywhere a learned shift
 * brings an int16 accumulator back into int8 range. */
signed char shift_sat_int8(signed int acc, unsigned char shift);

/* Full BitNet linear layer:  out = sat_int8((W @ x + bias) >> shift)
 * W is ternary, x is int8, bias is int16, shift is per-layer learned.
 * `out` must be pre-allocated with shape (W->height, x->width).
 *
 * W->width may be any positive integer; the packed row size is
 * (W->width + 3) / 4 bytes, and the unused trailing nibbles are zero.
 * Reads beyond x[..., W->width-1] are bounded by `(W->width + 3) & ~3`,
 * so callers must zero-pad x's data buffer up to that index.
 */
void ternary_linear(struct ternary_matrix *W,
                    struct char_matrix    *x,
                    struct int_matrix     *bias,    /* shape (W->height, 1) — one int16 per output row */
                    unsigned char          shift,
                    struct char_matrix    *out);

/* Head linear: int8 input · int4 weight → int16 logits, no bias, no saturation.
 *   out[i] = (signed int)((W @ x)[i] >> shift)
 * The accumulator is signed long because n_embd × 127 × 7 (≈72k) can exceed
 * int16. After the shift, the result is well within int16 for the deployable
 * model (head_shift ≈ 6, so post-shift |value| ≲ 1125). We deliberately do not
 * saturate to int8: argmax needs the full int16 dynamic range so the top-k
 * masking trick (set picked slot to INT_MIN) works.
 *
 * W->width may be any positive integer; packed row size is (W->width + 1) / 2
 * bytes. The caller's x buffer must be zero-padded to (W->width + 1) & ~1.
 */
void int4_logits(struct int4_matrix *W,
                 struct char_matrix *x,
                 unsigned char       shift,
                 struct int_matrix  *out);

/* Element-wise: out[i] = sat_int8((a[i] * b[i]) >> shift). For SSM gating. */
void vec_mul_shift_sat(struct char_matrix *a,
                       struct char_matrix *b,
                       unsigned char       shift,
                       struct char_matrix *out);

/* Causal depthwise conv1d, single-step emission.
 *   window: int8 (K, C)        — last K input timesteps; oldest at row 0, newest at K-1
 *   W:      ternary (C, K)
 *   shift:  per-layer learned right-shift
 *   out:    int8 (1, C)         — one output timestep per channel
 *
 * Per channel c:  acc = sum_k W[c,k] * window[k,c];  out[c] = sat_int8(acc >> shift)
 * Width of W (K) must be a multiple of 4 for ternary packing. K=4 in our model.
 */
void depthwise_conv1d_step(struct char_matrix    *window,
                           struct ternary_matrix *W,
                           unsigned char          shift,
                           struct char_matrix    *out);

/* Same as depthwise_conv1d_step but with int4 weights ([-7, +7]). K must be
 * a multiple of 2 (we have K=4). Per-channel accumulator stays int16:
 * K=4 × 127 × 7 = 3556 ≪ 32767. */
void int4_depthwise_conv1d_step(struct char_matrix *window,
                                struct int4_matrix *W,
                                unsigned char       shift,
                                struct char_matrix *out);

/* Roll window down one step: window[0..K-2] = window[1..K-1]; window[K-1] = new_row */
void push_conv_window(struct char_matrix *window, struct char_matrix *new_row);

/* Embedding lookup: copy row `token_id` from `table` (vocab x C) to `out` (1 x C). */
void embedding_lookup(struct char_matrix *table, unsigned char token_id,
                      struct char_matrix *out);

/* Argmax over an int16 vector. Returns index of largest element (first wins on ties). */
unsigned char argmax_int16(struct int_matrix *logits);

/* Top-k sample: pick one of the top-k logits using an internal 8-bit LCG.
 * Deterministic on 6502 (no hardware RNG), but cycles through enough of the
 * top-k that the output stops collapsing into greedy loops. Mutates `logits`
 * (scratch — we mask each top entry to INT_MIN as we find it). */
unsigned char top_k_sample(struct int_matrix *logits, unsigned char k);

/* Reset the LCG seed (for reproducibility / tests). */
void rng_seed(unsigned char seed);

/* Diagonal SSM update for one timestep. Mutates `ssm_state` in place.
 *
 *   u_t:        int8 (1, C)            — input at this step
 *   ssm_state:  int8 (C, S)            — recurrent state (mutated)
 *   decay:      int8 (C, S), [0,127]   — effective decay = decay / 128
 *   B:          ternary (C, S)
 *   C_mat:      ternary (C, S)
 *   D:          int8 (1, C)
 *   ssm_out_shift, d_shift:           — learned right-shifts
 *   y_t:        int8 (1, C)            — output at this step (written)
 *
 * S must be a multiple of 4 (ternary packing). In our trained model, S=8.
 */
void ssm_step(struct char_matrix    *u_t,
              struct char_matrix    *ssm_state,
              struct char_matrix    *decay,
              struct ternary_matrix *B,
              struct ternary_matrix *C_mat,
              struct char_matrix    *D,
              unsigned char          ssm_out_shift,
              unsigned char          d_shift,
              struct char_matrix    *y_t);

/* Same as ssm_step but reads C as int4 instead of ternary. State update
 * (phase 1) is unchanged — B is still ternary. C_mat is (C, S) int4. */
void ssm_step_int4_C(struct char_matrix    *u_t,
                     struct char_matrix    *ssm_state,
                     struct char_matrix    *decay,
                     struct ternary_matrix *B,
                     struct int4_matrix    *C_mat,
                     struct char_matrix    *D,
                     unsigned char          ssm_out_shift,
                     unsigned char          d_shift,
                     struct char_matrix    *y_t);

#endif
