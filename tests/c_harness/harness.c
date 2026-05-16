/*
 * sim65 test harness — reads opcodes from stdin, dispatches to ops, writes
 * results to stdout. Lets pytest A/B-test the real C inference functions
 * against the Python reference (modelling/bitnet_quant_inference.py).
 *
 * Wire protocol: every call is [op_byte][op-specific args...] → [op-specific result...]
 * All multi-byte integers are little-endian. Ternary matrices are 2-bit packed,
 * 4 values per byte, LSB first (same encoding as src/matrix_const.c).
 *
 * The harness uses pre-allocated max-size scratch buffers. No malloc.
 */
#include <stdio.h>

#include "matrix.h"

/* Op codes — keep in sync with tests/c_harness.py */
#define OP_PING             0x00
#define OP_SHIFT_SAT_INT8   0x01
#define OP_TERNARY_LINEAR   0x02
#define OP_VEC_MUL_SHIFT    0x03
#define OP_DW_CONV1D_STEP   0x04
#define OP_SSM_STEP         0x05
#define OP_EMBED_LOOKUP     0x06
#define OP_ARGMAX_INT16     0x07
#define OP_INT4_LOGITS      0x08
#define OP_INT4_DW_CONV1D   0x09
#define OP_SSM_STEP_INT4_C  0x0A
#define OP_SOFTMAX_SAMPLE   0x0B
#define OP_QUIT             0xFF

/* Maximum sizes seen by any test. n_embd=84, in_proj doubles to 168. */
#define MAX_FEAT 168
#define MAX_BYTES 4096
#define MAX_SEQ   8     /* longest sequence we'll ever stuff through ternary_linear in tests */

/* Scratch buffers reused across calls. Declared with maximum sizes. */
static unsigned char io_buf[MAX_BYTES];

/* Scratch matrices for ternary_linear and downstream ops. */
static char         W_buf[(MAX_FEAT * MAX_FEAT) / 4];   /* up to 168x168 ternary, packed */
static signed char  x_buf[MAX_FEAT * MAX_SEQ];          /* int8 input */
static signed int   bias_buf[MAX_FEAT];                 /* int16 bias */
static signed char  out_buf[MAX_FEAT * MAX_SEQ];        /* int8 output */

static struct ternary_matrix W_m   = { W_buf, 0, 0 };
static struct char_matrix    x_m   = { x_buf, 0, 0 };
static struct int_matrix     bias_m= { bias_buf, 0, 0 };
static struct char_matrix    out_m = { out_buf, 0, 0 };

/* ---- IO helpers ---- */

static unsigned char read_u8(void) {
    return (unsigned char)getchar();
}

/* Little-endian signed 16-bit read. */
static signed int read_i16(void) {
    unsigned char lo = read_u8();
    unsigned char hi = read_u8();
    return (signed int)(((unsigned int)hi << 8) | lo);
}

static void write_u8(unsigned char x) {
    putchar(x);
}

/* ---- Ops ---- */

/* Round-trip echo: read N (1 byte), then N bytes, write them back.
 * Smoke test for the wire protocol. */
static void op_ping(void) {
    unsigned char n = read_u8();
    unsigned char i;
    for (i = 0; i < n; i++) {
        io_buf[i] = read_u8();
    }
    for (i = 0; i < n; i++) {
        write_u8(io_buf[i]);
    }
}

/* Wire: [acc int16 LE][shift u8] -> [result i8].
 * Calls shift_sat_int8 and returns its single-byte output. */
static void op_shift_sat_int8(void) {
    signed int   acc   = read_i16();
    unsigned char shift = read_u8();
    signed char  out   = shift_sat_int8(acc, shift);
    write_u8((unsigned char)out);
}

static signed char b_local_buf[MAX_FEAT];
static struct char_matrix vec_a_m = { x_buf, 0, 0 };
static struct char_matrix vec_b_m = { b_local_buf, 0, 0 };
static struct char_matrix vec_o_m = { out_buf, 0, 0 };

/* Wire: [n u8][shift u8][a int8 * n][b int8 * n] -> [out int8 * n] */
static void op_vec_mul_shift(void) {
    unsigned char n     = read_u8();
    unsigned char shift = read_u8();
    unsigned int idx;

    for (idx = 0; idx < n; idx++) x_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n; idx++) b_local_buf[idx] = (signed char)read_u8();

    vec_a_m.height = 1; vec_a_m.width = n;
    vec_b_m.height = 1; vec_b_m.width = n;
    vec_o_m.height = 1; vec_o_m.width = n;
    vec_mul_shift_sat(&vec_a_m, &vec_b_m, shift, &vec_o_m);

    for (idx = 0; idx < n; idx++) write_u8((unsigned char)out_buf[idx]);
}

/* Wire: [C u8][K u8][shift u8]
 *       [window int8 (K*C) bytes, oldest first]
 *       [W ternary packed (C*K/4) bytes]
 *       -> [out int8 (C) bytes]
 */
static struct char_matrix    conv_win_m = { x_buf, 0, 0 };
static struct ternary_matrix conv_W_m   = { W_buf, 0, 0 };
static struct char_matrix    conv_out_m = { out_buf, 0, 0 };

static void op_dw_conv1d_step(void) {
    unsigned char Cch  = read_u8();
    unsigned char K    = read_u8();
    unsigned char shift= read_u8();
    unsigned int n_win = (unsigned int)Cch * K;
    unsigned int n_W   = ((unsigned int)Cch * K) >> 2;
    unsigned int idx;

    for (idx = 0; idx < n_win; idx++) x_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n_W; idx++)   W_buf[idx] = read_u8();

    conv_win_m.height = K;   conv_win_m.width = Cch;
    conv_W_m.height   = Cch; conv_W_m.width   = K;
    conv_out_m.height = 1;   conv_out_m.width = Cch;

    depthwise_conv1d_step(&conv_win_m, &conv_W_m, shift, &conv_out_m);

    for (idx = 0; idx < Cch; idx++) write_u8((unsigned char)out_buf[idx]);
}

/* Wire: [vocab u8][C u8][token_id u8][table int8 (vocab*C)] -> [out int8 (C)] */
static signed char embed_table[32 * 168];   /* vocab=27, C=84 in our model */
static struct char_matrix embed_table_m = { embed_table, 0, 0 };
static struct char_matrix embed_out_m = { x_buf, 0, 0 };

static void op_embed_lookup(void) {
    unsigned char vocab = read_u8();
    unsigned char C_dim = read_u8();
    unsigned char token_id = read_u8();
    unsigned int n = (unsigned int)vocab * C_dim;
    unsigned int idx;
    for (idx = 0; idx < n; idx++) embed_table[idx] = (signed char)read_u8();

    embed_table_m.height = vocab; embed_table_m.width = C_dim;
    embed_out_m.height = 1;       embed_out_m.width = C_dim;
    embedding_lookup(&embed_table_m, token_id, &embed_out_m);

    for (idx = 0; idx < C_dim; idx++) write_u8((unsigned char)x_buf[idx]);
}

/* Wire: [n u8][values int16 LE * n] -> [argmax_idx u8] */
static signed int argmax_buf[256];
static struct int_matrix argmax_m = { argmax_buf, 0, 0 };

static void op_argmax_int16(void) {
    unsigned char n = read_u8();
    unsigned int idx;
    for (idx = 0; idx < n; idx++) argmax_buf[idx] = read_i16();
    argmax_m.height = 1; argmax_m.width = n;
    write_u8(argmax_int16(&argmax_m));
}

/* SSM step: large inputs, separate scratch buffers. */
#define MAX_C_SSM 168     /* up to 2*n_embd just in case (in our model C=84 here) */
#define MAX_S_SSM 8

static signed char ssm_u_buf[MAX_C_SSM];
static signed char ssm_state_buf[MAX_C_SSM * MAX_S_SSM];
static signed char ssm_decay_buf[MAX_C_SSM * MAX_S_SSM];
static char        ssm_B_buf[(MAX_C_SSM * MAX_S_SSM) / 4];
static char        ssm_C_buf[(MAX_C_SSM * MAX_S_SSM) / 4];
static signed char ssm_D_buf[MAX_C_SSM];
static signed char ssm_y_buf[MAX_C_SSM];

static struct char_matrix    ssm_u_m     = { ssm_u_buf, 0, 0 };
static struct char_matrix    ssm_state_m = { ssm_state_buf, 0, 0 };
static struct char_matrix    ssm_decay_m = { ssm_decay_buf, 0, 0 };
static struct ternary_matrix ssm_B_m     = { ssm_B_buf, 0, 0 };
static struct ternary_matrix ssm_C_m     = { ssm_C_buf, 0, 0 };
static struct char_matrix    ssm_D_m     = { ssm_D_buf, 0, 0 };
static struct char_matrix    ssm_y_m     = { ssm_y_buf, 0, 0 };

/* Wire:
 *   [C u8][S u8][ssm_out_shift u8][d_shift u8]
 *   [u_t int8 (C)]
 *   [state int8 (C*S)]    — initial state
 *   [decay int8 (C*S)]
 *   [B ternary packed (C*S/4)]
 *   [C ternary packed (C*S/4)]
 *   [D int8 (C)]
 *   ->
 *   [y_t int8 (C)]
 *   [new_state int8 (C*S)]
 */
static void op_ssm_step(void) {
    unsigned char C_dim = read_u8();
    unsigned char S_dim = read_u8();
    unsigned char ssm_out_shift = read_u8();
    unsigned char d_shift = read_u8();
    unsigned int n_cs = (unsigned int)C_dim * S_dim;
    unsigned int n_packed = n_cs >> 2;
    unsigned int idx;

    for (idx = 0; idx < C_dim; idx++) ssm_u_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n_cs; idx++)  ssm_state_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n_cs; idx++)  ssm_decay_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n_packed; idx++) ssm_B_buf[idx] = read_u8();
    for (idx = 0; idx < n_packed; idx++) ssm_C_buf[idx] = read_u8();
    for (idx = 0; idx < C_dim; idx++) ssm_D_buf[idx] = (signed char)read_u8();

    ssm_u_m.height = 1;     ssm_u_m.width = C_dim;
    ssm_state_m.height = C_dim; ssm_state_m.width = S_dim;
    ssm_decay_m.height = C_dim; ssm_decay_m.width = S_dim;
    ssm_B_m.height = C_dim; ssm_B_m.width = S_dim;
    ssm_C_m.height = C_dim; ssm_C_m.width = S_dim;
    ssm_D_m.height = 1;     ssm_D_m.width = C_dim;
    ssm_y_m.height = 1;     ssm_y_m.width = C_dim;

    ssm_step(&ssm_u_m, &ssm_state_m, &ssm_decay_m, &ssm_B_m, &ssm_C_m, &ssm_D_m,
             ssm_out_shift, d_shift, &ssm_y_m);

    for (idx = 0; idx < C_dim; idx++) write_u8((unsigned char)ssm_y_buf[idx]);
    for (idx = 0; idx < n_cs; idx++)  write_u8((unsigned char)ssm_state_buf[idx]);
}

/* Wire: [in_f u8][out_f u8][seq u8][shift u8]
 *       [W_packed (out_f*in_f/4) bytes]
 *       [x int8 (in_f*seq) bytes]
 *       [bias int16 LE (out_f*2) bytes]
 *       -> [out int8 (out_f*seq) bytes]
 */
static void op_ternary_linear(void) {
    unsigned char in_f  = read_u8();
    unsigned char out_f = read_u8();
    unsigned char seq   = read_u8();
    unsigned char shift = read_u8();
    unsigned int n_w_bytes = ((unsigned int)out_f * in_f) >> 2;
    unsigned int n_x_bytes = (unsigned int)in_f * seq;
    unsigned int n_out_bytes = (unsigned int)out_f * seq;
    unsigned int idx;

    for (idx = 0; idx < n_w_bytes; idx++) W_buf[idx] = read_u8();
    for (idx = 0; idx < n_x_bytes; idx++) x_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < out_f; idx++)     bias_buf[idx] = read_i16();

    W_m.data = W_buf;     W_m.height = out_f; W_m.width = in_f;
    x_m.data = x_buf;     x_m.height = in_f;  x_m.width = seq;
    bias_m.data = bias_buf; bias_m.height = out_f; bias_m.width = 1;
    out_m.data = out_buf; out_m.height = out_f; out_m.width = seq;

    ternary_linear(&W_m, &x_m, &bias_m, shift, &out_m);

    for (idx = 0; idx < n_out_bytes; idx++) write_u8((unsigned char)out_buf[idx]);
}

/* ---- int4 ops ---- */

/* Wire: [in_f u8][out_f u8][seq u8][shift u8]
 *       [W_packed (out_f*in_f/2) bytes]
 *       [x int8 (in_f*seq) bytes]
 *       -> [out int16 LE (out_f*seq) bytes]
 */
static signed int int4_logits_buf[MAX_FEAT];
static struct int4_matrix int4_W_m = { W_buf, 0, 0 };
static struct int_matrix  int4_out_m = { int4_logits_buf, 0, 0 };

static void op_int4_logits(void) {
    unsigned char in_f  = read_u8();
    unsigned char out_f = read_u8();
    unsigned char seq   = read_u8();
    unsigned char shift = read_u8();
    unsigned int n_w_bytes = ((unsigned int)out_f * in_f) >> 1;
    unsigned int n_x_bytes = (unsigned int)in_f * seq;
    unsigned int n_out = (unsigned int)out_f * seq;
    unsigned int idx;

    for (idx = 0; idx < n_w_bytes; idx++) W_buf[idx] = read_u8();
    for (idx = 0; idx < n_x_bytes; idx++) x_buf[idx] = (signed char)read_u8();

    int4_W_m.data = W_buf;            int4_W_m.height = out_f; int4_W_m.width = in_f;
    x_m.data = x_buf;                 x_m.height = in_f;       x_m.width = seq;
    int4_out_m.data = int4_logits_buf; int4_out_m.height = out_f; int4_out_m.width = seq;

    int4_logits(&int4_W_m, &x_m, shift, &int4_out_m);

    /* Emit int16 little-endian. */
    for (idx = 0; idx < n_out; idx++) {
        signed int v = int4_logits_buf[idx];
        write_u8((unsigned char)(v & 0xFF));
        write_u8((unsigned char)((v >> 8) & 0xFF));
    }
}

/* Wire: [C u8][K u8][shift u8][window int8 (K*C)][W int4 packed (C*K/2)]
 *       -> [out int8 (C)]
 */
static struct int4_matrix int4_conv_W_m = { W_buf, 0, 0 };

static void op_int4_dw_conv1d(void) {
    unsigned char Cch  = read_u8();
    unsigned char K    = read_u8();
    unsigned char shift= read_u8();
    unsigned int n_win = (unsigned int)Cch * K;
    unsigned int n_W   = ((unsigned int)Cch * K) >> 1;
    unsigned int idx;

    for (idx = 0; idx < n_win; idx++) x_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n_W; idx++)   W_buf[idx] = read_u8();

    conv_win_m.height = K;       conv_win_m.width = Cch;
    int4_conv_W_m.data = W_buf;  int4_conv_W_m.height = Cch; int4_conv_W_m.width = K;
    conv_out_m.height = 1;       conv_out_m.width = Cch;

    int4_depthwise_conv1d_step(&conv_win_m, &int4_conv_W_m, shift, &conv_out_m);

    for (idx = 0; idx < Cch; idx++) write_u8((unsigned char)out_buf[idx]);
}

/* Wire (same layout as OP_SSM_STEP except C is int4-packed instead of ternary): */
static struct int4_matrix ssm_C_int4_m = { ssm_C_buf, 0, 0 };

static void op_ssm_step_int4_C(void) {
    unsigned char C_dim = read_u8();
    unsigned char S_dim = read_u8();
    unsigned char ssm_out_shift = read_u8();
    unsigned char d_shift = read_u8();
    unsigned int n_cs = (unsigned int)C_dim * S_dim;
    unsigned int n_packed_t = n_cs >> 2;   /* B is ternary */
    unsigned int n_packed_4 = n_cs >> 1;   /* C is int4 */
    unsigned int idx;

    for (idx = 0; idx < C_dim; idx++) ssm_u_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n_cs; idx++)  ssm_state_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n_cs; idx++)  ssm_decay_buf[idx] = (signed char)read_u8();
    for (idx = 0; idx < n_packed_t; idx++) ssm_B_buf[idx] = read_u8();
    for (idx = 0; idx < n_packed_4; idx++) ssm_C_buf[idx] = read_u8();
    for (idx = 0; idx < C_dim; idx++) ssm_D_buf[idx] = (signed char)read_u8();

    ssm_u_m.height = 1;     ssm_u_m.width = C_dim;
    ssm_state_m.height = C_dim; ssm_state_m.width = S_dim;
    ssm_decay_m.height = C_dim; ssm_decay_m.width = S_dim;
    ssm_B_m.height = C_dim; ssm_B_m.width = S_dim;
    ssm_C_int4_m.data = ssm_C_buf; ssm_C_int4_m.height = C_dim; ssm_C_int4_m.width = S_dim;
    ssm_D_m.height = 1;     ssm_D_m.width = C_dim;
    ssm_y_m.height = 1;     ssm_y_m.width = C_dim;

    ssm_step_int4_C(&ssm_u_m, &ssm_state_m, &ssm_decay_m, &ssm_B_m, &ssm_C_int4_m,
                    &ssm_D_m, ssm_out_shift, d_shift, &ssm_y_m);

    for (idx = 0; idx < C_dim; idx++) write_u8((unsigned char)ssm_y_buf[idx]);
    for (idx = 0; idx < n_cs; idx++)  write_u8((unsigned char)ssm_state_buf[idx]);
}

/* Wire: [seed u8][n u8][k u8][lut_size u8][lut bytes][logits int16 LE * n]
 *       -> [chosen_idx u8]
 * Seeds the RNG to `seed`, then calls softmax_sample(logits, k, lut, lut_size).
 */
static unsigned char softmax_lut_buf[256];

static void op_softmax_sample(void) {
    unsigned char seed     = read_u8();
    unsigned char n        = read_u8();
    unsigned char k        = read_u8();
    unsigned char lut_size = read_u8();
    unsigned int idx;

    for (idx = 0; idx < lut_size; idx++) softmax_lut_buf[idx] = read_u8();
    for (idx = 0; idx < n; idx++)        argmax_buf[idx] = read_i16();

    argmax_m.height = 1; argmax_m.width = n;
    rng_seed(seed);
    write_u8(softmax_sample(&argmax_m, k, softmax_lut_buf, lut_size));
}

/* ---- Dispatcher ---- */

int main(void) {
    unsigned char op;

    while (1) {
        op = read_u8();
        if (op == OP_QUIT) break;

        switch (op) {
            case OP_PING:
                op_ping();
                break;
            case OP_SHIFT_SAT_INT8:
                op_shift_sat_int8();
                break;
            case OP_TERNARY_LINEAR:
                op_ternary_linear();
                break;
            case OP_VEC_MUL_SHIFT:
                op_vec_mul_shift();
                break;
            case OP_DW_CONV1D_STEP:
                op_dw_conv1d_step();
                break;
            case OP_SSM_STEP:
                op_ssm_step();
                break;
            case OP_EMBED_LOOKUP:
                op_embed_lookup();
                break;
            case OP_ARGMAX_INT16:
                op_argmax_int16();
                break;
            case OP_INT4_LOGITS:
                op_int4_logits();
                break;
            case OP_INT4_DW_CONV1D:
                op_int4_dw_conv1d();
                break;
            case OP_SSM_STEP_INT4_C:
                op_ssm_step_int4_C();
                break;
            case OP_SOFTMAX_SAMPLE:
                op_softmax_sample();
                break;
            default:
                /* Unknown op: write a sentinel and quit. */
                write_u8(0xEE);
                fflush(stdout);
                return 1;
        }

        fflush(stdout);
    }

    return 0;
}
