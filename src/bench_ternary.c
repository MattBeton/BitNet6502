/* Cycle benchmark for ternary_linear (C) vs ternary_linear_asm (hand-written 6502).
 *
 * Runs N_ITER calls on a representative shape and exits. Total cycle count
 * is measured externally with `sim65 -c`. By choosing which variant to call
 * at compile time (BENCH_ASM=1), the same harness measures both.
 *
 * Shape is in_proj for n_embd=84 — the dominant matmul in our per-token
 * inference work, called 6× per token (in_proj+out_proj × 3 blocks).
 *
 *   make bench-c     -> build for C version
 *   make bench-asm   -> build for asm version
 */
#include <stdio.h>
#include "matrix.h"

#define IN_F     84
#define OUT_F    168
#define N_ITER   100         /* enough to amortise fixed costs */

/* Packed W: OUT_F rows × (IN_F/4) bytes per row. Static initializer; cc65 puts
 * this in DATA but the values are arbitrary — only the cycle count matters. */
static char W_data[OUT_F * (IN_F / 4)];
static struct ternary_matrix W = { W_data, OUT_F, IN_F };

static signed char x_data[IN_F];
static struct char_matrix x = { x_data, IN_F, 1 };

static signed int bias_data[OUT_F];
static struct int_matrix bias = { bias_data, OUT_F, 1 };

static signed char out_data[OUT_F];
static struct char_matrix out = { out_data, OUT_F, 1 };

int main(void) {
    unsigned int i, j;

    /* Populate inputs with deterministic non-trivial data so the inner loop
     * exercises all 3 ternary cases (skip/add/sub). The pattern doesn't
     * matter for cycle counting — only that it's not all zeros. */
    for (i = 0; i < sizeof(W_data); i++) {
        W_data[i] = (char)(0x6Du * (i + 1));     /* mix of 00/01/10 in each nibble */
    }
    for (i = 0; i < IN_F; i++) {
        x_data[i] = (signed char)(i - 42);
    }
    for (i = 0; i < OUT_F; i++) {
        bias_data[i] = (signed int)(i * 3 - 100);
    }

    /* Print baseline so the test runner can sanity-check the binary did
     * something. Output is captured by sim65 on stdout. */
    putchar('B');
    putchar('\n');

    for (j = 0; j < N_ITER; j++) {
#ifdef BENCH_ASM
        ternary_linear_asm(&W, &x, &bias, 5, &out);
#else
        ternary_linear(&W, &x, &bias, 5, &out);
#endif
    }

    /* Print the last output byte to prevent dead-code elimination. */
    putchar('E');
    putchar(out_data[0] + '0' & 0x7F);   /* just any value-dependent byte */
    putchar('\n');
    return 0;
}
