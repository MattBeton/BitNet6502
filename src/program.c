#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "weights.h"
#include "model.h"

#define N_GENERATE 200
#define TOP_K       8

int main(void) {
    unsigned char tok = 0;        /* prompt: " " (token id 0) */
    unsigned int  step;

    /* Prefill: feed the prompt, then sample the first new token. */
    lm_step(tok);
    tok = top_k_sample(&logits, TOP_K);

    /* Top-k decode N_GENERATE more tokens, printing each as it's emitted. */
    for (step = 0; step < N_GENERATE; step++) {
        putchar(itos[tok]);
        lm_step(tok);
        tok = top_k_sample(&logits, TOP_K);
    }
    putchar('\n');
    return EXIT_SUCCESS;
}
