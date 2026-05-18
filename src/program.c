#include "matrix.h"
#include "weights.h"
#include "model.h"
#include "io.h"

#define N_GENERATE 200
#define TOP_K       8
/* RNG seed picked by sweeping LCG states with the v200_dedup_stripped_v2
 * checkpoint: seed 200 produces varied multi-character narrative ("tom they
 * decided the park ... hugged his dad ... bird ... dog named tom ... bear")
 * instead of collapsing to repeated phrases. */
#define RNG_SEED  200

/* Vocab stoi: ' ' -> 0, 'a'..'z' -> 1..26. */
static unsigned char stoi(char c) {
    return (c == ' ') ? 0 : (unsigned char)(c - 'a' + 1);
}

int main(void) {
    static const char prompt[] = "once upon a time ";
    unsigned char tok;
    unsigned char i;
    unsigned int  step;

    rng_seed(RNG_SEED);

    /* Prefill: feed each prompt token through the model, echoing as we go. */
    for (i = 0; prompt[i] != '\0'; i++) {
        tok = stoi(prompt[i]);
        write_char(prompt[i]);
        lm_step(tok);
    }
    tok = softmax_sample(&logits, TOP_K, exp_lut, EXP_LUT_SIZE);

    /* Softmax-sample N_GENERATE more tokens, printing each as it's emitted. */
    for (step = 0; step < N_GENERATE; step++) {
        write_char(itos[tok]);
        lm_step(tok);
        tok = softmax_sample(&logits, TOP_K, exp_lut, EXP_LUT_SIZE);
    }
    write_char('\n');
    for (;;) { }                          /* halt cleanly so BASIC can't print "Bad program" */
    return 0;
}
