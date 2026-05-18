#include "matrix.h"
#include "weights.h"
#include "model.h"
#include "io.h"

/* Trimmed to 148 generated chars so the output ends cleanly at 'yes' rather
 * than the disambiguating 'mommy and...' that follows in the byte-exact
 * sequence. With RNG_SEED 99 + the v200_dedup_stripped_v2 checkpoint the
 * generated suffix is:
 *   "tom and lily saw things lily were sad her house he heard them
 *    lily and tom said yes she saw a little girl smiled tom was so
 *    excited her mom said yes"
 */
#define N_GENERATE 148
#define TOP_K       8
#define RNG_SEED   99

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
