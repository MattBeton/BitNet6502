#ifndef MODEL_H
#define MODEL_H

#include "matrix.h"
#include "weights.h"

/* One timestep through one SSM block. Mutates `s` in place; writes `out`. */
void block_step(struct char_matrix   *x,
                struct block_weights *w,
                struct block_state   *s,
                struct char_matrix   *out);

/* Full LM forward step: token_id in, fills global `logits` (int16 vector). */
void lm_step(unsigned char token_id);

#endif
