#include "matrix.h"
#include "weights.h"
#include "model.h"

/* Views into proj_buf so we can split the (N_HIDDEN, 1) in_proj output into
 * (N_EMBD, 1) u and (N_EMBD, 1) gate without copying. */
static struct char_matrix proj_u    = { 0, N_EMBD, 1 };
static struct char_matrix proj_gate = { 0, N_EMBD, 1 };

static unsigned char block_step_iter;

/* Run one timestep through one SSM block. Mutates state and writes `out`. */
void block_step(struct char_matrix   *x,
                struct block_weights *w,
                struct block_state   *s,
                struct char_matrix   *out) {
    /* 1) in_proj: x (1, N_EMBD) -> proj_buf (1, N_HIDDEN) */
    ternary_linear(w->in_proj_W, x, w->in_proj_bias, w->in_proj_shift, &proj_buf);

    /* 2) Split proj_buf into u (first N_EMBD) and gate (second N_EMBD). */
    proj_u.data    = proj_buf.data;
    proj_gate.data = proj_buf.data + N_EMBD;

    /* 3) Push u into the conv ring buffer, then emit one conv step (int4 kernel). */
    push_conv_window(s->conv_window, &proj_u);
    int4_depthwise_conv1d_step(s->conv_window, w->conv_W, w->conv_shift, &u_post_conv);

    /* 4) SSM: u_post_conv (1, C) + state -> y_buf (1, C), state mutated. C is int4. */
    ssm_step_int4_C(&u_post_conv, s->ssm_state, w->decay, w->B, w->C_mat, w->D,
                    w->ssm_out_shift, w->d_shift, &y_buf);

    /* 5) Gate: y_gated = sat_int8((y_buf * gate) >> gate_shift) */
    vec_mul_shift_sat(&y_buf, &proj_gate, w->gate_shift, &y_gated);

    /* 6) out_proj: y_gated (1, C) -> y_outproj (1, C) */
    ternary_linear(w->out_proj_W, &y_gated, w->out_proj_bias, w->out_proj_shift, &y_outproj);

    /* 7) Residual: out[i] = sat_int8(x[i] + y_outproj[i]) */
    for (block_step_iter = 0; block_step_iter < N_EMBD; block_step_iter++) {
        signed int sum = (signed int)x->data[block_step_iter]
                       + (signed int)y_outproj.data[block_step_iter];
        if (sum > 127) sum = 127;
        else if (sum < -128) sum = -128;
        out->data[block_step_iter] = (signed char)sum;
    }
}

/* Run a full LM forward step.
 *  token_id: input token (vocab id)
 *  Returns logits in the global `logits` (int16 vector of size VOCAB_SIZE).
 *  Mutates block_states[*]. */
void lm_step(unsigned char token_id) {
    unsigned char layer;

    /* Embed: copy table[token_id] into x_buf. No position embedding for this model. */
    embedding_lookup(&token_embedding, token_id, &x_buf);

    /* 3 blocks */
    for (layer = 0; layer < N_LAYER; layer++) {
        block_step(&x_buf, &blocks[layer], &block_states[layer], &x_buf);
    }

    /* Head: int4 weight, no bias. Writes int16 logits directly (no saturation):
     *   logits[v] = (head_W @ x_buf)[v] >> head_shift
     * argmax / top-k consumes the int16 vector in `logits`. */
    int4_logits(&head_W, &x_buf, head_shift, &logits);
}
