#include "matrix.h"
#include "matrix_const.h"
#include "model.h"

void mlp_forward(struct char_matrix *x, struct char_matrix *out) {
    linear(&W1, x,   &tmp_buf, &h1);
    linear(&W2, &h1, &tmp_buf, &h2);
    linear(&W3, &h2, &tmp_buf, out);
}
