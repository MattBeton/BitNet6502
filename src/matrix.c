#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include "matrix.h"

char i, j, k, l;
signed int a;
unsigned char b;

void print_int_matrix(struct int_matrix *m) {
    for (i = 0; i < m->height; i++) {
        for (j = 0; j < m->width; j++) {
            printf("%6d", m->data[i * m->width + j]);
        }
        printf("\n");
    }
}

void print_char_matrix(struct char_matrix *m) {
    for (i = 0; i < m->height; i++) {
        for (j = 0; j < m->width; j++) {
            printf("%4d", m->data[i * m->width + j]);
        }
        printf("\n");
    }
}

void print_ternary_matrix(struct ternary_matrix *m) {
    if (m->width % 4 != 0) printf("Error encountered: ternary matrix width must be a multiple of 4.");

    for (i = 0; i < m->height; i++) {
        for (j = 0; j < m->width / 4; j++) {
            b = m->data[i * m->width / 4 + j];

            for (k = 0; k < 4; k++) {
                switch (b & 0b11) {
                    case 0b0:
                        printf("0  ");
                        break;
                    case 0b1:
                        printf("1  ");
                        break;
                    case 0b10:
                        printf("-1 ");
                        break;
                }
                b = b >> 2;
            }
        }
        printf("\n");
    }
}

// z := W @ x  (ternary weights, int8 input, int16 output, no activation)
void matrix_multiply(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *z) {
    if (W->width != x->height) {
        printf("Error encountered: matrix dimensions must align\n");
        printf("%dx%d, %dx%d \n", W->height, W->width, x->height, x->width);
    }

    for (i = 0; i < W->height; i++) {
        for (j = 0; j < x->width; j++) {
            a = 0;

            for (k = 0; k < W->width / 4; k++) {
                b = W->data[i * (W->width / 4) + k];

                for (l = 0; l < 4; l++) {
                    switch (b & 0b11) {
                        case 0b0:
                            break;
                        case 0b1:
                            a += x->data[j + x->width * (4 * k + l)];
                            break;
                        case 0b10:
                            a -= x->data[j + x->width * (4 * k + l)];
                            break;
                        default:
                            printf("\nError encountered: unknown value in ternary");
                            printf("\ni: %d, j: %d, k: %d, l: %d\n", i, j, k, l);
                            break;
                    }

                    b = b >> 2;
                }
            }

            z->data[i * z->width + j] = a;
        }
    }
}

// Clips int16 values element-wise to [-scale, +scale] and writes to int8 output
void hard_tanh(struct int_matrix *x, struct char_matrix *y, signed char scale) {
    for (i = 0; i < x->height; i++) {
        for (j = 0; j < x->width; j++) {
            a = x->data[i * x->width + j];
            if (a > scale) {
                y->data[i * y->width + j] = scale;
            } else if (a < -scale) {
                y->data[i * y->width + j] = -scale;
            } else {
                y->data[i * y->width + j] = (signed char)a;
            }
        }
    }
}

// z := hard_tanh(W @ x, SCHAR_MAX): full linear layer, int8 in, int8 out
// tmp is a caller-supplied int16 scratch buffer; z must be pre-allocated to the correct shape
void linear(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *tmp, struct char_matrix *z) {
    if (z->height != W->height || z->width != x->width) {
        printf("linear: z shape %dx%d does not match expected %dx%d\n",
               z->height, z->width, W->height, x->width);
        return;
    }
    tmp->height = W->height;
    tmp->width = x->width;
    matrix_multiply(W, x, tmp);
    hard_tanh(tmp, z, SCHAR_MAX);
}
