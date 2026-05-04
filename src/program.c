#include <stdio.h>
#include <stdlib.h>

#include "matrix.h"
#include "matrix_const.h"
#include "model.h"

int main(void) {
    mlp_forward(&y, &z);
    print_char_matrix(&z);
    return EXIT_SUCCESS;
}
