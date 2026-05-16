#include <stdint.h>

#include "F.h"
#include "matrix.h"

// extern struct int_matrix rms_result;

unsigned long start;
unsigned long end;
unsigned long result;
unsigned char rms;

signed char tmp;

int32_t abs_long(int32_t value) {
    if (value < 0) {
        return -value;
    } else {
        return value;
    }
}

signed char signed_unsigned_divide_char(signed char a, unsigned char b) {
    signed char abs_a = a < 0 ? -a : a;
    signed char quotient = abs_a / b;
    return a < 0 ? -quotient : quotient;
}

unsigned long sqrt_long(unsigned long num) {
    if (num == 0 || num == 1) {
        return num;
    }

    start = 1;
    end = num;
    result = 0;

    while (start <= end) {
        unsigned long mid = start + ((end - start) >> 1);

        if (mid <= num / mid) {
            result = mid; // Update result as mid is a candidate
            start = mid + 1;
        } else {
            end = mid - 1;
        }
    }

    return result;
}

struct int_matrix *rms_norm(struct int_matrix *a) {
    long acc = 0;
    int value;  

    char i;
    for (i = 0; i < a->height; i++) {
        value = a->data[i];
        acc += value * value;
    }


    acc /= a->height;
    // printf("\n%ld", acc);
    acc = sqrt_long(abs_long(acc));
    // printf("\n%ld", acc);
    rms = (unsigned char)acc;
    // printf("\n%ld", acc);

    for (i = 0; i < a->height; i++) {
        tmp = signed_unsigned_divide_char(a->data[i], rms);
        // printf("%d %u %d \n", a->data[i], rms, tmp);
//         rms_result.data[i] = tmp;
    }
    

//    return &rms_result;
}

signed char ReLU(signed char input) {
    if (input > 0)
        return input;
    else
        return 0;
}

// ReLU is pointwise, so it acts inplace.
// For consistency we could return a pointer to the same a. But not necessary
void MatrixReLU(struct int_matrix *a) {
    char i;
    for (i = 0; i < a->height; i++) {
        a->data[i] = ReLU(a->data[i]);
    }
}

 
