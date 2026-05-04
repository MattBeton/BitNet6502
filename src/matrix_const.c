#include "matrix.h"

#define INPUT_DIM  8
#define HIDDEN_DIM 8
#define OUTPUT_DIM 8
#define SEQ_LEN    2

// Layer 1 weights: HIDDEN_DIM x INPUT_DIM
char W1_data[] = {
    0b01010101, 0b00011010,
    0b01010101, 0b01011010,
    0b01010101, 0b10011010,
    0b01010101, 0b00011010,
    0b01010101, 0b10011010,
    0b01010101, 0b00011010,
    0b01010101, 0b00011010,
    0b01010101, 0b00011010,
};
struct ternary_matrix W1 = { W1_data, HIDDEN_DIM, INPUT_DIM };

// Layer 2 weights: HIDDEN_DIM x HIDDEN_DIM
char W2_data[] = {
    0b10101010, 0b00011010,
    0b10101010, 0b01011010,
    0b10101010, 0b10011010,
    0b10101010, 0b00011010,
    0b10101010, 0b10011010,
    0b10101010, 0b00011010,
    0b10101010, 0b00011010,
    0b10101010, 0b00011010,
};
struct ternary_matrix W2 = { W2_data, HIDDEN_DIM, HIDDEN_DIM };

// Layer 3 weights: OUTPUT_DIM x HIDDEN_DIM
char W3_data[] = {
    0b01010101, 0b00011010,
    0b10011010, 0b01010101,
    0b01010110, 0b10010101,
    0b10100101, 0b01101010,
    0b01011001, 0b10010110,
    0b10010110, 0b01011001,
    0b01101010, 0b10100101,
    0b10010101, 0b01010110,
};
struct ternary_matrix W3 = { W3_data, OUTPUT_DIM, HIDDEN_DIM };

// Input
signed char y_data[] = { 125, 14, 26, -23, -1, 12, 14, -123,
                            1, 14, -26,  -3, -1, 12, 112,  -13 };
struct char_matrix y = { y_data, INPUT_DIM, SEQ_LEN };

// Hidden layer activation buffers
signed char h1_data[HIDDEN_DIM * SEQ_LEN];
struct char_matrix h1 = { h1_data, HIDDEN_DIM, SEQ_LEN };

signed char h2_data[HIDDEN_DIM * SEQ_LEN];
struct char_matrix h2 = { h2_data, HIDDEN_DIM, SEQ_LEN };

// Output
signed char z_data[OUTPUT_DIM * SEQ_LEN];
struct char_matrix z = { z_data, OUTPUT_DIM, SEQ_LEN };

// int16 scratch buffer for linear — declared last so it can be resized freely
signed int tmp_data[HIDDEN_DIM * SEQ_LEN];
struct int_matrix tmp_buf = { tmp_data, 0, 0 };
