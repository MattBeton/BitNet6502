struct int_matrix {
    signed int *data;
    char height;
    char width;
};

struct char_matrix {
    signed char *data;
    char height;
    char width;
};

struct ternary_matrix {
    char *data;
    char height;
    char width;
};

void print_int_matrix(struct int_matrix *m);
void print_char_matrix(struct char_matrix *m);
void print_ternary_matrix(struct ternary_matrix *m);
void matrix_multiply(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *z);
void hard_tanh(struct int_matrix *x, struct char_matrix *y, signed char scale);
void linear(struct ternary_matrix *W, struct char_matrix *x, struct int_matrix *tmp, struct char_matrix *z);
