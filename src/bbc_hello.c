/* Smallest possible BBC Micro program for cc65: prints a banner via OSWRCH.
 * Compiled with `-t none -C bbc.cfg`; loads at &0E00. */

#include "io.h"

static const char msg[] = "hello from cc65 on the beeb\r";

int main(void) {
    unsigned char i;
    for (i = 0; msg[i] != '\0'; i++) {
        write_char(msg[i]);
    }
    return 0;
}
