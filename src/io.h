#ifndef IO_H
#define IO_H

/* Single-character output. Target-specific implementation:
 *   - sim6502/apple2: src/io.c forwards to putchar (stdio).
 *   - BBC Micro:      src/io_bbc.s calls OSWRCH ($FFEE). */
void write_char(char c);

#endif
