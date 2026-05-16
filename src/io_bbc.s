; void write_char(char c) for BBC Micro builds.
; cc65 passes a single char argument in the A register; OSWRCH ($FFEE)
; prints the byte in A to the current output stream and preserves A.
        .export _write_char

_write_char:
        jsr     $FFEE           ; OSWRCH
        rts
