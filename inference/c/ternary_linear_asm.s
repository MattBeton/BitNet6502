; Hand-written 6502 implementation of ternary_linear, the dominant per-token
; matmul in BitNet inference. Drop-in replacement for the C version in
; matrix.c; byte-identical output (verified by test_ternary_linear_asm in
; tests/test_equivalence.py).
;
; Algorithm (same as the C version):
;   For each output row i:
;     acc = bias[i]                     ; int16
;     For each packed byte k in row i:
;       For each of the 4 nibbles (low pair first):
;         w = nibble & 0b11
;         if w == 0:        no-op
;         if w == 0b01:     acc += x[col]
;         if w == 0b10:     acc -= x[col]
;         col++
;     out[i] = sat_int8(acc >> shift)
;
; Performance choices:
;   - Copy the current W row to a fixed scratch buffer at row start so the
;     inner loop reads weights with `lda w_scratch,X` (4 cycles, no need to
;     dance Y around).
;   - Hot state (acc, pack, all pointers) lives in zero page.
;   - +/- ops inlined (no JSR/RTS — saves 12 cycles per nonzero weight).
;
; cc65 ABI: last argument (out*) arrives in A=lo / X=hi; the others are on
; cc65's parameter stack, popped via popa / popax.

        .export _ternary_linear_asm
        .import popa, popax
        .macpack longbranch     ; jcc/jcs/jeq/jne/jmi/jpl macros for long branches

; -------------------------------------------------------------------- zero page
;
; All `(zp),Y` indirect-indexed addressing requires the pointer in zero page,
; so both the user-facing struct pointers (used briefly at entry) and the
; pre-extracted data pointers live here. Total ZP footprint: 19 bytes.
;
        .segment "ZEROPAGE"
w_ptr:          .res 2          ; pointer to current ROW of packed W
x_ptr:          .res 2          ; constant: pointer to x->data
out_ptr:        .res 2          ; pointer to current cell in out->data
bias_ptr:       .res 2          ; pointer to current bias[i] (advances by 2 per row)
w_struct_p:     .res 2          ; argument-extraction temps (reused per call)
x_struct_p:     .res 2
bias_struct_p:  .res 2
out_struct_p:   .res 2
acc_lo:         .res 1          ; int16 accumulator
acc_hi:         .res 1
pack:           .res 1          ; current packed weight byte (shifted as we consume nibbles)
shift_amt:      .res 1
w_packed:       .res 1          ; bytes per packed row of W
n_rows:         .res 1
row_counter:    .res 1

; --------------------------------------------------------------- BSS scratch
        .bss
w_scratch:      .res 64         ; copy of current W row (max 64 packed bytes → width ≤ 256)

; -------------------------------------------------------------------- code
        .code

; void ternary_linear_asm(W*, x*, bias*, shift, out*)
;   W*    : sp + 6/7   (popped 4th)
;   x*    : sp + 4/5   (popped 3rd)
;   bias* : sp + 2/3   (popped 2nd)
;   shift : sp + 0     (popped 1st, single byte)
;   out*  : A/X        (last arg, in registers)

_ternary_linear_asm:
        ; -- Stash out pointer first (it's in A/X) ----
        sta     out_struct_p
        stx     out_struct_p+1

        ; -- Pop remaining args in LIFO order: shift, bias, x, W ----
        jsr     popa
        sta     shift_amt
        jsr     popax
        sta     bias_struct_p
        stx     bias_struct_p+1
        jsr     popax
        sta     x_struct_p
        stx     x_struct_p+1
        jsr     popax
        sta     w_struct_p
        stx     w_struct_p+1

        ; -- Extract W->data, W->height, W->width = (w_packed * 4 rounded down) --
        ldy     #0
        lda     (w_struct_p),y
        sta     w_ptr
        iny
        lda     (w_struct_p),y
        sta     w_ptr+1
        iny
        lda     (w_struct_p),y          ; W->height
        sta     n_rows
        iny
        lda     (w_struct_p),y          ; W->width
        clc                             ; w_packed = (width + 3) / 4
        adc     #3
        lsr
        lsr
        sta     w_packed

        ; -- Extract x->data ----
        ldy     #0
        lda     (x_struct_p),y
        sta     x_ptr
        iny
        lda     (x_struct_p),y
        sta     x_ptr+1

        ; -- Extract bias->data ----
        ldy     #0
        lda     (bias_struct_p),y
        sta     bias_ptr
        iny
        lda     (bias_struct_p),y
        sta     bias_ptr+1

        ; -- Extract out->data ----
        ldy     #0
        lda     (out_struct_p),y
        sta     out_ptr
        iny
        lda     (out_struct_p),y
        sta     out_ptr+1

        ; -- Outer row loop ----
        lda     #0
        sta     row_counter

; ============================================================================
; Per-output-row work: copy W row into scratch, init acc from bias, run inner
; loop, then shift+saturate+store.
; ============================================================================
row_loop:
        ; ---- Copy current W row to w_scratch[0 .. w_packed-1] ----
        ldy     w_packed
        dey                             ; Y = last index (w_packed - 1)
copy_row:
        lda     (w_ptr),y
        sta     w_scratch,y
        dey
        bpl     copy_row                ; until Y < 0

        ; ---- Init acc = bias[i] (int16 little-endian) ----
        ldy     #0
        lda     (bias_ptr),y
        sta     acc_lo
        iny
        lda     (bias_ptr),y
        sta     acc_hi

        ; ---- Inner loop: 4 nibble dispatches per packed byte ----
        ldy     #0                      ; Y = x column index
        ldx     #0                      ; X = packed byte index in row

byte_loop:
        lda     w_scratch,x             ; fetch packed byte
        sta     pack

; -------- nibble 0 (bits 0-1) --------
        and     #$03
        beq     n0_done                 ; 00 → skip
        cmp     #2
        beq     n0_sub
        ; 01 → add x[Y]  (fall through to add, treat invalid 11 same as add)
        .scope add0
        clc
        lda     (x_ptr),y
        bmi     neg
        adc     acc_lo
        sta     acc_lo
        bcc     done
        inc     acc_hi
        jmp     done
neg:
        adc     acc_lo
        sta     acc_lo
        bcs     done
        dec     acc_hi
done:
        .endscope
        jmp     n0_done
n0_sub:
        .scope sub0
        sec
        lda     acc_lo
        sbc     (x_ptr),y
        sta     acc_lo
        lda     (x_ptr),y
        bmi     neg
        lda     acc_hi
        sbc     #0
        sta     acc_hi
        jmp     done
neg:
        lda     acc_hi
        sbc     #$FF
        sta     acc_hi
done:
        .endscope
n0_done:
        iny

; -------- nibble 1 (bits 2-3) --------
        lda     pack
        lsr
        lsr
        and     #$03
        beq     n1_done
        cmp     #2
        beq     n1_sub
        .scope add1
        clc
        lda     (x_ptr),y
        bmi     neg
        adc     acc_lo
        sta     acc_lo
        bcc     done
        inc     acc_hi
        jmp     done
neg:
        adc     acc_lo
        sta     acc_lo
        bcs     done
        dec     acc_hi
done:
        .endscope
        jmp     n1_done
n1_sub:
        .scope sub1
        sec
        lda     acc_lo
        sbc     (x_ptr),y
        sta     acc_lo
        lda     (x_ptr),y
        bmi     neg
        lda     acc_hi
        sbc     #0
        sta     acc_hi
        jmp     done
neg:
        lda     acc_hi
        sbc     #$FF
        sta     acc_hi
done:
        .endscope
n1_done:
        iny

; -------- nibble 2 (bits 4-5) --------
        lda     pack
        lsr
        lsr
        lsr
        lsr
        and     #$03
        beq     n2_done
        cmp     #2
        beq     n2_sub
        .scope add2
        clc
        lda     (x_ptr),y
        bmi     neg
        adc     acc_lo
        sta     acc_lo
        bcc     done
        inc     acc_hi
        jmp     done
neg:
        adc     acc_lo
        sta     acc_lo
        bcs     done
        dec     acc_hi
done:
        .endscope
        jmp     n2_done
n2_sub:
        .scope sub2
        sec
        lda     acc_lo
        sbc     (x_ptr),y
        sta     acc_lo
        lda     (x_ptr),y
        bmi     neg
        lda     acc_hi
        sbc     #0
        sta     acc_hi
        jmp     done
neg:
        lda     acc_hi
        sbc     #$FF
        sta     acc_hi
done:
        .endscope
n2_done:
        iny

; -------- nibble 3 (bits 6-7) --------
        lda     pack
        lsr
        lsr
        lsr
        lsr
        lsr
        lsr                             ; A now holds just original bits 6-7 (as bits 0-1)
        beq     n3_done                 ; AND #$03 implied — bits 2-7 are already 0
        cmp     #2
        beq     n3_sub
        .scope add3
        clc
        lda     (x_ptr),y
        bmi     neg
        adc     acc_lo
        sta     acc_lo
        bcc     done
        inc     acc_hi
        jmp     done
neg:
        adc     acc_lo
        sta     acc_lo
        bcs     done
        dec     acc_hi
done:
        .endscope
        jmp     n3_done
n3_sub:
        .scope sub3
        sec
        lda     acc_lo
        sbc     (x_ptr),y
        sta     acc_lo
        lda     (x_ptr),y
        bmi     neg
        lda     acc_hi
        sbc     #0
        sta     acc_hi
        jmp     done
neg:
        lda     acc_hi
        sbc     #$FF
        sta     acc_hi
done:
        .endscope
n3_done:
        iny

        inx
        cpx     w_packed
        jcc     byte_loop               ; long branch (byte_loop is > 127 bytes back)

; ============================================================================
; End of row: arithmetic right-shift acc by shift_amt, saturate to int8, store.
; ============================================================================
        ldx     shift_amt
        beq     shift_done
shift_loop:
        lda     acc_hi
        cmp     #$80                    ; preserve sign: copy MSB into carry
        ror     acc_hi
        ror     acc_lo
        dex
        bne     shift_loop
shift_done:
        ; Saturate to [-128, 127]
        lda     acc_hi
        bmi     check_neg
        ; positive (or zero) side
        bne     clamp_pos               ; hi > 0 → overflow positive
        lda     acc_lo
        bpl     store_out               ; lo in [0..127], fits
clamp_pos:
        lda     #$7F
        bne     store_out               ; always taken
check_neg:
        cmp     #$FF
        bne     clamp_neg               ; more negative than -256
        lda     acc_lo
        bmi     store_out               ; lo in [128..255] i.e. -128..-1, fits
clamp_neg:
        lda     #$80
store_out:
        ldy     #0
        sta     (out_ptr),y

        ; -- Advance bias_ptr by 2, w_ptr by w_packed, out_ptr by 1 ----
        clc
        lda     bias_ptr
        adc     #2
        sta     bias_ptr
        bcc     bias_no_carry
        inc     bias_ptr+1
bias_no_carry:
        clc
        lda     w_ptr
        adc     w_packed
        sta     w_ptr
        bcc     w_no_carry
        inc     w_ptr+1
w_no_carry:
        inc     out_ptr
        bne     out_no_carry
        inc     out_ptr+1
out_no_carry:

        inc     row_counter
        lda     row_counter
        cmp     n_rows
        jcc     row_loop                ; long branch (row_loop is > 127 bytes back)

        rts
