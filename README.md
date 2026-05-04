# Bitnet 6502

Goal: a language model that can run on a 6502 processor. 

The 6502 processor is an 8-bit (integer-only) processor with 32KB of addressable ROM. It was the processor in the Apple-II and the BBC Micro. The processor operates at a clock speed of 2 MHz.

This goal splits into a number of sub-requirements:

## Modelling

The modelling challenge here is creating the most efficient model given the tiny RAM space. To make the inference code simpler (removing any multiplication steps) we will use a bitnet architecture; the only allowable parameters are -1, 0, or +1 (see [1.58 Bits](https://arxiv.org/abs/2402.17764)).

This constrains us to approximately 80k parameters (20KB model parameters + inference code).

### Modelling constraints

- Residual stream should be stored in int8. After multiplication with a weight matrix, this might overflow int8 - so we should accumulate into int16 during inference. An activation function can be used to map the int16 value back into an int8 range. 
- We should use a simple activation function such as hard tanh; this is easy to compute as an if-statement in 6502-C.
- We won't be able to use any layer norm due to the integer quantized nature of the model

## Inference

Secondly, we need to write an inference engine that compiles to 6502 bytecode. We will do this in 6502 C, a variant of C that understands the 8-bit constraints of the 6502.

## Build & test

The inference engine is C compiled with the **cc65** toolchain (`cc65` → `ca65` → `ld65`). The default target is **sim65** — the 6502 CPU simulator bundled with cc65 — which runs the resulting binary directly from the command line without needing a disk image or emulator. `make run` compiles and executes in one step; output goes to stdout.

```bash
make run        # build and execute in sim65
make test       # run Python unit tests (creates .venv automatically)
make test-compare  # build C, run both C and Python, diff stdout
```

### Python/C equivalence testing

The Python reference implementation in `tests/bitnet_python/` mirrors every C operation (ternary decode, matrix multiply with saturating int8 arithmetic, ReLU, RMSNorm). Unit tests in `tests/test_*.py` verify the Python implementation against hardcoded expected values taken from C output. `make test-compare` then validates byte-for-byte that the C binary and the Python runner produce identical output.

To extend this pattern when adding a new C function:

1. Add the equivalent Python function in `tests/bitnet_python/`.
2. Run the C version with instrumented `printf` to capture its output for a known input.
3. Hardcode that output as the expected value in a pytest test, and assert the Python function matches it.
4. Add the new function's output to `test_runner.py` so `make test-compare` catches any future divergence at the program level.

