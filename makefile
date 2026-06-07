# Compiler and assembler settings
CC65 = cc65
CA65 = ca65
LD65 = ld65

# Target platform (sim6502 for fast iteration, apple2 for Apple II)
TARGET = sim6502
LIB = $(TARGET).lib

# Source files. io.c routes write_char() to stdio; for BBC we substitute io_bbc.s.
# matrix_extras.c contains harness-only functions (matrix_multiply, ssm_step,
# depthwise_conv1d_step, etc.) — omitted from BBC build to save ~3 KB.
SRC_DIR = inference/c
C_SRC = $(SRC_DIR)/matrix.c $(SRC_DIR)/matrix_extras.c $(SRC_DIR)/weights.c $(SRC_DIR)/F.c $(SRC_DIR)/model.c $(SRC_DIR)/program.c $(SRC_DIR)/io.c
ASM_SRC =

# Intermediate assembly files
C_ASM = $(C_SRC:$(SRC_DIR)/%.c=build/%.s)

# Object files
C_OBJ = $(C_SRC:$(SRC_DIR)/%.c=build/%.o)
ASM_OBJ = $(ASM_SRC:$(SRC_DIR)/%.s=build/%.o)
OBJ = $(C_OBJ) $(ASM_OBJ)

# Output binary (different file per target to avoid overwrites)
OUTPUT = build/program.$(TARGET)

# Include directories
INCLUDES =

# Compiler and assembler flags
CFLAGS = -t $(TARGET) $(INCLUDES)
AFLAGS = -t $(TARGET)

# Linking flags
LDFLAGS = -t $(TARGET)

# Rule to build all targets
all: $(OUTPUT)

# Rule to compile C files to assembly files
build/%.s: $(SRC_DIR)/%.c
	$(CC65) $(CFLAGS) -o $@ $<

# Rule to assemble generated assembly files to object files
build/%.o: build/%.s
	$(CA65) $(AFLAGS) -o $@ $<

# Rule to assemble ASM files to object files
build/%.o: $(SRC_DIR)/%.s
	$(CA65) $(AFLAGS) -o $@ $<

# Rule to link object files into the final binary
$(OUTPUT): $(OBJ)
	$(LD65) $(LDFLAGS) -o $@ $(OBJ) $(LIB)

# Run in sim65 emulator
run: $(OUTPUT)
	sim65 $(OUTPUT)

# Build for Apple II (clean needed since object files are target-specific)
apple2:
	rm -f build/*.o build/*.s
	$(MAKE) TARGET=apple2 all

# Build for BBC Micro. cc65 has no `bbc` target/lib, but `bbc.cfg` ships
# with cc65 — combined with the `none` target it produces a flat binary
# loaded at &0E00 (cassette filing system origin). We swap io.c for
# io_bbc.s so output goes through OSWRCH ($FFEE).
#
# Uses recursive make to reuse the general pattern rules with TARGET=none,
# overriding C_SRC/ASM_SRC/LDFLAGS/OUTPUT. We also shrink __STACKSIZE__ to
# 256 bytes (default 2 KB) — our recursion is shallow and we need every
# byte of the 27 KB MAIN region.
BBC_CFG = tools/bbc.cfg
bbc:
	rm -f build/*.o build/*.s
	$(MAKE) TARGET=none \
	        C_SRC="$(SRC_DIR)/matrix.c $(SRC_DIR)/weights.c $(SRC_DIR)/F.c $(SRC_DIR)/model.c $(SRC_DIR)/program.c" \
	        ASM_SRC="$(SRC_DIR)/io_bbc.s" \
	        LDFLAGS="-C $(BBC_CFG) -D __STACKSIZE__=0x0200 -m build/program.bbc.map" \
	        LIB=none.lib \
	        OUTPUT=build/program.bbc \
	        all
	@echo "wrote build/program.bbc ($$(wc -c < build/program.bbc) bytes); load address &1900"

# Clean up build artifacts
clean:
	rm -f build/*.s
	rm -f build/*.o
	rm -f build/program.*
	rm -f build/test_harness.*

# Python virtual environment
VENV = .venv
PYTHON = $(VENV)/bin/python

# ----------------------------------------------------------------------------
# Test harness: a sim65 binary that exposes our C ops to the Python test driver
# over stdin/stdout. Lets pytest A/B-test C against the Python reference.
# ----------------------------------------------------------------------------

HARNESS_LIB_C = $(SRC_DIR)/matrix.c $(SRC_DIR)/matrix_extras.c $(SRC_DIR)/F.c
HARNESS_LIB_S = $(SRC_DIR)/ternary_linear_asm.s
HARNESS_LIB_OBJ = $(HARNESS_LIB_C:$(SRC_DIR)/%.c=build/%.o) $(HARNESS_LIB_S:$(SRC_DIR)/%.s=build/%.o)
HARNESS_OUTPUT = build/test_harness.$(TARGET)

# Compile harness.c with -I inference/c so it can find matrix.h
build/harness.s: tests/c_harness/harness.c
	$(CC65) $(CFLAGS) -I $(SRC_DIR) -o $@ $<

build/harness.o: build/harness.s
	$(CA65) $(AFLAGS) -o $@ $<

$(HARNESS_OUTPUT): $(HARNESS_LIB_OBJ) build/harness.o
	$(LD65) $(LDFLAGS) -o $@ $(HARNESS_LIB_OBJ) build/harness.o $(LIB)

harness: $(HARNESS_OUTPUT)

# Run Python unit tests (auto-builds both program and harness binaries)
test: $(VENV) $(OUTPUT) $(HARNESS_OUTPUT)
	$(VENV)/bin/pytest tests/test_equivalence.py -v

# Compare C and Python outputs
test-compare: $(OUTPUT) $(VENV)
	@echo "Running C implementation..."
	@sim65 $(OUTPUT) > /tmp/bitnet_c_output.txt
	@echo "Running Python implementation..."
	@$(PYTHON) tests/test_runner.py > /tmp/bitnet_py_output.txt
	@echo "Comparing outputs..."
	@diff /tmp/bitnet_c_output.txt /tmp/bitnet_py_output.txt && echo "PASS: C and Python outputs match!" || echo "FAIL: Outputs differ"

# Create virtual environment and install dependencies (numpy, torch, pytest)
$(VENV): pyproject.toml
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -e ".[dev]"

.PHONY: all clean run apple2 bbc bbc-wav bbc-uef test test-compare harness

# Build a BBC Micro tape WAV from the BBC binary. Plays into the cassette port.
# Uses system python3 — make_bbc_tape.py is stdlib-only, no venv needed.
bbc-wav: build/program.bbc
	python3 tools/make_bbc_tape.py build/program.bbc --load 0x0E00 --exec 0x0E00 --name BITNET --out build/bitnet.wav
	@echo "wrote build/bitnet.wav"

# Build a BBC Micro tape UEF from the BBC binary. Drop into PlayUEF
# (playuef.8bitkick.cc/?LOCAL=true) to convert to audio in-browser, or
# load directly into an emulator (JSBeeb, BeebEm, b-em).
bbc-uef: build/program.bbc
	python3 tools/make_bbc_tape.py build/program.bbc --load 0x0E00 --exec 0x0E00 --name BITNET --out build/bitnet.uef
	@echo "wrote build/bitnet.uef"
