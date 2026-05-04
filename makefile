# Compiler and assembler settings
CC65 = cc65
CA65 = ca65
LD65 = ld65

# Target platform (sim6502 for fast iteration, apple2 for Apple II)
TARGET = sim6502
LIB = $(TARGET).lib

# Source files
C_SRC = src/matrix.c src/weights.c src/F.c src/model.c src/program.c
ASM_SRC = src/text.s

# Intermediate assembly files
C_ASM = $(C_SRC:src/%.c=build/%.s)

# Object files
C_OBJ = $(C_SRC:src/%.c=build/%.o)
ASM_OBJ = $(ASM_SRC:src/%.s=build/%.o)
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
build/%.s: src/%.c
	$(CC65) $(CFLAGS) -o $@ $<

# Rule to assemble generated assembly files to object files
build/%.o: build/%.s
	$(CA65) $(AFLAGS) -o $@ $<

# Rule to assemble ASM files to object files
build/%.o: src/%.s
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

HARNESS_LIB_C = src/matrix.c src/F.c
HARNESS_LIB_OBJ = $(HARNESS_LIB_C:src/%.c=build/%.o)
HARNESS_OUTPUT = build/test_harness.$(TARGET)

# Compile harness.c with -I../../src so it can find matrix.h
build/harness.s: tests/c_harness/harness.c
	$(CC65) $(CFLAGS) -I src -o $@ $<

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

# Create virtual environment and install dependencies
$(VENV): tests/requirements.txt
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -r tests/requirements.txt

.PHONY: all clean run apple2 test test-compare harness
