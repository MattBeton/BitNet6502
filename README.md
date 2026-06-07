# Bitnet 6502

A language model that runs on a 6502 — the 8-bit, 2 MHz CPU from the BBC Micro and Apple II. 32 KB of addressable RAM, integer-only, no FPU. The model + inference engine fit in that budget.

Architecture: a 3-block diagonal SSM (Mamba-style) with ternary {-1, 0, +1} weights for most projections and int4 for the head/conv/SSM-C tensors. All activations are int8, accumulators int16, no float anywhere on the device. Trained via fake-quant + straight-through estimator.

## Repo layout

```
BitNet6502/
├── dataset/          # data loading + tokenizer
│   ├── data.py             # TinyStories streaming loader, 27-char vocab
│   ├── prepare_tinystories.py
│   └── data/               # raw TinyStories txt + cache (gitignored)
├── model/            # architecture + training
│   ├── model.py            # BitNetLM (the SSM)
│   ├── quant.py            # STE helpers
│   ├── budget.py           # 6502 byte-budget solver
│   └── train.py            # training entrypoint
├── inference/        # python reference + 6502 C engine + weight export
│   ├── reference.py        # pure-integer Python reference (spec for C)
│   ├── export_weights.py   # checkpoint -> inference/c/weights.{c,h}
│   └── c/                  # 6502 C source — compiled by makefile at repo root
├── tests/            # A/B parity tests: C (via sim65 harness) vs Python
├── apple2files/      # Apple II disk images
├── tools/            # BBC Micro tape (UEF/WAV) builder
├── build/            # build outputs + checked-in deployable checkpoint
└── makefile
```

## Setup

Three install paths depending on what you want to do:

```bash
# A. Run the BBC binary in sim65 — no Python at all.
brew install cc65        # macOS; or your distro's cc65 package
make run                 # builds + runs build/program.sim6502

# B. Run the Python reference / parity tests.
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"        # numpy, torch, pytest, plus jupyter/datasets/matplotlib for poking around
make test                                # builds harness + runs pytest

# C. Train your own model.
# Path B, then download TinyStories (one-time) — see the docstring in
# dataset/prepare_tinystories.py for the curl commands.
python -m dataset.prepare_tinystories
python -m model.train
```

The default makefile bootstrap (`make test`) creates `.venv/` from `pyproject.toml` on first run.

## Train

Download TinyStories first (one-time setup — see the docstring in `dataset/prepare_tinystories.py`), then:

```bash
python -m dataset.prepare_tinystories     # collapse raw -> one-story-per-line
python -m model.train                      # 30k steps, valid loss ~1.0
```

Checkpoints write to `build/`. Default config produces an n_embd=56 model matching the deployed `bitnet_quant_n56_v200_dedup_stripped_v2_finetune.pt` (~162 KB of fake-quant weights; ~24 KB after the export-to-int packing).

## Inference

Two implementations: a Python reference (integer-only, no autograd, no exp/softmax-via-float) and a C engine that compiles to 6502 bytecode via cc65.

```bash
# Python reference — same integer ops the 6502 runs
python -m inference.reference

# Build + run the C engine in sim65 (the cc65-bundled 6502 simulator)
make run

# Export a checkpoint to C source for the engine
python -m inference.export_weights [checkpoint.pt]   # writes inference/c/weights.{c,h}
```

Other build targets:

```bash
make apple2        # Apple II binary -> build/program.apple2
make bbc           # BBC Micro flat binary -> build/program.bbc
make bbc-uef       # BBC tape UEF for emulators / PlayUEF
make bbc-wav       # BBC tape WAV (plays into a real BBC's cassette port)
```

## Tests

A/B equivalence tests run every C op against its Python reference and assert byte-exact equality. Run via:

```bash
make test          # builds harness + program binaries, runs pytest
make test-compare  # runs C and Python end-to-end and diffs stdout
```

The Python venv is created automatically on first run from `tests/requirements.txt`.
