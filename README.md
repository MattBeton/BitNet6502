# Bitnet 6502

<img src="https://mattbeton.com/img/blog/bitnet-6502/header.png" alt="A BBC Micro generating text from the model" width="260" align="right">

A language model that runs on a 6502 — the 8-bit, 2 MHz CPU from the BBC Micro and Apple II. 32 KB of addressable RAM, integer-only, no FPU. The model + inference engine fit in that budget.

The prompt `once upon a time` generates the following output:

> once upon a time tom and lily saw things lily were sad her house he heartd them ilily and tom said yes she saw a little girl smiled tom was so excited her mom said yes

Full write-up: [mattbeton.com/blog/bitnet-6502](https://mattbeton.com/blog/bitnet-6502.html)

## Design decisions

Every modelling choice falls out of a hardware constraint:

- **Ternary weights (BitNet).** The 6502 has no multiply instruction, so weights are constrained to {−1, 0, +1} and matmul reduces to add/subtract. Four weights pack per byte, unpacked with a shift. The head/conv/SSM-C tensors keep int4 resolution where ternary is too coarse.
- **Mamba-style SSM backbone.** A fixed-size recurrent state avoids a KV cache that grows with sequence length — inference memory stays flat, which matters against a 32 KB budget.
- **int8 activations, int16 accumulate.** Dot products accumulate in 16-bit, then re-scale through a learned right-shift so values don't saturate on the way back to int8. No float anywhere on the device.
- **Character tokenization** (27 tokens: a–z + space) keeps the embed/unembed matrices small against the parameter budget.

Trained in PyTorch via fake-quant + straight-through estimator, then exported to packed integer weights for the C engine.

## Repo layout

```
BitNet6502/
├── dataset/       # data loading + tokenizer
├── model/         # architecture + training
├── inference/     # python reference + 6502 C engine + weight export
├── tests/         # A/B parity tests: C (via sim65 harness) vs Python
├── apple2files/   # Apple II disk images
├── tools/         # BBC Micro tape (UEF/WAV) builder
├── build/         # build outputs + checked-in deployable checkpoint
└── makefile
```

# Usage

## Requirements

```bash
brew install cc65                     # 6502 C compiler + sim65 emulator
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # numpy, torch, pytest
```

## Train + export pipeline

A trained model is checked in (`inference/c/weights.c`), so `make run` works out of the box. To train your own, the flow is linear — data → train → export → run:

```bash
python -m dataset.prepare_tinystories                # build corpus
python -m model.train                                # train checkpoint in build/
python -m inference.export_weights [checkpoint.pt]   # export checkpoint to C weights
make run                                             # build + run C engine in sim65
```

Deploy the built binary to hardware:

```bash
make bbc-uef       # BBC tape UEF for emulators / PlayUEF
```

This produces a UEF image of the tape that can be read into a BBC. This can be ingested by [PlayUEF](http://playuef.8bitkick.cc/?LOCAL=true) to play the audio into the BBC Micro's tape drive input.

## Tests

A/B equivalence tests run every C op against its Python reference and assert byte-exact equality:

```bash
make test          # build harness + binaries, run pytest
make test-compare  # run C and Python end-to-end, diff stdout
```
