# BitNet quant LM on TinyStories — findings

Companion to the Shakespeare findings (`findings.md`). Same model
architecture, same 27-char vocab, same 23,770-byte ROM budget. The
data changed: from ~30k chars of Shakespeare to **336.6M tokens** of
TinyStories filtered to a 500-word vocabulary.

The headline number: **valid loss 0.9013** at step 28000 of a 30k run
on TinyStories — vs the Shakespeare best of 1.5425. Most of the gap
is the dataset (much simpler distribution, much larger), not the
architecture.

`build/bitnet_quant_tinystories_final.pt` is the deployable
checkpoint (final-eval valid 0.9190; min-eval valid 0.9013).

## Why we did ablations: the v1 divergence

The first attempt copied the Shakespeare v3 stack verbatim (E1c +
E4 + E6 = int4 head/SSM-C/conv + freeze shifts at 50% + quantization
annealing) but with `weight_decay` relaxed from 0.04 → 0.01 and
`dropout` dropped to 0 (both motivated by the larger dataset
removing overfitting risk). Result:

| step | train | valid |
| ---: | ---: | ---: |
| 0    | 4.15 | 4.16 |
| 1000 | 1.38 | **1.32** |
| 2000 | **12.50** | **12.62** (collapsed) |

A V-shaped loss curve with a catastrophic explosion at step 2000.
Plot in `tinystories_v1_loss.png`.

## Ablation grid

5 cells × 5,000 steps each (~22 min on MPS), single-axis flips from
the v1 config. Gradient norm logged at every eval to look for
explosions even when clipping hides them:

| Cell | Anneal | WD | LR | Final valid | gn_avg | gn_max | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| **A** | off | **0.04** | 2e-3 | **1.0209** | 0.06 | 0.08 | ✅ stable, best |
| B | on  | 0.04 | 2e-3 | **74.40** | **34M** | **35M** | ❌ exploded |
| C | off | 0.01 | 2e-3 | 1.0334 | 0.06 | 0.08 | ✅ stable |
| D | on  | 0.01 | 2e-3 | **20.17** | **9.7M** | **9.9M** | ❌ exploded |
| E | on  | 0.04 | 1e-3 | 1.2404 | 0.32 | 0.42 | ✅ stable but worse |

(`tinystories_ablations_loss.png`, `tinystories_ablations_gn.png`)

### Reading the table

The single dimension that matters is **anneal on/off**. With anneal
on at 2e-3 the model **always** explodes (gradient norm in the
millions). Lowering the LR to 1e-3 saves it but loses 0.22 nats vs
the same config without anneal. Weight decay barely matters when
anneal is off — A and C are within run-to-run noise.

### Why anneal works on Shakespeare but not TinyStories

The annealing forward path during the float phase is just
`forward = w` — i.e. the weights are unconstrained floats. The
shifts in the model were tuned for ternary `{-1, 0, +1}` weights,
which have small std (~0.71). In the float phase, weights have no
±1 cap. Gradient updates push them around, accumulator outputs
saturate the int8 activations everywhere, and once activations are
all stuck at ±127 the loss landscape becomes a cliff.

On Shakespeare's 30k-character corpus this rarely bit because
gradients were noisy enough to be self-regularising — successive
batches pulled in conflicting directions and weight magnitudes
hovered around their target. On TinyStories' 336.6M tokens (and the
filtered-to-500-words simplicity), gradients across batches are
much more coherent: each step pushes weights in a consistent
direction, magnitudes grow monotonically, and the model crosses
into the saturating regime within ~1500 steps. The grad-norm
trajectory shows it cleanly: cell B's `gn_max` is 35 million.

**Conclusion:** anneal is a Shakespeare-scale trick, not a
universal one. It only helped because the training was tiny enough
for the float-phase magnitude blow-up not to dominate; on a real
dataset, ternary STE from step 0 (no anneal) is both stable and
better.

## Final stack (TinyStories)

| | |
| --- | --- |
| Architecture | n_embd=81, 3 layers, gated, no pos embed |
| Quantization | int4 head + int4 SSM C + int4 conv kernel; ternary in_proj/out_proj/B; int8 acts |
| Steps | 30,000 |
| Optimizer | AdamW, LR 2e-3 (cosine to 1e-4), warmup 1000, weight_decay 0.04, grad_clip 1.0 |
| Dropout | 0.0 |
| Anneal (E6) | **off** |
| Freeze shifts (E4) | **at 50%** (step 15,000) |
| Batch size × block size | 128 × 64 |

Total parameters 71,218; 66,096 ternary; 23,652 / 23,770 byte ROM.

## Training trajectory

Plot in `tinystories_final_loss.png`. Key milestones:

| Phase | Step range | Behaviour |
| --- | --- | --- |
| Warmup | 0 — 1,000 | LR ramps 0 → 2e-3; loss collapses 4.15 → 1.40 |
| Free training | 1,000 — 15,000 | Loss dives to ~0.93, then **wobbles** (gn_max spikes to 13.3 around step 13k); shifts oscillate across integer boundaries |
| Frozen-shift training | 15,000 — 30,000 | gn_max collapses 13 → **0.06** the moment shifts freeze; loss settles cleanly to 0.90 |

Freezing the shifts at 50% (E4) does the same thing here as on
Shakespeare — and the effect is even more visible because the
gradient norm logging makes it obvious. The freeze step is at the
exact moment the gradient noise drops by ~100×, and the post-freeze
trajectory is monotonic.

Best valid loss: **0.9013** at step 28000.
Final-eval valid loss: 0.9190 (slightly higher — eval batches are
randomly sampled from 3.3M valid windows, so ±0.02 noise is normal).

## Sample generations

Final checkpoint, prompt `"once upon a time "`, 75 tokens × 5 samples,
top-k=8, temperature=0.9:

```
[1] once upon a time there was a little girl named timmy tog buy
    she alseed tom they are safe an

[2] once upon a time there and alsmall asked timmy and tom says
    they played timmy had a louse be

[3] once upon a time there was a little be safe she said i love
    they hear mom it was nice and th

[4] once upon a time there was a little dog went out his mom said
    help mom smiled and dad and no

[5] once upon a time there was a little bear fast ben he says he
    went to the red his friends one
```

Greedy: `once upon a time there was a little girl named lily and
ben are friends and said i am a litt`

Almost every word is real English. TinyStories character names
appear (lily, ben, tom, timmy). Story openings, basic dialogue, and
sentiment words (loved, sad, brave, nice) come through. There's
clear residual ungrammaticality ("there and and said") and some
malformed words ("alseed", "louse"), but the gap from Shakespeare's
v3 (mostly malformed words, no character names, no dialogue
structure) is large.

## Comparison: Shakespeare v3 vs TinyStories final

| | Shakespeare v3 | TinyStories final |
| --- | ---: | ---: |
| Train chars / tokens | ~30k chars | 336.6M tokens |
| Best test/valid loss | 1.5425 | **0.9013** |
| Architecture / ROM | identical (n_embd=81, 23,652 B) | identical |
| Anneal | yes (won −0.09 on this corpus) | **no** (loses on this corpus) |
| Freeze shifts | yes, at 50% | yes, at 50% |
| Steps | 20,000 | 30,000 |
| Sample quality | mostly malformed words, hints of "romeo", "peace", "farewell" | real English words, character names, story scaffolds |

The 0.64-nat gap is mostly the dataset (entropy of TinyStories'
500-word vocab is far below open Shakespeare). The architecture
ports cleanly; the only training-side knob that needed flipping was
**turning anneal off**.

## Reproducing

```bash
# Build the filtered token cache (~3 minutes for the 1.9 GB train file):
.venv/bin/python -c "
from modelling.tinystories import filtered_tokens_with_cache, DEFAULT_TRAIN_PATH, DEFAULT_VALID_PATH
from pathlib import Path
v = Path('modelling/data/tinystories_vocab_top500.txt')
filtered_tokens_with_cache(DEFAULT_VALID_PATH, v)
filtered_tokens_with_cache(DEFAULT_TRAIN_PATH, v)
"

# Final 30k run (~63 min on MPS):
.venv/bin/python -u modelling/experiments/tinystories_run.py \
    --steps 30000 --wd 0.04 --no-anneal --freeze-frac 0.5 \
    --save build/bitnet_quant_tinystories_final.pt

# Sample (5 short generations):
.venv/bin/python -u modelling/experiments/sample.py \
    --ckpt build/bitnet_quant_tinystories_final.pt \
    --prompt 'once upon a time ' --n 75 --num-samples 5
```

## Things that should be tried next

- **Longer training.** Loss was still trending down in the last
  ~5,000 steps; a 50k or 80k run is cheap and probably worth
  another 0.05–0.10 nats.
- **Larger batch size.** The dataset is essentially infinite for our
  purposes, and per-step gradient quality is already good — bigger
  batches would amortise the per-step overhead and probably let us
  go past LR 2e-3 without re-introducing the float-phase
  instability anneal had.
- **int16 SSM state only** (the partial-int16 idea from the
  Shakespeare followup) — the same logic applies here, and the
  bigger eval set will give a much cleaner read on its impact.
- **Drop the freeze-shifts at 50% for an even cleaner schedule.**
  With anneal off, the early-training oscillation is small enough
  that freezing earlier (say 30%) might pay off.
