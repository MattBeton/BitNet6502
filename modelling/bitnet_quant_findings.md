# BitNet quantized state-space LM — findings

Companion to `modelling/bitnet_quant.py`. This document records the deployable
configuration, the ablation experiments that attribute the loss gap, and a
list of next things to try if we want to close more of the gap later.

## Deployable target

The deployable model is the one whose forward pass is bit-exact to what an
int-only inference engine on the 6502 will produce.

| | |
| --- | ---: |
| Parameters | 76,123 |
| Ternary parameters | 70,812 |
| Steps | 12,000 |
| Test loss | **1.6853** |
| Final train loss | 1.7093 |
| Architecture | recurrent diagonal SSM, 3 layers, n_embd=84, state_size=8, gated, no position embedding |
| Activations | int8 throughout the residual stream |
| Accumulator | int16 |
| Weights | ternary {−1, 0, +1} |
| Biases | int16 |
| Per-layer activation scale | learned non-negative integer right-shift, applied as `(acc >> shift)` then saturate to int8 |
| SSM decay `a` | int8 in [0, 127], directly learned (no `exp` / `softplus`) |
| SSM `B`, `C` | ternary |
| SSM `D` | int8 |
| Head | ternary weights, int16 logits, argmax for inference |
| Generation on 6502 | greedy argmax (no softmax / `exp` / division anywhere) |

Saved checkpoints:

- `build/bitnet_quant_npe84.pt` — 6,000-step run, test loss 1.7426 (earlier deployable target).
- `build/bitnet_quant_npe84_12k.pt` — **12,000-step run, test loss 1.6853 (current deployable target).**

### Sample generations

12k-step model, top-k (k=8, T=0.9):

```
arther of wome byat my wife man let for men for worthy this bring to surp and all will his art in with
mads back thee rated is the so guar a for she romeo in sees a will be honour oner worshor hart brother
hear bornd i blows as the an set now so all langed in is the and sh the mather for on the gentlows for
the shoply bereas that as with a cont cannotined chear in being alful areset whim fest nomi
```

Sample is recorded for completeness; loss is the metric we trade off against. At
2.43 bits / character (1.6853 nats / ln 2) the model is well above the entropy of
English text but still produces mostly malformed words, which is what would be
expected of a 76k-parameter, fully-quantized character-level LM.

## Comparison to upstream baselines

| Model | Params | Steps | Test loss | Notes |
| --- | ---: | ---: | ---: | --- |
| Unquantized SSM (`state-space.py`, `state-size=8`) | 96,903 | 2,800 | 1.4715 | float, parallel-scan SSM, LayerNorm |
| BitNet quant SSM (6k steps) | 76,123 | 6,000 | 1.7426 | int8/int16, ternary, recurrent, no norm |
| **BitNet quant SSM (12k steps, current target)** | **76,123** | **12,000** | **1.6853** | same architecture, longer training |

## Iteration history

### Baseline run

Initial config: `n_embd=80, n_layer=3, state_size=8, use_pos_embed=True, use_gate=True`,
74,739 params. Final test loss **1.7997** at step 5,999.

### Screen at 2,500 steps

Tested three variants. Lower loss is better.

| Variant | Params | Test loss @ 2,500 | Verdict |
| --- | ---: | ---: | --- |
| `wider-84x3-s8` (with pos embed) | 81,499 | — | over 80k budget, skipped |
| `lower-lr` (1e-3 instead of 2e-3) | 74,739 | 2.1054 | clearly worse |
| `no-pos-embed-84` | 76,123 | 1.9085 | best — promoted |

Removing the position embedding freed enough parameter budget to widen `n_embd`
from 80 to 84, more than compensating for the lost positional signal. The recurrent
SSM already encodes order through the state update, so explicit position information
is partially redundant.

### Promotion run (deployable)

`no-pos-embed-84` at 6,000 steps reaches **1.7426** test loss. Loss kept dropping
throughout training, with most of the win arriving in the second half (1.90 at step
1,800 → 1.74 at step 6,000) as the learned shifts and ternary weights settled.

## Ablation experiments

To attribute the gap between the deployable model and an unquantized version of the
same architecture, every quantization constraint was switched off in turn. All runs
used the deployable architecture (n_embd=84, no pos embed) trained for 4,000 steps.
The toggles are exposed via `Config.ablate_*` flags in `bitnet_quant.py` and consumed
by module-level state, so the same model code is used in every run.

### Ablation table

| Run | Weights | Activations | Saturation | Test loss | Δ vs full quant |
| --- | --- | --- | --- | ---: | ---: |
| **A0** full quant (deployable) | ternary | int8 + clip ±127 | yes | **1.8713** | — |
| A1 float weights | float | int8 + clip ±127 | yes | 2.8108 | +0.94 (confounded) |
| A2 float activations | ternary | float | n/a | 1.7941 | −0.08 |
| A3 no saturation | ternary | int8 + shift, no clip | no | 1.8184 | −0.05 |
| **A5 all float** (architecture-only baseline) | float | float | no | **1.5757** | −0.30 |

### Reading the table

The total cost of quantization (A5 → A0) is **0.30 nats**. Decomposing it into
"add ternary weights first, then add int8 activations":

| transition | constraint added | Δ |
| --- | --- | ---: |
| A5 → A2 | float weights → ternary weights (activations still float) | **+0.21 nats** |
| A2 → A0 | float activations → 8-bit activation precision | **+0.08 nats** |
| A5 → A0 (sum) | full quantization | **+0.30 nats** ✓ |

So most of the gap — about **2/3 — is the ternary weight rounding**, and the
8-bit activation precision accounts for only ~1/3 of it.

**On the activation cost: the split between "clip" (0.05) and "rounding" (0.03)
isn't a split between two different constraints.** The shift+clip is a
hard-tanh-shaped 8-bit quantizer with two failure modes — values inside ±127
get snapped to the nearest int (rounding loss), values that would have exceeded
±127 get capped (clipping loss). Both are consequences of *the same* 8-bit
budget; with 16-bit acts, both losses shrink in lockstep. The 0.05/0.03 split
just reflects how the trained model spends its 8 bits — its learned shifts
push activations close to the saturating edge of the int8 range, where each
unit covers more relative range. That's an optimal use of the bits, not
evidence of two separable cost sources.
matt's note: doesn't this mean that this ablation test was just wrongly performed? The test should have been performed using initialization

A1 (float weights, int8 activations) sits at 2.81, *worse* than A0. This is a
calibration artefact, not evidence that ternary weights help: the shift
initialisations and the uniform `[-1, 1]` weight inits are tuned so that ~50%
of weights round to ±1 in the ternary path. Removing the ternary round leaves
float weights with magnitudes well below 1, the int16 accumulator output is
much smaller than the shift expects, layers emit near-zero activations, and
training stalls. The clean reading of "what does ternary cost?" is A5 → A2
(both regimes share the same calibration), giving the 0.21 above.

The remaining **0.12 nats between A5 (1.58) and the upstream unquantized
state-space baseline (1.46)** is **architectural**: ~20% fewer parameters,
recurrent vs parallel-scan SSM, no LayerNorm, hardtanh-style saturation in
the residual stream.
- **The remaining 0.12 nats** between A5 (1.58 at 4,000 steps, this architecture) and
  the upstream unquantized state-space baseline (1.46 at 2,800 steps, ~95k params, no
  pos embed disabled, parallel-scan SSM with LayerNorm) is **architecture difference**:
  fewer parameters (76k vs 95k), recurrent vs parallel-scan SSM, no LayerNorm,
  hardtanh-style saturation in the residual stream.

### RMSNorm ablation (added later)

The unquantized SSM baseline uses LayerNorm before each block. We don't, because
LayerNorm requires float division. To test whether the missing normalisation is a
significant part of our gap, we added an integer-faithful pre-block RMSNorm
(`QuantRMSNorm` in `bitnet_quant.py`) with a per-channel learned int8 gain. The
forward path mirrors `rms_norm` in `src/F.c` (sum of squares → integer sqrt →
signed/unsigned divide) plus the gain, so it is deployable on the 6502. Same
4,000-step training schedule as the rest of the ablation.

| Run | Test loss | Δ vs no-norm sibling |
| --- | ---: | ---: |
| A0 full quant + RMSNorm | 1.8622 | −0.009 |
| A0 full quant (no norm) | 1.8713 | — |
| A5 all float + RMSNorm | 1.5537 | −0.022 |
| A5 all float (no norm) | 1.5757 | — |

**RMSNorm gives ~0.01 nats in the quantized regime and ~0.02 nats in the float
regime — both within run-to-run noise.** The gap to the upstream SSM baseline is
*not* explained by the missing LayerNorm.

Why? A few candidate reasons:

- The integer gain receives a very small gradient signal (~1e-5 at init, vs
  ~1e-3 for the matmul weights). AdamW normalises this, but the effective step
  size is still small relative to the int8 quantisation grain — the gain barely
  moves off its init value of 32.
- Our architecture only has 3 layers. RMSNorm matters more for deeper stacks
  where activation magnitudes drift across many residual additions.
- The saturating int8 add in the residual stream (`fake_quant_int8(residual + y)`)
  already keeps activation magnitudes bounded, performing some of the same
  job as RMSNorm at this depth.

**Implication for inference engine work:** integer RMSNorm is *not* worth
implementing on the C side for this architecture. The existing `rms_norm` in
`F.c` can stay unused.

### Longer training

The 6,000-step run was visibly still descending at the final eval. Doubling to
12,000 steps (with a proportionally longer cosine schedule, 600-step warmup)
recovered another **0.057 nats**:

| Run | Steps | Test loss |
| --- | ---: | ---: |
| no-pos-embed-84 (6k) | 6,000 | 1.7426 |
| **no-pos-embed-84 (12k)** | **12,000** | **1.6853** |

The trajectory is noisy in late training (e.g. step 8000 → step 9000 went 1.69 →
1.79 → back to 1.71), reflecting the integer shift parameters flipping between
adjacent values and causing discrete output magnitude jumps. A future
intervention worth trying is freezing the shifts after, say, 60% of training
to remove this oscillation source.

### Total gap accounting

Starting from the unquantized SSM baseline (1.46) and walking to the deployable
quantized model (1.69 at 12k steps), with each constraint measured *in the
regime where everything else above it is also off*:

| Step | Loss (4k steps unless noted) | Cumulative Δ |
| --- | ---: | ---: |
| Unquantized SSM baseline (95k params, parallel scan, LayerNorm, 2,800 steps) | 1.46 | 0.00 |
| Reproduce in our recurrent / no-norm / 76k-param architecture (A5) | 1.58 | +0.12 |
| Add ternary weight rounding (A2: ternary W, float acts) | 1.79 | +0.21 |
| Add int8 activations + shift + ±127 clip (A0, full quant) | 1.87 | +0.08 |
| Same A0 architecture trained 12k steps instead of 4k (current target) | 1.69 | (training time recovers ~0.18) |

So at 12k steps the deployable model is **0.23 nats away from the upstream
baseline**, decomposing roughly as:

- **~0.21 nats — ternary weight rounding** (the dominant single cost).
- **~0.08 nats — 8-bit activation precision**, expressed as a hard-tanh-shaped
  shift+clip. Within this, the trained model leans on the saturating edge of
  the int8 range, so more of the 0.08 manifests as clipping than as
  rounding-grain noise — but that's a property of how the bits are being used,
  not two separate constraints.
- **~0.12 nats — architecture / param-count** (recurrent SSM, no norm, ~20%
  fewer params). Integer normalisation does *not* close this — see the
  RMSNorm ablation above.
- **−0.18 nats — recovered by longer training** (4k → 12k steps).

## What to look into next

Ordered by how impactful I expect them to be on test loss, given the ablation results.

### 1. Add an integer RMSNorm before each block ❌ tried, no help

Hypothesis was wrong. Implemented as `QuantRMSNorm` and benchmarked at 4k steps
in both the quantized and float regimes (see RMSNorm ablation above). It buys
~0.01 nats quantized, ~0.02 nats float — both within run-to-run noise. Don't
implement on the C side.

### 2. Widen the SSM state internally

Right now the SSM state is int8 and the decay update saturates after every step:

```
state[s] ← saturate_int8((a * state[s]) >> 7  +  B * u)
```

If the residual contributions are small relative to int8 range, the state mostly
stays in the linear region; but when input bursts saturate the state, information
is lost. **Cost: a few hundred bytes of RAM** (n_embd × state_size × 2 = ~1.3KB).
**Benefit: probably 0.03–0.05 nats** — the saturation cost in A3 was ~0.05 and
the SSM state update is one of the biggest places where it bites.

Easiest version: keep the state in int16 internally, and only saturate to int8
when computing the output. The decay multiply is then int8 × int16 = int24, which
fits in int32 with a single shift back to int16. The C code change is small:
state buffer goes from `int8_t[n_embd][state_size]` to `int16_t[n_embd][state_size]`.

### 3. Per-channel activation shifts

Currently every layer has a single learned shift applied to all output channels.
For SSM-output channels especially, the per-channel dynamic range varies a lot
(because the decay parameter varies and so do the long-range integration scales).
A per-channel shift would give each channel its own dynamic range. **Cost: n_embd
extra bytes per shift point (~250 bytes total for the deployable model).**
**Benefit: maybe 0.02–0.05 nats**, low confidence — unclear how much per-channel
freedom matters versus just having more parameters.

### 4. Reintroduce position information via a different mechanism

Removing the position embedding gave us +4 channels of width and improved loss.
But the SSM only has 64 timesteps of effective context, and longer-range positional
information is lost entirely. Two cheap re-introductions worth screening:

- **Token-conditional bias on the first conv layer** — the conv kernel is depthwise
  ternary (4×n_embd values, ~330 ternary positions), and adding a trained int8 bias
  per channel would give the conv a natural "positional offset" without tying it to
  absolute index.
- **Sinusoidal-like fixed pattern**: a small (block_size, n_embd) lookup that's not a
  learned parameter but a precomputed int8 table. Costs no parameter budget (it's
  data, not weight), and could be packed into the inference binary.

### 5. Investigate the calibration confound (A1 result)

If we want a clean answer to "do ternary weights actually hurt vs. float weights of
similar magnitude," we need to repeat A1 with shift/init recalibration so that
float-weight A1 produces similar-magnitude pre-shift outputs to A0. As-is, A1's
2.81 number is misleading. This isn't architecture-relevant — it's just a sanity
check for the ablation methodology. Worth fixing only if we care about the per-
constraint attribution being clean.

### 6. Different ternary distribution targets

The current uniform `[−1, 1]` init produces ~50% zeros after rounding. Some BitNet
variants use a different init (or a sparsity regulariser) to push the zero rate
up to 60–70%, which makes inference faster (more zeros = more skipped adds on the
6502). Doesn't affect loss directly but matters for cycles/token. Worth measuring
once the C inference engine is real and we have a concrete clock-cycle target.

### 7. Schedule the quantisation in (not out)

A common BitNet recipe is to **ramp up** the quantisation strength during training,
e.g. start with float weights and gradually anneal toward {−1, 0, +1}. The intuition
is that the optimiser finds a better float minimum first, and the ternary round only
has to slightly displace it. Could be 0.02–0.05 nats. Has the disadvantage of being
slower to iterate on (no instant signal at step 0 about whether the architecture is
quantizable).

### 8. Train for longer ✅ tried, +0.057 nats

A 12,000-step run on the deployable config recovered 0.057 nats (1.74 → 1.69)
relative to the 6,000-step result. The current saved checkpoint
(`build/bitnet_quant_npe84_12k.pt`) reflects this. Could potentially go further
(20k steps) but with diminishing returns; the LR schedule and the discrete shift
oscillations in late training cap how much extra training time helps.

### 9. Freeze integer shifts in late training (untried)

Late-training loss oscillates by ~0.07 nats step-to-step (e.g. 8k → 9k went
1.69 → 1.79 → back to 1.71). This is consistent with the per-layer integer
shifts flipping between adjacent values when `round(shift_continuous)` lies near
a half-integer boundary. Freezing the shifts after some warmup percentage
(say 60%) and only updating weights / biases / decay / gain afterwards should
remove the oscillation source and could buy another 0.02–0.04 nats. Cheap to
try — one config flag and a check in the optimizer step.

## How to reproduce

```bash
# Deployable run (used for build/bitnet_quant_npe84.pt):
.venv/bin/python -u modelling/bitnet_quant.py
# (Then change Config defaults: use_pos_embed=False, n_embd=84.
#  Or run via the screen harness:)
.venv/bin/python -u modelling/bitnet_quant_screen.py

# Ablation suite (5 runs at 4,000 steps each, ~40 min total):
.venv/bin/python -u modelling/bitnet_quant_ablate.py
```
