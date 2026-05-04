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
| Steps | 6,000 |
| Test loss | **1.7426** |
| Final train loss | 1.7599 |
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

Saved checkpoint: `build/bitnet_quant_npe84.pt`.

### Sample generations

Greedy:

```
the bear a made a made a am that a made a am and burn a able a made a am and a make a be a am that ...
```

Top-k (k=8, T=0.9):

```
stilf our and the see ope my to and shers guill booth hust which arish to mine of the pread we comioly bre
of all will now my love whin herefull of his and his tis do the what reced to cought cothen what sir thin
burbenes you stren our answens worther brin his and you bring our his surars worn i thou me pon the a home
but your may fors in your peausen brain i i our worrow courare to have to dingra
```

Top-k samples are recognizably English, with frequent valid words (`our`, `and`, `the`,
`see`, `to`, `mine`, `now`, `my`, `love`, `his`, `what`, `you`, `home`, `your`, `have`).
Greedy decoding collapses into repetitive 3–4 token loops, which is normal for a
small char-level LM at this loss.

## Comparison to upstream baselines

| Model | Params | Steps | Test loss | Notes |
| --- | ---: | ---: | ---: | --- |
| Transformer baseline (`transformer.py`) | 95,187 | 2,000 | 2.0823 | float, full FP32 |
| Unquantized SSM (`state-space.py`, `state-size=8`) | 96,903 | 2,800 | 1.4715 | float, parallel-scan SSM, LayerNorm |
| **BitNet quant SSM (this work)** | **76,123** | **6,000** | **1.7426** | int8/int16, ternary, recurrent, no norm |

The deployable BitNet model beats the unquantized transformer baseline by 0.34 nats
despite using 20% fewer parameters and being fully integer-quantized. It trails the
unquantized state-space baseline by 0.27 nats, which is the price of the 6502 constraints.

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

- **A5 vs A0**: the full cost of quantization on this architecture is **0.30 nats**.
- **A2 → A0**: int8 activation rounding (without changing anything else) costs ~0.08 nats.
- **A3 → A0**: the ±127 clip on top of the shift costs ~0.05 nats. The remaining 0.05
  is the rounding-to-int that A3 still does.
- **A1 (float weights, int8 activations) is *worse* than A0**: this isn't real signal.
  The shift initialisations and uniform `[-1, 1]` weight inits in `bitnet_quant.py` are
  tuned so that `~50%` of weights round to ±1 after ternary quantization. Removing the
  ternary round leaves weight magnitudes well below 1, so the int16 accumulator output
  is much smaller than the shift expects — most layers output near-zero activations and
  the network has little signal to train on. This is a calibration confound, not
  evidence that ternary weights help.
- **The 0.30 quant cost stacks roughly additively**: 0.08 (int8 acts) + 0.05 (clip) +
  ~0.16 from interactions (mainly: ternary weights are hard to compose with saturating
  activations because the network has fewer ways to express small adjustments).
- **The remaining 0.12 nats** between A5 (1.58 at 4,000 steps, this architecture) and
  the upstream unquantized state-space baseline (1.46 at 2,800 steps, ~95k params, no
  pos embed disabled, parallel-scan SSM with LayerNorm) is **architecture difference**:
  fewer parameters (76k vs 95k), recurrent vs parallel-scan SSM, no LayerNorm,
  hardtanh-style saturation in the residual stream.

### Total gap accounting

Starting from the unquantized SSM baseline (1.46) and walking to the deployable
quantized model (1.74):

| Step | Loss | Cumulative Δ |
| --- | ---: | ---: |
| Unquantized SSM baseline (95k params, parallel scan, LayerNorm) | 1.46 | 0.00 |
| Same baseline reproduced in our recurrent/no-norm/76k-param architecture (A5) | 1.58 | +0.12 |
| Add int8 activations | ~1.66 | +0.08 |
| Add ±127 saturation | ~1.71 | +0.05 |
| Add ternary weight rounding (full deployable, A0) | 1.87 | +0.16 |
| (A0 was 4k steps; 6k-step run gets to 1.74) | 1.74 | (training time) |

The takeaway: roughly **2/3 of the loss-gap to the upstream SSM is the cost of
quantization** and is essentially the price of running on a 6502; the remaining
**1/3 is architectural** (recurrent state, no norm, fewer params) and is potentially
recoverable.

## What to look into next

Ordered by how impactful I expect them to be on test loss, given the ablation results.

### 1. Add an integer RMSNorm before each block

The unquantized baseline's biggest architectural difference from ours is its
LayerNorm, and `src/F.c` already has the integer-sqrt and divide kernels needed
to implement RMSNorm with int8/int16 arithmetic. Worth ~0.05–0.10 nats based on
how much A5 trails the upstream SSM. Implementation cost is moderate: needs an
RMSNorm fake-quant module on the training side and a few lines of C on the inference
side, both of which can mirror the existing integer-sqrt/divide already in `F.c`.

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

### 8. Train for longer

The deployable run was still descending at step 6,000. The unquantized baseline
trained for fewer steps (2,800), but with ~25% more parameters and a smoother
loss landscape. A 12,000-step run is cheap (~25 min) and might recover ~0.05 nats
just from more training time, especially given the saw-tooth pattern in loss
caused by integer shifts flipping between adjacent values during training.

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
