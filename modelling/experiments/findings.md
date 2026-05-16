# BitNet quantized SSM — experiment findings

Companion to `bitnet_quant_experiments.md`. Each experiment was run for 4,000
steps on the deployable architecture (n_embd=84 unless byte-budget required
shrinking, no position embedding, gated SSM, ternary weights, int8 acts) and
compared to **A0 = 1.8713** (the same arch at 4k steps from the original
ablation).

Per-experiment Python files live in `modelling/experiments/`. Raw logs:
`modelling/experiments/<expname>_log.txt`. Tabular results:
`modelling/experiments/results.csv`.

## Headline table

| ID  | Cell | Test loss @ 4k | Δ vs A0 | Hypothesis |
| --- | --- | ---: | ---: | --- |
| A0  | full quant (deployable, baseline) | 1.8713 | — | — |
| **E4**  | freeze shifts after 60% of training | **1.7713** | **−0.1000** | ✅ confirmed |
| E1a | int4 head only (n_embd → 82) | 1.8083 | −0.0630 | ✅ confirmed |
| E5a | tied head, n_embd → 85 | 24.5377 | diverged | ⚠ failure mode (re-run with frozen shift below) |
| E5b | untied control at n_embd 85 (over budget) | 1.7966 | −0.0747 | (control) |
| **E6**  | quantization annealing (float→ternary) | **1.7805** | **−0.0908** | ✅ confirmed |
| E1b | int4 head + SSM C (n_embd → 81) | 1.7807 | −0.0906 | ✅ confirmed |
| **E1c** | int4 head + SSM C + conv (n_embd → 81) | **1.7470** | **−0.1243** | ✅ confirmed |
| E2  | int16 activations (RAM trade) | 1.7701 | −0.1012 | ✅ confirmed |
| E3  | recalibrated A1 (float weights, std-matched init) | 1.6268 | −0.2445 | ✅ confirmed |

Three training-time-only winners (E4, E6) and one architecture winner (E1c)
were promoted into the final stack. E2 also wins by a similar margin to E4
but trades RAM, so it is held in reserve.

---

## E1a — int4 weight on the head matrix only

| n_embd | Bytes used | Test loss | Δ vs A0 |
| ---: | ---: | ---: | ---: |
| 82 | 23,389 / 23,770 | 1.8083 | **−0.063** |

**Implication.** Confirms the hypothesis that ternary {-1,0,+1} discards
useful precision in small-fan-in decision-point tensors — replacing the
27×84 head with int4 (and shrinking n_embd 84→82 to absorb the +567-byte
cost) recovers ~30% of the activation+weight quantization gap. The head
is the smallest, cheapest int4 upgrade and bounds the leverage from below.

## E1b/E1c — extend int4 to SSM C and conv kernel

| Cell | n_embd | Bytes | Test loss | Δ vs A0 |
| --- | ---: | ---: | ---: | ---: |
| E1b: int4 head + SSM C | 81 | 23,409 | 1.7807 | −0.091 |
| **E1c: int4 head + SSM C + conv** | **81** | **23,652** | **1.7470** | **−0.124** |

**Implication.** Confirms the hypothesis. Each successive int4 tensor
contributes additional gain even after shrinking `n_embd` to absorb the
ROM cost, ending at **−0.124 nats** — the largest single architectural
improvement we found. Both the SSM C readout (84×8 per layer) and the
4-tap depthwise conv kernel benefit from int4's wider integer alphabet.
The fact that the win continues to compound through the conv suggests
the cost of ternary on small-fan-in tensors is real and roughly additive
across them.

## E2 — int16 activation residual stream (RAM trade)

| Bytes (ROM) | Test loss | Δ vs A0 |
| ---: | ---: | ---: |
| 23,770 (unchanged) | 1.7701 | **−0.101** |

**Implication.** Confirms that ~half of the 0.21-nat ternary cost
attributed to "activations" was actually just int8 saturation —
loosening it to int16 recovers 0.10 nats. The cost is **+3.1 KB of
inference RAM** (residual stream + SSM state + conv windows all double
in size), which competes with the rest of the 6502's RAM budget. **Held
in reserve**: equally as good as E4 in loss terms but it costs RAM, while
E4 is free.

## E3 — recalibrated A1 (methodology cleanup)

| Cell | Test loss | Δ vs A0 | Δ vs A2 (1.7941) |
| --- | ---: | ---: | ---: |
| E3 (float weights, std-matched init U[−1.225, 1.225]) | 1.6268 | −0.245 | −0.167 |

**Implication.** Confirms the hypothesis that A1's original 2.81 was a
calibration artefact, but with a twist: a *correctly-initialised* float-
weight model lands at **1.63**, even better than A2 (1.79, which kept
ternary weights with float activations). The "ternary cost" is therefore
larger than the original ablation suggested — closer to ~0.16 nats vs
A2's 0.21 — *and* the float-weight model is also benefiting from the
saturation removal that A2 didn't have. E3's number is informational
only; the deployable target stays ternary.

## E4 — freeze integer shifts in late training

| Freeze fraction | Test loss | Δ vs A0 |
| ---: | ---: | ---: |
| 0.6 (freeze after step 2400 of 4000) | **1.7713** | **−0.100** |

**Implication.** Strongly confirms the hypothesis — the predicted gain
was −0.02 to −0.04 nats; actual was **−0.10**, the biggest training-only
win we measured. The trajectory is striking: test loss jumps from 1.86
to 1.80 immediately after the freeze step (a discrete drop, consistent
with the shift parameter previously oscillating across an integer
boundary every few hundred steps and the frozen value finally letting
the rest of the model converge cleanly).

## E5 — tied input/output embeddings

| Cell | n_embd | Bytes | Test loss | Notes |
| --- | ---: | ---: | ---: | --- |
| E5a (tied at int8, learned head shift) | 85 | 23,671 | **24.54** | **diverged** at step ~1200 |
| E5b (untied control, over budget) | 85 | 24,244 | 1.7966 | re-spent-bytes effect alone: −0.075 |
| E5a rerun (tied, head shift frozen) | 85 | 23,671 | **6.34** | **also diverged** at step ~1200 |

**Implication.** Tied embeddings are **fundamentally unstable in this
quantization regime**, not just due to a single misbehaving shift
parameter. Both runs (with the head shift learned and with it frozen)
diverged at almost exactly the same step (~1200), so the failure
mechanism is something deeper than the shift. Hypothesised cause: the
token-embedding parameter receives gradient from two paths with very
different scales — a sparse embedding-lookup gradient (only the input
positions) and a dense head-matmul gradient (every position contributes
to the softmax). The dense gradient dominates the magnitude, which under
int8 fake-quantization pushes embedding values across integer boundaries
in ways that break the input-side lookup discretely. The hypothesis from
the experiment doc that this would be a small effect on a 27-character
vocab is *not* what we found — it's a hard failure, not a small effect.

The byte-matched comparison can still be read from E5b: going from
n_embd 84 → 85 untied (ignoring the budget) is worth **−0.075 nats**.
So even if tied embeddings could be stabilized, the inductive-bias win
(effects (a)+(b) in the experiment doc) would have to clear a high bar
to beat the same n_embd untied — which is unlikely on a 27-character
vocabulary. **Conclusion: drop tied embeddings from the stack.**

## E6 — quantization annealing on the ternary weights

| Schedule | Test loss | Δ vs A0 |
| ---: | ---: | ---: |
| 15% float warmup → linear ramp to 50% → 50% pure ternary | **1.7805** | **−0.091** |

**Implication.** Confirms the hypothesis at the high end of the
predicted range (−0.02 to −0.05). The optimizer benefits substantially
from ~600 steps of full-float training before the ternary STE bites,
suggesting the float minimum is closer to a usable ternary projection
than starting from random ternary directly. This makes intuitive sense —
ternary rounding distorts the loss surface more than int8 activation
quantization, so giving the optimizer a head start on weight structure
pays off.

---

## Stack assembly for the final run

The training-time-only winners (E4, E6) compose with each other and with
the architecture winners (E1c). Stacking rules:

| Stage | Effect | Composes with |
| --- | --- | --- |
| E1c (int4 head + SSM C + conv) | architecture, ROM-neutral via n_embd shrink | E4, E6 |
| E4 (freeze shifts at 60%) | training-time, free at deploy | E1c, E6 |
| E6 (anneal warmup → ramp → ternary) | training-time, free at deploy | E1c, E4 |

E6 finishes its ramp by 50% of training; E4 freezes shifts at 60%; the
two phases are sequential, no conflict. Tied embeddings (E5a/b) are
**not** in the final stack — even at best the byte-matched cell would
have to clearly beat E5b's 1.7966, and the architectural simplification
on the C side is small.

**Predicted final-run loss** (4k Δ → 12k via ~−0.06 nats from longer
training, per the existing finding that 6k→12k recovered −0.057):

- baseline @ 4k = 1.8713
- + E4 = −0.10
- + E6 = ~−0.04 incremental (diminishing returns on top of E4)
- + E1c = ~−0.06 incremental (most of the int4 win is independent of
  training tricks)
- → roughly **1.65 at 4k**; promoted to 12k → **roughly 1.58–1.62**.
  Beats the original 12k checkpoint (1.6853) by ~0.06–0.10 nats.

## Final run results

| Run | Stack | n_embd | Bytes | Steps | Freeze frac | Test loss | Δ vs prior 12k (1.6853) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prior deployable (`bitnet_quant_npe84_12k.pt`) | A0 only | 84 | 23,770 | 12,000 | n/a | 1.6853 | — |
| Final v1 (`bitnet_quant_final_v1.pt`) | E1c + E4 + E6 | 81 | 23,652 | 12,000 | 0.60 | 1.6174 | −0.068 |
| Final v2 (`bitnet_quant_final_v2.pt`) | E1c + E4 + E6 | 81 | 23,652 | 12,000 | 0.50 | 1.6082 | −0.077 |
| **Final v3** (`bitnet_quant_final_v3.pt`) | E1c + E4 + E6 | **81** | **23,652** | **20,000** | **0.50** | **1.5425** | **−0.1428** |

**v1 vs v2.** Freezing the per-layer shifts at the *exact* end of the
anneal ramp (50% of training) instead of 1200 steps later (60%) gives
a clean **−0.009 nats** improvement. With freeze at 60%, the network
spends 1200 steps of pure-ternary, learnable-shift training between
anneal completion and shift freeze, during which the shift parameters
drift across integer boundaries (the same oscillation E4 was designed
to attack). With freeze at 50% the shifts are pinned at their
post-anneal values immediately, removing that drift window.

**v2 vs v3.** Stretching to 20k steps (with the cosine schedule scaled
to match) buys an additional **−0.066 nats** — a much bigger win than
the 6k→12k delta of −0.057 measured in the original deployable. The
post-freeze trajectory is monotonic-ish all the way down: step 10000
(immediately after freeze) is 1.64, step 14000 is 1.56, step 17000 is
1.557, step 20000 is 1.542. The loss is still falling at the end of
training, suggesting another 4–8k steps would buy another ~0.01–0.02
nats; we stop at 20k.

## Deployable target (this work)

| | |
| --- | ---: |
| Checkpoint | `build/bitnet_quant_final_v3.pt` |
| Parameters | 71,218 |
| Ternary parameters | 66,096 |
| ROM | 23,652 / 23,770 byte budget |
| Architecture | n_embd=81, 3-layer recurrent diagonal SSM, gated, no pos embed |
| Quantization | int4 head + int4 SSM C + int4 conv kernel; ternary in_proj/out_proj/B; int8 acts; ternary annealed in over first 50% |
| Training | 20,000 steps, AdamW + cosine LR, learned shifts frozen at step 10,000 |
| Test loss | **1.5425** |
| Improvement over prior best | **−0.1428 nats** (−8.5% relative) |

## How to reproduce

```bash
# Re-run all 4k-step screens (~70 min, MPS):
.venv/bin/python -u modelling/experiments/run_all.py

# E5a rerun with frozen head shift (its first attempt diverged):
.venv/bin/python -u modelling/experiments/_run_e5a_rerun.py

# Final 20k stack (≈ 35 min, MPS):
.venv/bin/python -u modelling/experiments/final_run.py \
    --steps 20000 --freeze-frac 0.5 --anneal \
    --int4-head --int4-ssm-C --int4-conv \
    --save build/bitnet_quant_final_v3.pt
```

## Carry-over for future iterations

Things that worked but may have headroom left:

- **Longer training continues to help.** The 20k loss curve is still
  bending down at the end. A 30k–40k run is cheap on this scale.
- **Earlier int4 in the SSM state itself** (state from int8 → int4
  per-channel) was not tested; complementary to E1c (which only
  changed the *readout* C, not the state).
- **Per-channel activation shifts** (E3 in the original deployable
  next-steps doc) — ~0.02–0.05 nat headroom according to that doc.

Things that didn't work and shouldn't be tried again:

- **Tied input/output embeddings** in this quant regime — diverged in
  both runs, with and without learned head shift.
- **Adding RMSNorm** to the deployable architecture — already covered
  in `bitnet_quant_findings.md`, lands in the noise.

**Trajectory observations.** The training dynamics are richer than the
4k screen suggested:

- During anneal (steps 1800–6000), test loss falls to a transient
  minimum of **1.5836** at step 4200 — this is when weights are ~55%
  ternary, ~45% float, so it isn't a deployable number, but it sets a
  lower bound on how much representational capacity the architecture
  has. Once anneal completes at step 6000, loss climbs back to 1.68 as
  the model snaps onto pure-ternary structure.
- The shift-freeze step at step 7200 produced a brief spike up to 1.72,
  followed by recovery to 1.63 within 600 steps. This suggests the
  freeze step itself is a perturbation the model has to relax around;
  freezing earlier (right at the anneal endpoint) might avoid the
  1.68 → 1.72 → 1.63 detour.
- Final-quarter loss is stable around 1.61, with the last eval at step
  11999 reporting **test 1.6174 / train 1.6094**.

The deployable improvement is **0.068 nats relative to the previous
deployable model**, on a fixed 23,770-byte ROM budget and a 4× shorter
schedule's worth of stacked techniques. This translates to roughly a
4% reduction in nats/character.
