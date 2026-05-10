# BitNet quantized SSM — experiment design

Concrete next-step experiments to push the deployable model below the current
1.6853 test loss. The constraint is **bytes of weight storage on the 6502**,
not parameter count. Every experiment must hold the total weight blob fixed
at 23,770 bytes by trading bytes from one place to another. Non-clean
dimensions (e.g. `n_embd = 81`) are allowed; the C-side bit-packer handles
the remainder bytes with row padding.

## Current byte budget

Computed exactly from the deployable model (n_embd = 84, no pos embed,
state_size = 8, n_layer = 3, vocab = 27).

| Type | Bytes | Share |
| --- | ---: | ---: |
| Ternary weights (in_proj, out_proj, conv, B, C, head) | 17,703 | 74.5% |
| int8 (token embedding, decay, D) | 4,536 | 19.1% |
| int16 (biases) | 1,512 | 6.4% |
| Shifts (uint8 each) | 19 | 0.1% |
| **Total** | **23,770** | **100%** |

The argument for holding bytes fixed: improvements found in this regime
should scale up when we move to a target machine with a slightly larger
ROM. The ratios are what matter — "swap bytes from in_proj to head, does
loss go down?" — not the absolute model size.

## Methodology

Each experiment specifies:

- **Bytes added** by the upgrade.
- **Compensation knob**: which dimension is shrunk to absorb the cost
  (almost always `n_embd`, since it's the strongest single byte lever).
- **Net Δ bytes**: should be 0 ± a few bytes (rounding / padding).

Each experiment is run as a 4,000-step training (matches the existing
ablation protocol) and reports test loss against the byte-matched
deployable baseline (also 4k steps, currently A0 = 1.8713). Promoting to
12k is a separate decision per experiment. **Loss is the metric**;
qualitative sample inspection is unreliable at this scale and not used.

---

## E1 — int4 on the highest-leverage small tensors

**Hypothesis.** Ternary {−1, 0, +1} discards too much precision in the
places where the model has to make sharp distinctions on small fan-in:
the head (27 classes from an 84-vector), the SSM `C` matrix (reads back
state through a ternary 8-vector), and the depthwise conv (4-tap local
pattern). int4 ({−7, …, +7}, 4 bits/weight) doubles their byte footprint
but quadruples their representational range. Of the 0.21 nat ternary
cost, some is plausibly concentrated in these "decision-point" tensors.

### Sub-experiments and byte arithmetic

The current ternary→int4 sizes:

| Tensor (deployable) | Ternary bytes | int4 bytes | Δ |
| --- | ---: | ---: | ---: |
| head (27×84) | 567 | 1,134 | +567 |
| SSM C (84×8) × 3 layers | 504 | 1,008 | +504 |
| Conv (84×4) × 3 layers | 252 | 504 | +252 |
| Sum (E1c, all three) | 1,323 | 2,646 | +1,323 |

| Cell | Bytes added | Compensation | Net |
| --- | ---: | --- | ---: |
| E1a: head only | +567 | shrink n_embd from 84 to (~82) until ternary blob shrinks by 567 | ≈ 0 |
| E1b: head + SSM C | +1,071 | n_embd to (~80) | ≈ 0 |
| E1c: head + SSM C + conv | +1,323 | n_embd to (~79) | ≈ 0 |

(Exact compensation `n_embd` solved at run time per cell — the relationship
between `n_embd` and total bytes is roughly quadratic via in_proj/out_proj,
linear via embedding/B/C/decay/D/biases.)

**Method.**

- Add `int4_quantize` STE alongside `ternary_quantize` in `bitnet_quant.py`:
  `q = clamp(round(w), -7, 7)`.
- Add `int4_head`, `int4_ssm_C`, `int4_conv` flags to `Config`.
- Add a helper that, given a target byte budget and the int4 flags,
  solves for the largest `n_embd` whose total weight bytes ≤ budget.
- 4-cell screen at 4,000 steps each. Promote winner to 12k.

**Cost (6502 inference).**

- Ternary mac: ~3 cycles (test, conditional add/sub).
- int4 mac: ~10 cycles (multiply via shift-and-add, no hardware mul).
- E1a: head matmul once per token = 27 × 84 = 2,268 macs ⇒ ~16k extra
  cycles per token.
- E1b: + SSM C readout per timestep × 3 layers × 64 timesteps ⇒
  ~650k extra cycles per full block.
- E1c: + conv ⇒ ~325k extra cycles per full block.
- Engineering: int4 × int8 → int16 kernel in `src/matrix.c`, parity test,
  packer in `export_weights.py`.

**Expected Δ.** −0.03 to −0.10 nats (best guess −0.04 to −0.07). Cells
with smaller upgrades may give smaller improvements, but the head-only
upgrade is the cheapest implementation and bounds the leverage from below.

**Priority.** **H.** Highest leverage near the dominant 0.21 ternary cost.

---

## E2 — int16 activation residual stream

**Hypothesis.** Per the corrected ablation attribution, 8-bit activation
precision costs **0.08 nats**. With int16 acts and int32 accumulators,
both rounding-grain noise and saturation collapse to near zero, recovering
essentially the entire activation-side gap.

**This is a RAM trade, not a ROM trade.** Weights are unchanged, so the
23,770-byte budget is irrelevant. The constraint is inference RAM:

| Buffer (n_embd = 84) | int8 RAM | int16 RAM | Δ |
| --- | ---: | ---: | ---: |
| Residual stream | 84 | 168 | +84 |
| SSM state (84 × 8 × 3 layers) | 2,016 | 4,032 | +2,016 |
| Conv window (84 × 4 × 3 layers) | 1,008 | 2,016 | +1,008 |
| **Total** | **~3,108** | **~6,216** | **+3,108** |

Plus int32 scratch for matmul accumulators (small, transient).

**Method.**

- Replace `fake_quant_int8` with `fake_quant_int16` for residual stream
  and SSM state.
- Replace `saturating_shift_int8` with a 16-bit version (clip ±32,767).
- C side: parallel int16 inference path (substantial new code: `int16_t`
  buffers, ternary × int16 → int32 matmul kernel, etc.).

**Cost.**

- ROM: unchanged.
- RAM: +3,108 bytes inference state.
- Cycles: 1.5–2× slowdown on matmul-heavy paths.
- Engineering: significant — separate int16 kernels in C, separate parity
  tests.

**Expected Δ.** −0.06 to −0.09 nats.

**Priority.** **M.** Bigger absolute gain than E1, but engineering-heavy
and trades RAM (which is also limited on a real 6502 target). Defer
until E1 is in the bag.

---

## E3 — Recalibrated A1 (float weights, int8 acts)

**Hypothesis.** A1 in the ablation hit 2.81 (worse than A0 at 1.87) — a
calibration artefact, not signal. The shift inits and weight init U[−1, 1]
are tuned so ternary rounding gives ~50% nonzero ±1 weights. With float
weights and the same init, magnitudes stay well below 1 and the network
stalls. Re-running A1 with init recalibrated to give the same pre-shift
accumulator variance as A0 should land near A2 (1.79), confirming the
0.21 nat ternary cost in `findings.md` is real.

**Method.**

- Run with `ablate_float_weights=True` plus a wider weight init: roughly
  U[−1.225, 1.225] (matches the float-W variance of 0.5 to ternary's
  density-0.5 variance), or equivalently keep U[−1, 1] and reduce the
  in_proj shift by 1 to compensate for smaller magnitudes.
- Verify with a smoke-test forward pass that pre-shift accumulator
  ranges match A0's.
- 4,000-step training, single cell.

**Cost.** Trivial — config change, one run.

**Expected Δ.** A1 lands at 1.78–1.85.

**Priority.** **L.** Methodology cleanup; doesn't change the deployable
model.

---

## E4 — Freeze integer shifts in late training

**Hypothesis.** The 12k-step run shows ±0.07 nat oscillation in late
training (e.g. step 8k → 9k went 1.69 → 1.79 → 1.71). This matches the
failure mode where `round(shift_continuous)` flips between adjacent
integers when the continuous proxy is near a half-integer boundary,
causing a discrete 2× change in output magnitude. Freezing the shifts
after some warmup percentage and only training weights / biases / decay /
gain afterwards should remove the oscillation source.

**Method.**

- Add `freeze_shift_after_frac: float = 0.6` to `Config`.
- After `step >= freeze_shift_after_frac * num_steps`, set
  `requires_grad=False` on every parameter named `*shift*`. The rounded
  values are kept as constants for the rest of training.
- 12k-step run, compare to the unfrozen 1.6853.

**Cost.** Trivial — training-loop change. No inference-side change
(shifts are already integer constants at deploy).

**Expected Δ.** −0.02 to −0.04 nats.

**Priority.** **H.** Cheapest possible change. Run alongside E1.

---

## E5 — Tied input/output embeddings

**Hypothesis.** With tying, the logit for token `t` is forced to be
`embedding(t) · h_final` — i.e. the head row for `t` *is* the input
embedding row for `t`. There are three things going on, which we want
to disentangle:

- **(a) Inductive bias from shared I/O token representation.** In
  transformer LMs on word-level data, tying typically reduces perplexity
  by 1–3% (Press & Wolf 2017). The reason is that input embeddings tend
  to encode meaningful token structure (semantic neighbours land near
  each other), and reusing that structure for the head means the head
  doesn't have to re-discover it. **For 27 unstructured character
  tokens, there is essentially no semantic structure to share**, so we
  expect this effect to be small to zero, possibly even slightly
  negative if the constraint hurts.
- **(b) Head precision changes.** Tied at int8, the head matmul becomes
  int8 × int8 → int16 instead of ternary × int8 → int16. This is a
  separate effect (similar to E1a) and can show up as a positive Δ.
- **(c) Re-spent bytes.** Tying saves 567 bytes (the head's ternary
  table). At fixed total bytes, those bytes get re-invested in widening
  `n_embd`, which is its own source of improvement.

The naive comparison (tied vs untied) confounds all three. The byte-
matched experiment we actually want runs two cells:

- **E5a**: tied at int8, with the freed 567 bytes spent on widening
  `n_embd` until the total byte budget matches the deployable model.
- **E5b (control)**: untied, same widened `n_embd` as E5a. This is
  *over budget* by 567 bytes, so it isn't a deployable config — it's
  there to isolate (a) + (b) from (c).

If E5a − baseline ≈ E5b − baseline, the win is entirely from (c) (extra
width), and tying is neutral. If E5a < E5b, tying is genuinely helping
via (a) + (b). If E5a > E5b, the inductive bias is hurting (which would
be a surprise for our scale).

**The other precision option.** Tying at ternary would force the
embedding to be ternary too — only 3 distinct values per channel, used
for the input rep of every char. With 27 chars and 84 channels, the
rows would be near-random ternary patterns at init and recovery is
unlikely. We don't run this.

**Byte arithmetic.**

| | Bytes |
| --- | ---: |
| Untied embedding (int8) | 2,268 |
| Untied head (ternary) | 567 |
| Untied total | **2,835** |
| Tied at int8 (one matrix shared) | 2,268 |
| **Saving from tying** | **567** |

| Cell | Spec | Net bytes |
| --- | --- | ---: |
| E5a | tied int8, `n_embd` widened to absorb the freed 567 bytes | ≈ 0 |
| E5b (control) | untied, same widened `n_embd` as E5a | ≈ +567 over budget |

**Method.**

- `tie_embeddings: bool` flag in `Config`.
- When True, `head.weight` becomes a view of `token_embedding`. The
  head's separate parameter is removed.
- Init range: stay with U[−16, 16] (the embedding's range). The head's
  shift will adapt.
- Calibration risk: same trap as A1 — current inits were tuned for
  separate distributions. Watch initial loss.

**Cost.**

- ROM: −567 bytes from the tie, then re-spent on the chosen knob.
- RAM: unchanged.
- Cycles: head matmul becomes int8 × int8 instead of ternary × int8.
  Ternary mac ~3 cycles → int8 mac ~12 cycles. For a 27 × 84 head,
  ~2,268 macs × 9 extra cycles ≈ 20k extra cycles/token. Bounded.
- Engineering: trivial training side. C side: swap head from
  `ternary_linear` to `int8_linear`.

**Expected Δ.** −0.01 to −0.03 nats. Most of any positive Δ is likely
from re-spent bytes (effect (c)), with the inductive-bias win from (a)
expected to be small at char-level vocab. The control cell E5b is what
makes the result interpretable.

**Priority.** **M.** Cheap to test, modest expected size, and
informative on whether the I/O-tying inductive bias is worth the head
matmul slowdown.

---

## E6 — Quantization annealing on the ternary weights

**Hypothesis.** From step 0, the optimizer is steering through a loss
surface that is heavily distorted by the ternary STE: every weight is
discretised to {−1, 0, +1}, gradients pass through a mismatched
forward/backward pair, and the network has to find ternary minima
*before* it has even formed useful representations. A standard QAT
remedy is to **anneal the quantisation in**: train at full float
precision early, then gradually push toward ternary. The expectation
is that the optimizer first finds a good *float* minimum, and the
ternary rounding then only has to displace the solution by a small
amount, rather than the network having to discover ternary structure
from scratch.

The dominant cost in our deployable model is ternary weight rounding
(0.21 of the 0.30 nat quant gap, per the corrected ablation), so this
is the right place to attack: if the float minimum has structure that
survives the ternary projection, annealing can recover some of it.
Like E4, this is a training-schedule lever — **no architecture or
byte-budget change**, no inference-side change.

**Method.**

Add an `anneal_alpha(step)` schedule to `bitnet_quant.py` returning
`α ∈ [0, 1]`. Replace `ternary_quantize` with an annealed version:

```python
def ternary_quantize_annealed(w, alpha):
    q   = torch.clamp(torch.round(w), -1.0, 1.0)
    fwd = alpha * q + (1.0 - alpha) * w   # interpolate float ↔ ternary
    return w + (fwd - w).detach()         # STE through w
```

At α=0 the forward returns `w` unchanged (float). At α=1 the forward
returns `q` (ternary STE — identical to current behaviour). Intermediate
values produce a partial-ternary forward. Backward is identity-on-w
throughout, so AdamW's running statistics stay coherent across the
ramp.

Schedule (initial guess, tune in screen):

| Phase | Step range | α |
| --- | --- | ---: |
| float warmup | 0 — 0.15 × N | 0 |
| linear anneal | 0.15 × N — 0.50 × N | 0 → 1 |
| pure ternary | 0.50 × N — N | 1 |

That gives the optimizer ~1,800 steps of full-float training at the
start of a 12k run before any ternary pressure, ~4,200 steps of
gradually-increasing ternary, and ~6,000 steps of pure-ternary
fine-tuning at the end. The pure-ternary tail is necessary because
the deployable model needs to converge to actual {−1, 0, +1} weights,
not to a floaty pre-image.

**Activation quantisation** (int8 acts) stays on throughout — only
weights are annealed. Activation precision (0.08 nats) is a smaller
cost, and the activation forward/backward is less of a discontinuity
than ternary rounding.

**Apply to which weights?** All ternary tensors: in_proj, out_proj,
conv, B, C, head. Same α schedule for all of them.

**Risks.**

- **Magnitude blow-up during float phase.** With ternary off, weights
  are no longer pinned to ±1. Existing weight decay (0.04) and
  AdamW will partially control this, but the shifts (init: in_proj=3,
  out_proj=5, etc.) were tuned for ~50%-nonzero ±1 weights. During
  the float phase, weight magnitudes will likely be smaller (they're
  not pulled to ±1), so accumulator outputs will be smaller and
  shifts will adapt downward. When α reaches 1 the weights snap to
  ±1, accumulators jump up, and shifts have to adapt upward. This
  whiplash is the main thing to watch.
- **Gradient distribution shift mid-training.** AdamW's per-parameter
  variance estimates are tuned to the early gradient magnitudes; if
  those change substantially during anneal, adaptation may lag.
  Reset Adam's stats at α=1, or use a coarser schedule, if this
  hurts.
- **It might just not help.** Some QAT papers report negative results
  for very low-bit quantization (ternary is at the extreme end), where
  the float minimum is so far from any ternary one that the
  pre-training is wasted. Char-LMs at 76k params are also at a scale
  where the float overparameterisation might dominate the
  inductive-bias benefit.

**Cost.** Trivial — schedule + one branch in the quantizer. No
inference-side change.

**Expected Δ.** −0.02 to −0.05 nats, with low confidence. Could be
zero or slightly negative if the schedule whiplash dominates.

**Priority.** **M.** Same kind of lever as E4 but a more invasive
training-time change and lower confidence. Run after E4 lands.

---

## Suggested ordering

1. **E4 (freeze shifts).** Cheapest possible change — a few lines in
   the training loop, no architecture or inference-side change. The
   12k-step trajectory shows ±0.07 nat oscillation in late training
   that this directly attacks. Run first.
2. **E1a (head-only int4)** as the smallest int4 upgrade test, possibly
   combined with E4. Bounds the int4-on-decision-points leverage.
3. **E5a + E5b (tied embeddings, both cells).** E5b is over-budget by
   design and isolates the tying win from the re-spent-bytes win.
4. **E6 (quantization annealing).** Same family as E4, but more
   invasive and with lower confidence. Worth running once E4's
   trajectory shape is known.
5. **E1b–c (extend int4 to SSM C and conv)** if E1a delivers ≥ −0.03.
6. **E2 (int16 acts)** only if E1 lands and we still want to push.
7. **E3 (recalibrated A1)** as methodology cleanup, attached to
   findings doc.
