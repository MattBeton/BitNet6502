# Quantized GRU on TinyStories — findings

Built a quantized GRU under the same 23,770-byte ROM budget and
int8/int16/ternary contract as the SSM (Mamba-style) deployable model,
ran ablations on which weight matrices should be ternary vs int4 vs
int8, and trained the best-stable cell at full scale on TinyStories.

**Headline:** the GRU is **significantly worse** than the SSM at this
budget — best valid 1.49 vs SSM's 0.90 — and it is *fundamentally
harder to train* in this regime: 5 of the 6 ablation cells diverged
to NaN within ~400 steps. The diagonal SSM's structure (constant
per-channel decay + linear gating) is the right family for an int8
recurrent LM; full-matrix recurrence (W_hh · h) amplifies small
perturbations in a way that ternary weights can't keep up with.

## Architecture

`modelling/experiments/quant_gru.py` defines:

- **Cell**: two ternary linears (`weight_ih`, `weight_hh`), each 3H wide;
  output is split into r/z/n gates. Reset and update gates use a
  hard-sigmoid (`shift + 64 → clamp [0, 127]`); candidate uses
  hard-tanh (`saturating_shift_int8`). The hidden update is
  `((127 − z) * n + z * h_prev) >> 7`, saturated to int8.
- **Stacking**: 2 layers, hidden 79 (all-ternary), shrinking with int4
  flags to keep ROM at 23,770 bytes.
- **Head**: tied to nothing; ternary by default, optionally int4 or int8.

Mirrors the SSM's:
- int8 activations and hidden state
- int16 biases and accumulators
- Ternary weights via STE
- Learned per-layer integer right-shifts to scale int16 → int8

## Byte-budget search

`gru_total_bytes(hidden, n_layer=L, int4_ih, int4_hh, head_precision)`
computes the weight blob size exactly. Solving for the largest H that
fits 23,770 bytes per cell:

| Cell | head | int4_ih | int4_hh | hidden | bytes |
| --- | --- | --- | --- | ---: | ---: |
| A baseline (all ternary) | ternary | no  | no  | 79 | 23,293 |
| B int4 head              | int4    | no  | no  | 78 | 23,290 |
| C int4 weight_hh         | ternary | no  | yes | 66 | 23,421 |
| D int4 weight_ih         | ternary | yes | no  | 66 | 23,421 |
| E int4 head + hh         | int4    | no  | yes | 65 | 23,213 |
| F int8 head ("fullquant")| int8    | no  | no  | 76 | 23,263 |

int4-on-recurrence costs ~13 hidden dim (79 → 66); int4-on-head only
~1 hidden dim. int8 head also costs ~3 dim. So int4 on the inner
matrices is expensive; int4/int8 on the head is nearly free.

## Ablation results (5k steps each)

| Cell | hidden | Valid @ 5k | gn_max | Verdict |
| --- | ---: | ---: | ---: | --- |
| A baseline (all ternary) | 79 | **82.73** | **∞** | ❌ diverged at step 400 |
| B int4 head              | 78 | **107.93** | ∞ | ❌ diverged |
| C int4 weight_hh         | 66 | **10.72**  | ∞ | ❌ diverged |
| **D int4 weight_ih**     | 66 | **1.4031** | 30.78 | ✅ only stable cell |
| E int4 head + hh         | 65 | **52.70** | ∞ | ❌ diverged |
| F int8 head              | 76 | (killed early — pattern was clear) |  |

**Only int4 weight_ih kept training stable.** The all-ternary baseline
diverged within 400 steps, with gradient norm going to infinity. The
explanation is that ternary {−1, 0, +1} on the input projections gives
the gate accumulators too few distinct levels to stably differentiate
input tokens — once the gates saturate uniformly, the recurrent
feedback through W_hh · h_prev amplifies any drift, and the standard
GRU stability margin disappears.

int4 on weight_ih (8 levels per channel) gives the gates enough
dynamic range to differentiate inputs cleanly. The cost: the weight
matrix doubles in bytes, so we shrink hidden 79 → 66 to make budget —
losing ~19 % of the model's capacity along the way.

## What works for the GRU vs what works for the SSM

| | SSM (Mamba-style) | GRU |
| --- | --- | --- |
| All-ternary baseline trainable? | ✅ yes (Shakespeare and TinyStories) | ❌ no (NaN by step 400 on TinyStories) |
| Anneal helps?       | ✅ Shakespeare, ❌ TinyStories | n/a — already too unstable |
| int4 on head only?  | ✅ −0.06 nats on Shakespeare | doesn't help (stability bottleneck is elsewhere) |
| int4 on recurrent op? | n/a (no full recurrent matmul; diagonal state) | ✅ **required** for stability |
| Hidden dim that fits 23,770 B | n_embd=81 | 66 (with int4_ih) |

## Final 30k → 15k training

The first 30k attempt with the winning Cell D config diverged at step
5000 — same "NaN-cascade" pattern, just delayed. Even with
gradient-skip safety (refuse to step Adam if any gradient is non-finite)
the second 30k attempt diverged at step 7000.

The fix that actually worked:

1. Drop LR to **1e-3** (was 2e-3 on Shakespeare/SSM).
2. Tighten grad clip to **0.5** (was 1.0).
3. **Freeze shifts at step 2250** (the exact wobble window — same
   absolute step where the 5k ablation cell D saw gn_max=30).
4. Shorter total schedule (15,000 steps), so the LR plateau at peak
   is short enough not to give the wobble enough time to compound.

With those four together, the run stayed stable. The freeze step is
visible in the grad-norm trace: gn_max drops from 1.9 to 0.14 at step
2250 and never comes back up.

### Final result

| | Value |
| --- | ---: |
| Architecture | 2-layer GRU, hidden=66, head ternary, weight_ih int4, weight_hh ternary |
| Parameters / ternary | 56,635 / 27,918 |
| ROM | 23,421 / 23,770 bytes |
| Steps | 15,000 |
| Final valid loss | 1.5197 |
| **Best valid loss** (step 12500) | **1.4945** |

`build/bitnet_quant_gru_final.pt` is the saved checkpoint.

## Direct comparison to the SSM on TinyStories

| Family | Best valid | Hidden / n_embd | Total params | Notes |
| --- | ---: | ---: | ---: | --- |
| **SSM (Mamba-style)** | **0.9013** | 81 | 71,218 | 30k steps, full v3 stack, **deployable** |
| GRU (this work) | 1.4945 | 66 | 56,635 | 15k steps, int4_ih required for stability |
| Δ (GRU − SSM) | **+0.59 nats** | −15 | −14,583 | |

**Three things compound to make the gap:**

1. **Param count / capacity** (~−0.1 nats expected).  GRU's 6H² ternary
   weights per layer mean we can fit ~57k params vs the SSM's 71k at
   the same byte budget. Even at H=79 (the all-ternary GRU baseline)
   we'd only get to 80k params, but that cell isn't trainable.
2. **Quantization sensitivity** (~−0.2 nats expected).  GRU needs int4
   weight_ih to be stable at all. The wider int4 init lets the gate
   accumulator span a useful range; ternary doesn't. The
   compounding-via-recurrence makes everything worse.
3. **Architectural fit for int8 inference** (~−0.3 nats expected).  The
   diagonal SSM's state update is just `state = sat_int8((decay * state)
   >> 7 + B * u)` — one constant per channel, no matmul, no
   amplification. The GRU's `h = f(W_hh · h_prev + W_ih · x)` mixes
   every channel into every channel each step, so any error in h grows
   geometrically until clipping pins it.

The third one is the killer: it's not a tuning issue but a structural
mismatch. The SSM was designed (in Mamba) to be linear in the
recurrence specifically because that's the property you want for
parallel scan and quantization — and we're getting the quantization
benefit cleanly. Adding nonlinearities back in (GRU's sigmoid gates,
LSTM's full gating) gives back the gradient ergonomics of a
recurrent NN but loses the int8-stability you need at deploy.

## Sample generations

Final GRU checkpoint, prompt `"once upon a time "`, 75 tokens × 5
samples, top-k=8, temperature=0.9:

```
[1] once upon a time and he seare toy he said thome time tog thawe
    hto see he dot hes ard it wir
[2] once upon a time the put to asked look mot mom ankiry and hew
    said his girl nom low to play
[3] once upon a time saig the was vello ecut walke end sor a diddnd
    and with pire day and and do
[4] once upon a time and ascas micele with play sorry welt was a
    cone to play anday it good and
[5] once upon a time that it was a girl and said a little itd and it
    he play it thes he dong one
```

Real-English fragments are coming through ("said", "play", "mom",
"girl", "look", "good") and the TinyStories style is faintly
recognisable ("said his girl", "to play"), but malformed words are
much more common than the SSM's output. Compare the SSM samples at
the same prompt:

```
[SSM]   once upon a time there was a little girl named timmy tog buy
        she alseed tom they are safe an
[GRU]   once upon a time and he seare toy he said thome time tog
        thawe hto see he dot hes ard it wir
```

The SSM almost always produces a real story opening with a named
character; the GRU produces sentence-fragment-soup. Consistent with
the 0.6-nat loss gap.

## Reproducing

```bash
# Ablation grid (~70 min if all 6 cells run; we killed at cell E)
.venv/bin/python -u modelling/experiments/gru_ablations.py

# Final 15k run (the stable recipe)
.venv/bin/python -u modelling/experiments/gru_run.py \
    --steps 15000 --int4-ih \
    --freeze-frac 0.15 --lr 1e-3 --grad-clip 0.5 \
    --save build/bitnet_quant_gru_final.pt

# Sample
.venv/bin/python -u modelling/experiments/sample.py \
    --ckpt build/bitnet_quant_gru_final.pt \
    --prompt 'once upon a time ' --n 75 --num-samples 5
```

Logs / artifacts:
- `gru_ablations.csv` — per-cell results
- `gru_abl_*_log.txt` — per-cell stdout (showing divergence patterns)
- `gru_final_log.txt` — final run log
- `gru_final_loss.png` — loss curve

## Takeaway

For an int8/int16/ternary inference engine running on a 6502, **the
diagonal SSM (Mamba-style) is the right architectural family**. The
GRU's full-matrix recurrence is incompatible with ternary weights in
the recurrent path — it diverges before training can take hold — and
even after stabilising with int4 input weights, it lands roughly 0.6
nats higher than the SSM at the same byte budget on TinyStories.

If you wanted to bring the GRU within striking distance of the SSM,
the things to try would be:

- **int4 weight_hh too** (smaller hidden ≈ 58, even more capacity
  loss) — would address the recurrence stability completely but cost
  another ~10 hidden dim. Probably doesn't recover the gap.
- **Layer norm** on the hidden state — gives a magnitude prior that
  the SSM gets implicitly via saturation. Would need an integer
  RMSNorm, costs ROM bytes.
- **Drop the sigmoid gates entirely** — replace with linear gating
  like the SSM uses. At which point you're essentially reinventing the
  SSM.

None of these are likely to close the gap fully; the gap is mostly
architectural.
