# Curriculum experiment — short→long context expansion

Tests whether starting training on short single-sentence windows
(block_size=32) and then extending to ~3-sentence windows
(block_size=128) helps vs training on long windows from step 0.

Both cells use the same TinyStories config that the previous run
established (no anneal, wd=0.04, freeze shifts at 50%, int4 head +
SSM-C + conv). Eval is fixed at block_size=128 in both, so the
valid-loss curves are directly comparable.

## Setup

| | C1 control | C2 curriculum |
| --- | --- | --- |
| Phase 0 | bs=128 × 8000 steps | **bs=32 × 4000 steps** |
| Phase 1 | — | **bs=128 × 4000 steps** |
| Eval block_size | 128 | 128 |
| Freeze shifts at step | 4000 | 4000 |
| Eval interval | 200 (high-fidelity) | 200 |
| Per-step train logged | EMA α=0.05 | EMA α=0.05 |

## Headline results

| Cell | Final valid | Best valid | Wall time | Time per step |
| --- | ---: | ---: | ---: | ---: |
| C1 control     | 0.9229 | 0.9203 | 2987 s (50 min) | 0.37 s/step |
| **C2 curriculum** | **0.9116** | **0.9116** | **1795 s (30 min)** | 0.06 s phase 0 / 0.32 s phase 1 |
| Δ (C2 − C1) | **−0.0113** | −0.0087 | **−1192 s (−40 %)** | |

So curriculum is **fractionally better on loss AND ~40 % faster in
wall time** at matched step count. The loss difference (0.01 nats)
is within run-to-run eval noise — but the wall-clock saving is
*not* — it falls out cleanly from phase 0's 4× shorter sequences
costing ~6× less per gradient step than phase 1.

## Loss + grad-norm trajectories

`curriculum_curves.png` — top: per-step EMA train (light) + per-eval
valid (dashed dots) for both cells; bottom: per-step grad norm.

What the plot shows:

- **The two trajectories are essentially superimposed.** Curriculum's
  bs=32 phase reaches the same loss-vs-step curve as control's
  bs=128 phase, despite training on 4× shorter sequences. The
  recurrent SSM is genuinely indifferent to the training-time
  context length within this range — the gradient signal per
  position is enough.
- **Phase boundary at step 4000 is visible but small.** When the
  curriculum cell switches bs=32 → 128 (vertical orange line), the
  loss has a brief 1000-step "settling" wobble around 1.00 and
  then converges to the same minimum the control reaches. No
  dramatic divergence or recovery — the model state generalizes
  cleanly from one window length to another.
- **Grad norm tells the same story.** Both cells' gn trace is
  noisy-and-bounded for the first 4000 steps (peak ~10), then
  collapses to ~0.05 the moment shifts freeze at step 4000.
  Curriculum doesn't change the noise profile — only the per-step
  cost.

## What this means for the hypothesis

The hypothesis was that a short-then-long curriculum could help the
model learn local patterns first and then long-range structure. On
this architecture and dataset:

- The valid-loss-per-step curve is **basically the same** between
  cells. So curriculum doesn't *learn faster* in gradient-step terms.
- It does train faster in **wall-clock terms** — phase 0 costs
  ~6× less per step. So the same model quality is reached in 60% of
  the wall time.
- Implication for "context expansion": the SSM's recurrent state
  carries information forward indefinitely already, so training on
  longer windows isn't doing additional work the model can't get
  from short windows. The phase 1 (bs=128) extension is mostly
  about giving the optimizer *more positions per backward pass*, not
  about teaching long-range structure.

So the right framing isn't "curriculum helps you learn long context"
— at this scale, all of context length is already captured by the
recurrent state. The right framing is "**curriculum is a
compute-efficiency knob**: you can trade off bs-per-step for steps
without losing loss." That's a useful result.

## Sample generations

Both checkpoints, prompt `"once upon a time "`, 75 tokens × 5
samples, top-k=8, temperature=0.9.

**C1 control** (final valid 0.9229):

```
[1] once upon a time there was a little girl named timmy tom tought
    to see read tom said you are
[2] once upon a time there and a smalud once there was a smally the
    end ben the end ben said the
[3] once upon a time there was a little ben were scared that day
    she he was time they came him d
[4] once upon a time there was a long who all not have to do she
    could happy back home to play w
[5] once upon a time there was a little bear feel she hugged her
    felt sad looked his friends one
```

**C2 curriculum** (final valid 0.9116):

```
[1] once upon a time there was a little girl name and lily a back
    better to play with the box ca
[2] once upon a time there and alsy were happened and they see ith
    say a little boy named lily i
[3] once upon a time there was a little big asked her home there
    was a boiny its they played tog
[4] once upon a time there curious not his mom they came on a
    bravend bepon it was so muve it is
[5] once upon a time there was a little bird flowers they led away
    tom and her mommy he says of
```

Sample quality is broadly similar. Both produce real-English-mostly
output with TinyStories character names (tom, ben, lily, timmy),
basic plot fragments ("hugged her felt sad", "led away tom and her
mommy"), and TinyStories openers. Curriculum's samples don't look
qualitatively better — consistent with the loss being within noise.

## Reproducing

```bash
.venv/bin/python -u modelling/experiments/curriculum_run.py \
    --steps 8000 --eval-interval 200 --eval-block 128

.venv/bin/python -u modelling/experiments/plot_curriculum.py
```

CSVs:
- `curriculum_C1_control_steps.csv` / `_eval.csv`
- `curriculum_C2_curriculum_steps.csv` / `_eval.csv`

Checkpoints:
- `build/bitnet_quant_C1_control.pt`
- `build/bitnet_quant_C2_curriculum.pt`

## Followups worth trying

- **Run curriculum at iso-wall-time.** Keep C1 at 8k steps and let
  C2 spend its saved 20 min in extra phase-1 training. That gives
  curriculum a real opportunity to beat control rather than just
  match it.
- **Try a 3-phase curriculum** (e.g. 16 → 64 → 256). If the SSM is
  truly indifferent within 32–128, maybe 256 unlocks something. The
  block_size=128 ceiling here was arbitrary.
- **Independent runs at different seeds.** The 0.01 nat gap is
  unstable; a 3-seed average would tell us whether C2's edge is
  signal or noise.
