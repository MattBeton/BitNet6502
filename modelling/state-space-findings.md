# Shakespeare State-Space LM Findings

## Fixed constraints

- Sequence length: `64`
- Dataset split: unchanged `train_fraction=0.9`
- Tokenization: unchanged character vocabulary from `shakespeare.py`
- Parameter budget: strictly under `100,000`
- Device used for measurements: Apple MPS
- Main metric: loss on the fixed eval set
- Secondary metric: sampled character-level text coherence

## Baseline

The starting point was `modelling/transformer.py`, run unchanged.

| Model | Params | Final eval loss | Notes |
| --- | ---: | ---: | --- |
| 2-layer causal transformer | 95,187 | 2.0823 | Fast, but generations were mostly short fragments and malformed words. |

The transformer baseline established that a sub-100k model can train quickly, but it leaves a lot of eval loss on the table.

Baseline checkpoint curve:

| Step | Train loss | Eval loss |
| ---: | ---: | ---: |
| 0 | 3.5314 | 3.5366 |
| 200 | 2.4221 | 2.4713 |
| 400 | 2.3486 | 2.3938 |
| 600 | 2.3080 | 2.3592 |
| 800 | 2.2644 | 2.3218 |
| 1000 | 2.2265 | 2.2789 |
| 1200 | 2.1859 | 2.2388 |
| 1400 | 2.1270 | 2.1889 |
| 1600 | 2.0880 | 2.1416 |
| 1800 | 2.0596 | 2.1126 |
| 1999 | 2.0295 | 2.0823 |

Baseline generated sample excerpt:

```text
ast wisins i is llak for efizen frioor in tic kn tie fors a usimng thock wilf the theer the thar thio yor ...
```

## Implemented state-space model

`state-space.py` now uses a compact diagonal state-space language model. The default block is Mamba/S4D-like:

- LayerNorm pre-normalization
- input projection
- causal depthwise local convolution
- diagonal state-space recurrence
- optional Mamba-style multiplicative gate
- output projection
- residual connection

The state recurrence is implemented as a causal convolution kernel over the fixed 64-token window:

```text
x_t = a * x_{t-1} + B * u_t
y_t = C * x_t + D * u_t
```

This is equivalent to the explicit diagonal recurrence for this fixed context length, but faster and easier to compare across variants.

Two implementation details mattered:

- The first version used an explicit Python loop over the 64 tokens. It produced good loss, but it made some trial runs awkward and slow to observe.
- The current version builds the diagonal SSM impulse response and applies grouped `conv1d`. It reproduced the same losses while making the implementation cleaner and fast enough to revisit higher-capacity settings.

## Experiments

All full runs below use the same train/eval split and tokenization. Short-screen results use `800` steps only and are included to explain why some variants were not promoted to full-run defaults.

| Variant | Config | Params | Steps | Eval loss | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Transformer baseline | existing `transformer.py` | 95,187 | 2,000 | 2.0823 | Baseline |
| Gated diagonal SSM | `n_embd=92`, `state_size=8` | 96,903 | 2,800 | 1.4715 | Strong |
| Gated diagonal SSM | `n_embd=93`, `state_size=8` | 98,793 | 2,800 | **1.4616** | Best full run |
| Gated diagonal SSM | `n_embd=94`, `state_size=7` | 99,855 | 2,800 | 1.4689 | Near cap, worse than 93x8 |
| Ungated S4D-style | `n_embd=108`, `state_size=8`, no gate | 93,987 | 800 | 1.8060 | Much worse in screen |
| S5-style dense state | `n_embd=96`, `s5_state_size=144` | 97,515 | 800 | 2.0424 | Much worse in screen |

## Full-run checkpoint curves

Gated diagonal SSM, `n_embd=92`, `state_size=8`, `96,903` parameters:

| Step | Train loss | Eval loss |
| ---: | ---: | ---: |
| 0 | 3.4154 | 3.4306 |
| 200 | 1.8151 | 1.8817 |
| 400 | 1.6276 | 1.7388 |
| 600 | 1.5544 | 1.6446 |
| 800 | 1.5124 | 1.6120 |
| 1000 | 1.4909 | 1.5649 |
| 1200 | 1.4759 | 1.5468 |
| 1400 | 1.4637 | 1.5245 |
| 1600 | 1.4554 | 1.5038 |
| 1800 | 1.4441 | 1.5026 |
| 2000 | 1.4313 | 1.4854 |
| 2200 | 1.4321 | 1.4830 |
| 2400 | 1.4328 | 1.4804 |
| 2600 | 1.4271 | 1.4783 |
| 2799 | 1.4262 | 1.4715 |

Gated diagonal SSM, `n_embd=93`, `state_size=8`, `98,793` parameters:

| Step | Train loss | Eval loss |
| ---: | ---: | ---: |
| 0 | 3.4688 | 3.4723 |
| 200 | 1.8155 | 1.8833 |
| 400 | 1.6231 | 1.7090 |
| 600 | 1.5520 | 1.6272 |
| 800 | 1.5060 | 1.5907 |
| 1000 | 1.4862 | 1.5494 |
| 1200 | 1.4664 | 1.5362 |
| 1400 | 1.4555 | 1.5089 |
| 1600 | 1.4477 | 1.4871 |
| 1800 | 1.4374 | 1.4875 |
| 2000 | 1.4251 | 1.4717 |
| 2200 | 1.4243 | 1.4700 |
| 2400 | 1.4247 | 1.4667 |
| 2600 | 1.4188 | 1.4643 |
| 2799 | 1.4200 | **1.4616** |

Gated diagonal SSM, `n_embd=94`, `state_size=7`, `99,855` parameters:

| Step | Train loss | Eval loss |
| ---: | ---: | ---: |
| 0 | 3.4546 | 3.4465 |
| 200 | 1.8035 | 1.8716 |
| 400 | 1.6154 | 1.7168 |
| 600 | 1.5481 | 1.6386 |
| 800 | 1.5037 | 1.6108 |
| 1000 | 1.4854 | 1.5669 |
| 1200 | 1.4668 | 1.5469 |
| 1400 | 1.4550 | 1.5299 |
| 1600 | 1.4479 | 1.5060 |
| 1800 | 1.4367 | 1.5074 |
| 2000 | 1.4254 | 1.4828 |
| 2200 | 1.4241 | 1.4804 |
| 2400 | 1.4247 | 1.4787 |
| 2600 | 1.4179 | 1.4759 |
| 2799 | 1.4196 | 1.4689 |

The `94x7` model used almost the entire parameter budget and looked slightly better in an 800-step screen, but the full run ended worse than `93x8`. That suggests the extra channel width did not compensate for reducing the per-channel state size.

## Short screens

These were used to reject variants cheaply before full training.

| Variant | Params | Step 400 eval | Step 799 eval | Generated sample quality |
| --- | ---: | ---: | ---: | --- |
| Gated diagonal SSM, `92x8` | 96,903 | 1.7576 | 1.6833 | recognizable words, still noisy |
| Ungated S4D-style, `108x8` | 93,987 | 1.8938 | 1.8060 | worse word formation |
| S5-style dense state, `96x144` | 97,515 | 2.1737 | 2.0424 | close to baseline quality |
| Gated diagonal SSM, `94x7` | 99,855 | 1.7442 | 1.6747 | slightly ahead in screen, failed to win full run |

## Mamba vs S4 vs S5

I did try all three families in compact form:

- **Mamba-style**: diagonal SSM plus causal local conv plus multiplicative gate. This won.
- **S4D-style**: same diagonal SSM/local conv, but no multiplicative gate. This was substantially worse in the 800-step comparison, even after using the saved parameters to widen the model.
- **S5-style**: dense input/output maps around a diagonal state vector. This trained, stayed under the cap, but was much worse in the 800-step comparison.

The important caveat is that these are small, dependency-free implementations tailored to this repo and parameter budget. They are not full reference S4/S5/Mamba implementations. Under this budget and dataset, the Mamba-like gate appears useful; removing it freed parameters, but the wider ungated model did not use those parameters effectively.

## Generation quality

The winning SSM still makes many character-level spelling errors, but it is qualitatively much more coherent than the transformer baseline. It produces recognizable Shakespeare-like word sequences and recurring names/phrases such as:

```text
and brows to my man that not friar let art thou ha children with mades these may come body and am but against ...
```

The generation quality tracks the eval loss improvement: it is not polished prose, but it is clearly beyond the baseline's mostly fragmented output.

The `94x7` near-cap model produced a similar but slightly less convincing sample:

```text
and brows to my man that not full mercutio now bold her think with when is whym catesby my lord more caly son ...
```

The ungated and S5-style screens had less stable word formation and did not justify full runs under the current constraints.

## Current recommendation

Use the default `state-space.py` configuration:

```python
Config(
    block_size=64,
    batch_size=96,
    n_embd=93,
    n_layer=3,
    state_size=8,
    block_type="mamba",
    use_gate=True,
    learning_rate=1.8e-3,
    min_learning_rate=2e-4,
    weight_decay=0.08,
    num_steps=2800,
)
```

Measured result:

- Parameters: `98,793`
- Final fixed eval loss: `1.4616`
- Training time: within the 5-minute budget on MPS after vectorizing the state-space recurrence

Useful commands:

```bash
uv run python modelling/transformer.py
uv run python modelling/state-space.py
uv run python -m py_compile modelling/state-space.py
```

The final syntax check passed, and the default parameter-count check returned `98,793`.

## Next ideas

The best remaining search space is likely training dynamics, not model family:

- tune dropout around `0.04-0.10`
- tune weight decay around `0.04-0.10`
- try slightly longer cosine tails or constant-plus-decay schedules
- evaluate whether more steps still fit after vectorization
- try small token dropout/span corruption only on train data

The architecture search so far suggests keeping the gate.
