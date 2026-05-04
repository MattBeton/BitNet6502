# Shakespeare State-Space LM 20k Findings

## Constraints

- Source file copied from `state-space.py` to `state-space_20k.py`
- Sequence length: `64`
- Dataset path, split, and tokenization: unchanged from `shakespeare.py`
- Parameter budget: strictly under `20,000`
- Device: Apple MPS
- Run limit: one training run under 5 minutes
- Main metric: fixed eval-set loss
- Secondary metric: generated text coherence

## Final recommendation

Use the default config in `state-space_20k.py`:

```python
Config(
    block_size=64,
    batch_size=128,
    n_embd=38,
    n_layer=4,
    state_size=5,
    block_type="s4d",
    use_gate=False,
    dropout=0.02,
    learning_rate=2.6e-3,
    min_learning_rate=1e-4,
    weight_decay=0.04,
    num_steps=6000,
    warmup_steps=300,
)
```

Measured result:

- Parameters: `19,939`
- Final fixed eval loss: `1.5464`
- Runtime: under 5 minutes on MPS

## Main finding

The 100k-parameter search favored a gated Mamba-style block. The 20k-parameter search did not. At this smaller budget, the best model was a deeper ungated S4D-style diagonal SSM:

- Removing the gate frees enough parameters for more layers.
- Four ungated S4D layers beat one near-capacity gated layer.
- S5-style dense-state variants were not competitive in this small-budget setting.

## Experiment table

| Variant | Config | Params | Steps | Eval loss | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| Gated Mamba-style | `n_embd=58`, `n_layer=1`, `state_size=8` | 19,109 | 900 | 1.7964 | Best initial gated shape |
| Ungated S4D-style | `n_embd=52`, `n_layer=2`, `state_size=5` | 19,683 | 900 | 1.8070 | Close early, worse than gated at 900 |
| S5-style | `n_embd=48`, `n_layer=2`, `s5_state_size=64` | 19,003 | 900 | 2.0760 | Rejected |
| Gated Mamba-style | `n_embd=59`, `n_layer=1`, `state_size=10` | 19,969 | 900 | 1.7734 | Best short gated screen |
| Gated Mamba-style | `n_embd=59`, `n_layer=1`, `state_size=10` | 19,969 | 2,200 | 1.7037 | Still improving |
| Gated Mamba-style | `n_embd=59`, `n_layer=1`, `state_size=10` | 19,969 | 5,000 | 1.6565 | Longer schedule helped |
| Gated Mamba-style | `n_embd=59`, `n_layer=1`, `state_size=10`, lower dropout/WD | 19,969 | 5,000 | 1.6495 | Better regularization |
| Gated Mamba-style | `n_embd=59`, `n_layer=1`, `state_size=10`, longer schedule | 19,969 | 8,000 | 1.6318 | Good, but beaten by deeper S4D |
| Ungated S4D-style | `n_embd=52`, `n_layer=2`, `state_size=5` | 19,683 | 8,000 | 1.5722 | Strong |
| Ungated S4D-style | `n_embd=52`, `n_layer=2`, `state_size=6` | 19,995 | 3,000 | 1.6439 | Worse than deeper shapes |
| Ungated S4D-style | `n_embd=43`, `n_layer=3`, `state_size=6` | 19,893 | 3,000 | 1.6085 | Depth helped |
| Ungated S4D-style | `n_embd=38`, `n_layer=4`, `state_size=5` | 19,939 | 3,000 | 1.6023 | Slightly better than 3-layer |
| Ungated S4D-style | `n_embd=33`, `n_layer=5`, `state_size=7` | 19,992 | 3,000 | 1.5999 | Similar, slower |
| Ungated S4D-style | `n_embd=38`, `n_layer=4`, `state_size=5` | 19,939 | 5,000 | 1.5568 | Best safe long screen |
| Ungated S4D-style | `n_embd=38`, `n_layer=4`, `state_size=5` | 19,939 | 6,000 | **1.5464** | Final default |

## Final checkpoint curve

Final model: `n_embd=38`, `n_layer=4`, `state_size=5`, `block_type="s4d"`, `19,939` parameters.

| Step | Train loss | Eval loss |
| ---: | ---: | ---: |
| 0 | 3.5130 | 3.4952 |
| 1200 | 1.6639 | 1.6612 |
| 2400 | 1.5923 | 1.5917 |
| 3600 | 1.5625 | 1.5644 |
| 4800 | 1.5553 | 1.5515 |
| 5999 | 1.5455 | **1.5464** |

Generated sample excerpt:

```text
out i with be against him gate or the parteer thing more friend my bettle the this death the cannot in that for over a brother ...
```

The text is still noisy, but it has recognizable phrase structure and is noticeably better than the weaker 20k screens.

## Notes

- The 20k budget makes parameter allocation much harsher than the 100k budget.
- The gate is not universally helpful. It won at 100k but lost at 20k because it consumed parameters that were better spent on depth.
- The final model remains a state-space model: it uses diagonal SSM kernels evaluated as grouped causal convolutions.
- The script has a hard guard that raises if the model reaches `20,000` parameters or more.

## Commands

```bash
uv run python modelling/state-space_20k.py
uv run python -m py_compile modelling/state-space_20k.py
```
