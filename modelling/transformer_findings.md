# Transformer Shakespeare Findings

## Fixed Constraints

- Sequence length stayed fixed at `64`.
- The train/eval split was unchanged: first 90% train, final 10% eval.
- Tokenization was unchanged: normalized lowercase character-level vocabulary from `shakespeare.py`.
- The model remained attention-based.
- Parameter count stayed under `100,000`.
- A full training run had to finish within 5 minutes on MPS.

## Baseline

The original `transformer.py` was run first without changing the model.

Configuration:

- `n_embd = 60`
- `n_head = 4`
- `n_layer = 2`
- MLP expansion ratio: `4x`
- `batch_size = 64`
- `num_steps = 2000`
- `learning_rate = 3e-4`
- `dropout = 0.1`
- Parameters: `95,187`

Result:

```text
step 1999 | train loss 2.0262 | test loss 2.0925
```

The generated text mostly captured character frequencies and occasional short word-like fragments, but had weak word structure and poor coherence.

## Best Architecture Change

The most useful architecture change was reallocating parameters away from the wide `4x` feed-forward network and into depth while tying the token embedding and output projection.

Final architecture:

- `n_embd = 60`
- `n_head = 4`
- `n_layer = 3`
- MLP expansion ratio: `2x`
- tied token embedding and LM head weights
- Parameters: `93,960`

This stayed below the baseline parameter count while adding an extra attention block.

## Training Changes

The final training setup uses:

- `batch_size = 128`
- `num_steps = 8000`
- AdamW
- `learning_rate = 2e-3`
- `min_learning_rate = 1e-4`
- `warmup_steps = 300`
- cosine learning-rate decay
- `weight_decay = 0.1`
- gradient clipping at `1.0`
- `dropout = 0.05`
- deterministic seed: `1337`

The final script also enforces the parameter budget with a runtime check.

## Experiment Results

| Experiment | Parameters | Steps | Eval batches | Test loss | Elapsed | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline | 95,187 | 2,000 | 20 | 2.0925 | ~17s | Original model |
| Wider tied 2-layer model | 91,224 | 6,000 | 50 | 1.5397 | ~212s | Big gain from schedule + tying |
| Wider tied 2-layer model | 91,224 | 10,000 | 50 | 1.4793 | ~250s observed | Better, but shallower than 3-layer |
| 3-layer, 60-wide model | 93,960 | 10,000 | 50 | 1.4593 | under test run budget estimate, later full run exceeded | Best loss found before hard timing check |
| 3-layer, batch 256 | 93,960 | 8,000 | 30 | 1.4680 | slower | Larger batch did not justify wall-clock cost |
| 3-layer, dropout 0.10 | 93,960 | 6,000 | 30 | 1.5987 | slower curve | Too much regularization |
| 3-layer, weight decay 0.01 | 93,960 | 6,000 | 30 | 1.5412 | slower curve | Worse than `0.1` weight decay |
| Final 3-layer model | 93,960 | 8,000 | 20 | 1.4689 | 179.6s | Best verified configuration under 5 minutes |

## Final Verified Run

The final default `transformer.py` run produced:

```text
device: mps
vocab size: 27
train batches: 7408
test batches: 822
parameters: 93,960
step    0 | lr 6.67e-06 | train loss 38.4778 | test loss 38.5910
step 2000 | lr 1.78e-03 | train loss 1.5814 | test loss 1.6821
step 4000 | lr 1.11e-03 | train loss 1.4653 | test loss 1.5272
step 6000 | lr 3.99e-04 | train loss 1.4370 | test loss 1.4792
step 7999 | lr 1.00e-04 | train loss 1.4155 | test loss 1.4689
elapsed: 179.6s
```

Generated sample:

```text
 with would hath father my procester thee powers of the lay next of york just usic grounds i have my depound sortant death here for to help of gill us officence you no vengeal this they over ourselves mine eterman tread lates did my suld such and all both name determockd cred that the rome like he so
```

The generated text is still limited by character-level modeling and the small parameter budget, but it is substantially more coherent than the baseline. It contains more recognizable names, function words, and Shakespeare-like local phrase structure.

## Main Findings

1. Weight tying is very valuable under the 100k parameter cap.
2. A deeper transformer with a smaller MLP outperformed a shallower model with a larger MLP.
3. The training schedule mattered as much as architecture: warmup plus cosine decay enabled a much higher peak learning rate.
4. `dropout = 0.10` was too strong for this setup; `0.05` worked better.
5. Larger batch size did not improve enough to justify the slower wall-clock time.
6. Intermediate evaluation frequency materially affected total runtime on MPS, so the final script evaluates less often while preserving a final fixed eval measurement.

## Verification

Syntax check:

```text
uv run python -m py_compile modelling/transformer.py
```

Final training command:

```text
uv run python modelling/transformer.py
```
