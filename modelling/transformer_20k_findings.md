# Transformer 20k Findings

## Setup

This round used the simplified Shakespeare dataset currently selected by `modelling/shakespeare.py`:

```text
modelling/data/shakespeare_speech_romeo_juliet.txt
```

The fixed constraints were:

- `seq_len = 64`
- unchanged train/eval split
- unchanged character-level tokenization
- attention-based transformer model
- parameter count under `20,000`
- full training run under 5 minutes on MPS

The working file was copied from `transformer.py` to:

```text
modelling/transformer_20k.py
```

## Final Result

The best verified model was:

- Parameters: `19,305`
- Runtime: `174.4s`
- Final train loss: `1.5160`
- Final test loss: `1.5115`

Final run:

```text
device: mps
vocab size: 27
train batches: 6854
test batches: 761
parameters: 19,305
step    0 | lr 1.00e-05 | train loss 30.8913 | test loss 30.5022
step 6000 | lr 2.32e-03 | train loss 1.5926 | test loss 1.5879
step 12000 | lr 8.47e-04 | train loss 1.5454 | test loss 1.5223
step 17999 | lr 1.00e-04 | train loss 1.5160 | test loss 1.5115
elapsed: 174.4s
```

Generated sample:

```text
 have indeed good now way day be quall you to foonforlhat i much with mumity which did brother whenced remon a soverands take o sostand you can brother not the twell you mean thou had murden beges one and with the earth you with thy subsed our me and romeo the let what youde noble indeeds in ever o s
```

## Final Configuration

Architecture:

- `n_embd = 39`
- `n_head = 3`
- `n_layer = 2`
- MLP hidden size: `39`
- tied token embedding and output projection
- fixed sinusoidal positional embedding
- bias-free attention and MLP linear layers
- affine-free LayerNorm

Training:

- `batch_size = 128`
- `num_steps = 18000`
- `learning_rate = 3e-3`
- `min_learning_rate = 1e-4`
- `warmup_steps = 300`
- cosine learning-rate decay
- AdamW
- `weight_decay = 0.01`
- `dropout = 0.0`
- gradient clipping at `1.0`
- `eval_interval = 6000`
- `eval_batches = 20`

## Experiment Log

| Experiment | Parameters | Steps | Test loss | Elapsed | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| 28-wide, 3-layer, learned positions | 17,556 | 10,000 | 1.6792 | 163.2s | First sub-20k baseline |
| 30-wide, 3-layer, sinusoidal positions | 17,970 | 10,000 | 1.6613 | 195.4s | Fixed positions helped |
| 38-wide, 2-layer, sinusoidal positions | 19,190 | 10,000 | 1.6485 | 147.2s | Wider and shallower was better |
| 39-wide, 2-layer, bias-free | 19,305 | 10,000 | 1.6434 | 198.1s | Bias/norm parameter savings helped |
| 39-wide, no dropout | 19,305 | 14,000 | 1.5328 | 182.5s | Large improvement; dropout was hurting |
| 39-wide, no dropout, 22k steps | 19,305 | 22,000 | 1.5128 | 350.3s | Best loss, but violated 5-minute limit |
| 39-wide, no dropout, 18k steps | 19,305 | 18,000 | 1.5194 | 293.5s | Under limit with default weight decay `0.1` |
| 39-wide, no dropout, weight decay `0.01` | 19,305 | 18,000 | 1.5115 | 174.4s | Best verified under 5 minutes |
| 39-wide, no dropout, no weight decay | 19,305 | 14,000 | 1.5392 | 136.7s | Worse than light weight decay |
| 40-wide, 4-head, narrower MLP | 19,640 | 18,000 | 1.5170 | 169.7s | Under budget but worse than 39-wide |

## Findings

1. The best use of the 20k budget was a shallow, wider transformer rather than a deeper, narrower one.
2. Learned positional embeddings were not worth the parameter cost at this scale; fixed sinusoidal embeddings improved loss while freeing parameters.
3. Removing linear biases and LayerNorm affine parameters made room for a wider model without hurting quality.
4. Dropout hurt substantially. The small model was underfitting, so `dropout = 0.0` worked best.
5. Light weight decay, `0.01`, was better than both heavy weight decay and no weight decay.
6. More training helped, but 22k steps exceeded the 5-minute limit. The final 18k-step schedule was the best verified tradeoff.
7. Sparse evaluation was important for runtime. Frequent evals consumed meaningful wall-clock time without improving the model.

## Verification

Syntax check:

```text
uv run python -m py_compile modelling/transformer_20k.py
```

Final training command:

```text
uv run python modelling/transformer_20k.py
```
