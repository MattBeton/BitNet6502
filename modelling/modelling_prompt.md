You are going to help to build the BEST language model possible on the shakespeare dataset. Use transformer.py as the starting point. You MUST keep the following things fixed:

- seq_len=64
- train/eval set split
- tokenization
- parameter count MUST be under 100k parameters.
- the model must be attention-based (transformer)
- one training run must finish within 5 minutes (this is generous; it currently takes only 20s)

You are allowed to change:

- architecture
- training dynamics, schedule
- train data ordering
- optimizer
- add data augmentation to train
- add any regularizers eg. dropout, weight decay

You will start by running transformers.py in its current state to understand the current state of things. You are on a macbook pro device, so you will use the MPS device to train. You will then act as a research scientist, iteratively changing things and seeing what modifications provide the best improvements to the model. The performance of the model should be measured by two things:

1. loss on fixed eval set
2. coherence of generated text at the end (less important, but should be a lagging metric of the above).