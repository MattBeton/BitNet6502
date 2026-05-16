"""Sample text from a saved BitNet quant checkpoint.

Loads either a final or mid-training checkpoint (anything saved by
final_run.py or tinystories_run.py), reconstructs the right model
class from the stored stack flags (or inspects the state dict), and
generates `--max-new-tokens` characters from a prompt.

Usage:
    .venv/bin/python -u modelling/experiments/sample.py --ckpt build/bitnet_quant_tinystories_v1_ckpt.pt --prompt 'once upon a time '
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from common import Config  # noqa: E402
from final_run import StackedBitNetLM  # noqa: E402
from bitnet_quant import QuantBitNetLM  # noqa: E402


def reconstruct_model(state_dict: dict, cfg: Config, vocab) -> nn.Module:
    # Detect stack flags from the saved state dict shape.
    int4_head = "head.weight" in state_dict and state_dict["head.weight"].shape[1] == cfg.n_embd
    # Tied = head not present at all
    tie_emb = "head.weight" not in state_dict and "head_shift" in state_dict
    # If int4_ssm_C was on, the SSM C weight init range was (-7,7), but the
    # SHAPE is identical to ternary. We can't detect from shape alone; we
    # fall through using the saved cfg (won't be different at inference time
    # because both ternary_quantize and int4_quantize round and the weights
    # already lived in their valid range when saved).
    int4_C = False
    int4_conv = False
    # Fallback: try to infer from saved checkpoint metadata (we save 'stack')
    return StackedBitNetLM(
        vocab.size, cfg,
        int4_head=int4_head,
        int4_ssm_C=int4_C,
        int4_conv=int4_conv,
        tie_embeddings=tie_emb,
    )


def load_model(ckpt_path: Path) -> tuple[nn.Module, object, dict]:
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = blob["state_dict"]
    cfg = blob["cfg"]
    vocab = blob["vocab"]
    stack = blob.get("stack", {})

    # Dispatch: GRU checkpoints have 'hidden' in stack, or 'layers.*' in
    # state_dict. SSM checkpoints have 'blocks.*'.
    is_gru = ("hidden" in stack) or any(k.startswith("layers.") for k in sd.keys())

    if is_gru:
        from quant_gru import QuantGRULM
        model = QuantGRULM(
            vocab_size=vocab.size,
            hidden=stack.get("hidden", cfg.n_embd),
            n_layer=stack.get("n_layer", cfg.n_layer),
            block_size=cfg.block_size,
            int4_ih=stack.get("int4_ih", False),
            int4_hh=stack.get("int4_hh", False),
            head_precision=stack.get("head_precision", "ternary"),
            dropout=0.0,
        )
    else:
        model = StackedBitNetLM(
            vocab.size, cfg,
            int4_head=stack.get("int4_head", False) or "head.weight" in sd,
            int4_ssm_C=stack.get("int4_ssm_C", False),
            int4_conv=stack.get("int4_conv", False),
            tie_embeddings=stack.get("tie_embeddings", False),
        )
        # If 'stack' isn't present (mid-training ckpts may omit it), assume the
        # full v3 stack (int4 head + SSM C + conv) since that's what we train.
        if "stack" not in blob:
            model = StackedBitNetLM(
                vocab.size, cfg,
                int4_head=True, int4_ssm_C=True, int4_conv=True,
                tie_embeddings=False,
            )

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:3]}{'...' if len(missing)>3 else ''}", flush=True)
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:3]}{'...' if len(unexpected)>3 else ''}", flush=True)
    return model, vocab, blob


def sample(model: nn.Module, vocab, prompt: str, max_new_tokens: int = 300,
           greedy: bool = False, top_k: int = 8, temperature: float = 0.9,
           device: torch.device | None = None) -> str:
    device = device or (torch.device("mps") if torch.backends.mps.is_available()
                        else torch.device("cpu"))
    model = model.to(device).eval()

    stoi = vocab.stoi if hasattr(vocab, "stoi") else vocab["stoi"]
    itos = vocab.itos if hasattr(vocab, "itos") else vocab["itos"]

    # Map any out-of-vocab chars to space (vocab is the 27-char a-z + space set)
    ids = [stoi.get(c, stoi.get(" ", 0)) for c in prompt.lower()]
    if not ids:
        ids = [stoi.get(" ", 0)]
    idx = torch.tensor([ids], dtype=torch.long, device=device)

    out = model.generate(
        idx, max_new_tokens=max_new_tokens,
        temperature=temperature, top_k=top_k, greedy=greedy,
    )[0].tolist()
    return "".join(itos[i] for i in out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="once upon a time ")
    p.add_argument("--n", type=int, default=75,
                   help="tokens to generate per sample (kept short — long "
                        "generations drift OOD)")
    p.add_argument("--num-samples", type=int, default=5,
                   help="how many independent samples to draw")
    p.add_argument("--greedy", action="store_true",
                   help="if set, also include a greedy sample first "
                        "(deterministic baseline)")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; each sample uses seed+i")
    args = p.parse_args()

    model, vocab, blob = load_model(Path(args.ckpt))
    step = blob.get("step", "final")
    losses = blob.get("losses", {})
    print(f"loaded {args.ckpt}  step={step}  losses={losses}")
    print(f"prompt: {args.prompt!r}  ({args.n} tokens × {args.num_samples} samples)")
    print()

    if args.greedy:
        torch.manual_seed(args.seed)
        text = sample(model, vocab, args.prompt,
                      max_new_tokens=args.n,
                      greedy=True)
        print(f"[greedy] {text}")
        print()

    for i in range(args.num_samples):
        torch.manual_seed(args.seed + i)
        text = sample(model, vocab, args.prompt,
                      max_new_tokens=args.n,
                      greedy=False,
                      top_k=args.top_k,
                      temperature=args.temperature)
        print(f"[{i+1}] {text}")
        print()


if __name__ == "__main__":
    main()
