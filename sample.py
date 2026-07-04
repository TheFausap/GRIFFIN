"""
Generate text from a trained Griffin / CG-LRU checkpoint.
=========================================================

Loads a checkpoint saved by train_verdict.py (weights + config + vocab) and
samples characters autoregressively.

Usage:
    python sample.py                                  # prompt "I ", 500 chars
    python sample.py --prompt "The " --n 800
    python sample.py --temperature 0.6 --top_k 40     # steadier output
    python sample.py --ckpt best.pt --device cpu

Note: this model has no recurrent-state cache, so each new token recomputes the
(cropped) context. That's fine for a few hundred tokens; --context bounds the
per-step cost and cost grows with it.
"""

import argparse

import torch
import torch.nn.functional as F

from griffin_cglru import Griffin, GriffinConfig  # noqa: F401 (GriffinConfig needed for unpickle)
from tokenizer import tokenizer_from_state, CharTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--prompt", default="I ")
    p.add_argument("--n", type=int, default=500, help="number of characters to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=0, help="0 = disabled; else keep top-k logits")
    p.add_argument("--context", type=int, default=512, help="max context chars fed per step")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # weights_only=False: the checkpoint stores the GriffinConfig dataclass + tokenizer.
    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg: GriffinConfig = ckpt["cfg"]
    if "tok" in ckpt:
        tok = tokenizer_from_state(ckpt["tok"])
    else:  # backward-compat: older checkpoints stored a raw {char: id} map
        tok = CharTokenizer(sorted(ckpt["stoi"], key=ckpt["stoi"].get))

    model = Griffin(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Encode prompt with the checkpoint's own tokenizer.
    ids = tok.encode(args.prompt) or [0]
    ids = torch.tensor([ids], dtype=torch.long, device=args.device)

    with torch.no_grad():
        for _ in range(args.n):
            ctx = ids[:, -args.context:]
            logits = model(ctx)[:, -1, :] / max(args.temperature, 1e-6)
            if args.top_k > 0:
                k = min(args.top_k, logits.size(-1))
                thresh = torch.topk(logits, k).values[..., -1, None]
                logits = logits.masked_fill(logits < thresh, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            ids = torch.cat([ids, nxt], dim=1)

    text = tok.decode(ids[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
