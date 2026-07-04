"""
Analyze next-token predictive entropy from a trained Griffin / CG-LRU model.
============================================================================

"Place boundaries where the model is surprised."

This runs a trained (byte-level) model over a span of text and, at every
position, measures the *predictive entropy* of the next-token distribution:

    H_t = -sum_v p(v | x_<=t) log2 p(v | x_<=t)      [bits]

High H_t means the model is uncertain about what comes next -- typically the
start of a new word or morpheme. Low H_t means it's mid-token and coasting.
That signal is exactly what a dynamic patcher would use to cut the byte stream
into variable-length patches, with NO fixed vocabulary: a boundary opens before
any token whose predictor was surprised (H_{t-1} > threshold).

Crucially the signal is *causal* -- it's read from the distribution before the
next byte is seen -- so the same rule works at generation time, not just here.

Usage:
    python analyze.py --file corpus_sample.txt
    python analyze.py --text "The quick brown fox jumps over the lazy dog."
    python analyze.py --file x.txt --color            # ANSI entropy heatmap
    python analyze.py --file x.txt --percentile 60    # more (lower-bar) boundaries
    python analyze.py --file x.txt --sweep            # threshold vs patch-length table
"""

import argparse
import json
import math

import torch
import torch.nn.functional as F

from griffin_cglru import Griffin, GriffinConfig  # noqa: F401 (needed for unpickle)
from tokenizer import tokenizer_from_state, CharTokenizer

LN2 = math.log(2)

DEFAULT_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "She sells seashells by the seashore, and the antidisestablishmentarianism "
    "of the committee was unmistakable."
)


def glyph(tok, i):
    """A single printable stand-in for token id `i` (keeps 1 glyph per token)."""
    if tok.kind == "byte":
        if i == 10:
            return "\n"
        if i == 9:
            return " "
        if 32 <= i < 127:
            return chr(i)
        return "\u00b7"  # non-printable byte / UTF-8 continuation byte
    c = tok.decode([i])
    if c == "\n":
        return "\n"
    if c == "\t":
        return " "
    return c if c.isprintable() else "\u00b7"


def color_for(norm):
    """Map normalized entropy in [0,1] to an xterm-256 fg code (blue->red)."""
    ramp = [27, 39, 45, 51, 46, 118, 226, 214, 208, 196]  # cool -> hot
    return ramp[min(len(ramp) - 1, max(0, int(norm * len(ramp))))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--file", default="")
    p.add_argument("--text", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--max_bytes", type=int, default=1000, help="analyze at most this many tokens")
    p.add_argument("--threshold", type=float, default=None, help="entropy (bits) boundary cutoff")
    p.add_argument("--percentile", type=float, default=70.0,
                   help="if --threshold unset, cut at this entropy percentile")
    p.add_argument("--color", action="store_true", help="ANSI entropy heatmap")
    p.add_argument("--sweep", action="store_true", help="print threshold vs patch-length table")
    p.add_argument("--top", type=int, default=12, help="show N highest-surprise positions")
    p.add_argument("--dump", default="", help="write bytes+boundaries to this JSON file (for patcher.py)")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = ckpt["cfg"]
    if "tok" in ckpt:
        tok = tokenizer_from_state(ckpt["tok"])
    else:
        tok = CharTokenizer(sorted(ckpt["stoi"], key=ckpt["stoi"].get))
    if tok.kind != "byte":
        print(f"note: checkpoint tokenizer is '{tok.kind}', not byte. "
              "Entropy analysis still works, but 'boundaries' are between "
              f"{tok.kind} tokens, not bytes.\n")

    model = Griffin(cfg).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Text source
    if args.file:
        text = open(args.file, encoding="utf-8").read()
    elif args.text:
        text = args.text
    else:
        text = DEFAULT_TEXT
    ids = tok.encode(text)[: args.max_bytes]
    if len(ids) < 2:
        raise SystemExit("need at least 2 tokens to analyze")
    x = torch.tensor([ids], dtype=torch.long, device=args.device)

    # Predictive entropy at each position (bits). logits[t] predicts token t+1.
    with torch.no_grad():
        logits = model(x)[0]                       # [T, vocab]
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1) / LN2       # [T] bits; ent[t] is surprise about token t+1
    ent = ent[:-1]                                 # drop last (no token to predict)
    ent_np = ent.detach().cpu()

    # Threshold
    if args.threshold is not None:
        thr = args.threshold
        thr_desc = f"{thr:.2f} bits (given)"
    else:
        thr = torch.quantile(ent_np, args.percentile / 100.0).item()
        thr_desc = f"{thr:.2f} bits (p{args.percentile:g})"

    # --- summary ---
    T = len(ids)
    # boundary before token j (j>=1) when ent[j-1] > thr
    boundary = [False] + [ent_np[j - 1].item() > thr for j in range(1, T)]
    n_bnd = sum(boundary)
    mean_patch = T / (n_bnd + 1)
    print(f"tokens analyzed : {T}")
    print(f"entropy (bits)  : mean {ent_np.mean():.2f} | median {ent_np.median():.2f} "
          f"| p90 {torch.quantile(ent_np, 0.9).item():.2f} | max {ent_np.max():.2f}")
    print(f"threshold       : {thr_desc}")
    print(f"boundaries      : {n_bnd}  ->  mean patch length {mean_patch:.2f} tokens "
          f"(latent sequence ~{mean_patch:.1f}x shorter)\n")

    # --- dump bytes + boundaries for the patcher (fixed-boundaries interface) ---
    if args.dump:
        starts = [0] + [j for j in range(1, T) if boundary[j]]
        lengths = [starts[k + 1] - starts[k] for k in range(len(starts) - 1)] + [T - starts[-1]]
        with open(args.dump, "w") as f:
            json.dump({
                "kind": tok.kind,
                "threshold": thr,
                "num_tokens": T,
                "ids": ids,
                "patch_starts": starts,     # patch k = ids[starts[k] : starts[k]+lengths[k]]
                "patch_lengths": lengths,
            }, f)
        print(f"dumped {len(starts)} patches over {T} tokens -> {args.dump}\n")

    # --- threshold sweep ---
    if args.sweep:
        print("percentile  threshold(bits)  boundaries  mean_patch_len")
        for pc in (50, 60, 70, 80, 90):
            t = torch.quantile(ent_np, pc / 100.0).item()
            nb = int((ent_np > t).sum().item())
            print(f"   p{pc:<7g} {t:>10.2f}     {nb:>8d}     {T/(nb+1):>10.2f}")
        print()

    # --- top surprise positions ---
    if args.top > 0:
        order = torch.argsort(ent_np, descending=True)[: args.top]
        print(f"top {args.top} surprises (token that the model was most uncertain about):")
        for rank, t in enumerate(sorted(order.tolist()), 0):
            j = t + 1                              # ent[t] predicts token j
            lo = max(0, j - 12)
            ctx = "".join(glyph(tok, ids[k]) for k in range(lo, j)).replace("\n", "\u21b5")
            nxt = glyph(tok, ids[j]).replace("\n", "\u21b5")
            print(f"  H={ent_np[t].item():4.2f}  ...{ctx!r} -> {nxt!r}")
        print()

    # --- boundary-marked text ---
    print("boundary view ( | = new patch starts here ):\n")
    parts = []
    for j, b in enumerate(ids):
        if boundary[j]:
            parts.append("\u2502")
        parts.append(glyph(tok, b))
    print("".join(parts))

    # --- optional color heatmap ---
    if args.color:
        emax = max(ent_np.max().item(), 1e-6)
        print("\n\nentropy heatmap (blue=confident -> red=surprised):\n")
        buf = []
        for j, b in enumerate(ids):
            g = glyph(tok, b)
            if j == 0:
                buf.append(g)
                continue
            norm = ent_np[j - 1].item() / emax
            buf.append(f"\x1b[38;5;{color_for(norm)}m{g}\x1b[0m")
        print("".join(buf))


if __name__ == "__main__":
    main()
