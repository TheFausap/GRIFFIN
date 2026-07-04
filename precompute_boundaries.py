"""
precompute_boundaries.py -- corpus-aligned entropy boundaries for Stage 1.
==========================================================================

Runs a FROZEN byte-level entropy model (a flat Griffin trained by
train_verdict.py --tokenizer byte) over the whole corpus, measures next-byte
predictive entropy at every position, and solves for the single threshold that
yields a target MEAN PATCH LENGTH globally (default 6, to match the fixed-patch
baseline). Dumps a byte-aligned boolean boundary mask + metadata that
hierarchical.py consumes in dynamic mode.

Why solve globally, not per-file: the entropy distribution shifts across books
(dialogue vs dense prose), so a percentile picked on one sample mis-sets the
compression ratio. Matching mean patch length is what makes the entropy-vs-fixed
comparison a test of boundary PLACEMENT rather than of compression.

The byte stream is reconstructed IDENTICALLY to hierarchical.load_bytes (sorted
*.txt, joined with a blank line, UTF-8) so the mask aligns byte-for-byte; a
checksum is stored and re-verified at train time.

Usage:
    python precompute_boundaries.py --corpus corpus --ckpt entropy_model/best.pt \
        --target_len 6 --out boundaries.npz
"""

import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from griffin_cglru import Griffin, GriffinConfig  # noqa: F401 (unpickle)
from tokenizer import tokenizer_from_state, CharTokenizer
from dynamic import stream_signature

LN2 = math.log(2)


def rebuild_stream(corpus_dir):
    """Reconstruct the exact byte stream hierarchical.load_bytes builds for a folder."""
    files = sorted(glob.glob(os.path.join(corpus_dir, "*.txt")))
    if not files:
        raise FileNotFoundError(f"no .txt files in {corpus_dir}")
    text = "\n\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
    raw = text.encode("utf-8")
    data = torch.tensor(list(raw), dtype=torch.long)
    return data, [os.path.basename(f) for f in files]


def load_entropy_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    tok = tokenizer_from_state(ck["tok"]) if "tok" in ck else \
        CharTokenizer(sorted(ck["stoi"], key=ck["stoi"].get))
    if tok.kind != "byte":
        raise SystemExit(f"entropy model tokenizer is '{tok.kind}', need a byte model.")
    model = Griffin(cfg).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok


@torch.no_grad()
def compute_surprise(model, data, device, block, ctx, batch):
    """
    surp[j] = H(next-byte dist predicting byte j | bytes < j)  in bits.
    Uses overlapping windows so every kept position has >= ctx bytes of left
    context (positions with too little context are recomputed by an earlier
    window). surp[0] is left 0 (position 0 is always a forced boundary anyway).
    """
    N = data.numel()
    surp = torch.zeros(N, dtype=torch.float32)
    stride = block - ctx
    assert stride > 0, "block must exceed ctx"
    starts = list(range(0, max(1, N - 1), stride))

    def run_window(s, w):                              # w: [Lw] on cpu
        Lw = w.numel()
        logits = model(w.to(device).unsqueeze(0))[0]   # [Lw, V]
        logp = F.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(-1) / LN2       # [Lw] bits; ent[t] ~ byte s+t+1
        ent = ent[:-1].cpu()                           # t=0..Lw-2 predict s+1..s+Lw-1
        lo = s + (ctx if s > 0 else 1)                 # first kept global index
        hi = s + Lw                                    # exclusive
        for j in range(lo, hi):
            surp[j] = ent[j - s - 1]

    # batch equal-length full windows for speed; handle ragged tail per-window
    i = 0
    full = [s for s in starts if s + block <= N]
    while i < len(full):
        chunk = full[i:i + batch]
        wins = torch.stack([data[s:s + block] for s in chunk])   # [b, block]
        logits = model(wins.to(device))                          # [b, block, V]
        logp = F.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(-1) / LN2                 # [b, block]
        ent = ent[:, :-1].cpu()
        for r, s in enumerate(chunk):
            lo = s + (ctx if s > 0 else 1)
            for j in range(lo, s + block):
                surp[j] = ent[r, j - s - 1]
        i += batch
    # tail windows that run past N
    for s in [s for s in starts if s + block > N]:
        w = data[s:N]
        if w.numel() >= 2:
            run_window(s, w)
    return surp


def solve_threshold(surp, target_len, sample=2_000_000, seed=0):
    """Threshold s.t. fraction of boundaries ~ 1/target_len (=> mean patch length target_len)."""
    s = surp[1:].numpy()                               # position 0 excluded (forced)
    q = 1.0 - 1.0 / target_len
    if s.size > sample:
        rng = np.random.default_rng(seed)
        s = s[rng.choice(s.size, size=sample, replace=False)]
    return float(np.quantile(s, q))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", required=True, help="folder of .txt (same one you train on)")
    p.add_argument("--ckpt", default="entropy_model/best.pt")
    p.add_argument("--target_len", type=float, default=6.0, help="target mean patch length")
    p.add_argument("--out", default="boundaries.npz")
    p.add_argument("--block", type=int, default=4096)
    p.add_argument("--ctx", type=int, default=256, help="min left-context bytes per kept position")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    data, files = rebuild_stream(args.corpus)
    N = data.numel()
    print(f"corpus: {len(files)} files, {N} bytes")

    model, _ = load_entropy_model(args.ckpt, args.device)
    print(f"entropy model loaded from {args.ckpt}; scanning corpus "
          f"(block={args.block}, ctx={args.ctx})...")
    surp = compute_surprise(model, data, args.device, args.block, args.ctx, args.batch)

    thr = solve_threshold(surp, args.target_len)
    mask = surp > thr
    mask[0] = True                                     # corpus start
    n_bnd = int(mask.sum())
    mean_len = N / n_bnd
    print(f"threshold {thr:.3f} bits -> {n_bnd} boundaries, "
          f"mean patch length {mean_len:.3f} (target {args.target_len})")
    # a quick within-vs-boundary entropy sanity check
    b_ent = surp[mask].mean().item()
    w_ent = surp[~mask][1:].mean().item()
    print(f"mean entropy at boundaries {b_ent:.2f} bits  vs  within-patch {w_ent:.2f} bits "
          f"(boundaries should be the higher-surprise bytes)")

    sig = stream_signature(data)
    meta = dict(n_bytes=sig["n_bytes"], checksum=sig["checksum"], threshold=thr,
                target_len=args.target_len, mean_patch_len=mean_len, n_boundaries=n_bnd,
                ctx=args.ctx, block=args.block, files=files)
    np.savez_compressed(args.out, mask=mask.numpy().astype(np.bool_),
                        meta=np.array(meta, dtype=object))
    print(f"wrote {args.out}  (aligned to {N} bytes; checksum {sig['checksum']})")


if __name__ == "__main__":
    main()
