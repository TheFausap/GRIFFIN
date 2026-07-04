"""
autoencode.py -- feasibility test for the patch bottleneck.
===========================================================

The one question to answer before building the hierarchy:

    Can a single d_model vector hold a patch's worth of bytes well enough to
    reconstruct them?

We train the local codec alone -- PatchEncoder -> patch vector -> PatchDecoder
-> bytes -- and measure byte-level reconstruction accuracy on HELD-OUT patches.
No big Griffin, no boundaries policy: just the codec.

Two diagnostics that keep the result honest:

  * encoder sweep (mean vs gru): a mean-pooled vector is order-agnostic, so it
    literally cannot encode byte order within a patch. Watching mean plateau
    below gru is the proof that an order-aware encoder is required (gru now,
    CG-LRU later).

  * zeroed-vector ablation: reconstruct with z set to zero. If accuracy barely
    drops, the decoder is cheating via the byte-level prior and the vector isn't
    pulling its weight. The gap (real - zeroed) is the vector's real content.

Boundaries here are fixed-length (--patch_len) so we can sweep the bottleneck
cleanly; pass --dump to use real surprise-based patches instead.

Usage:
    python autoencode.py --file corpus_sample.txt --patch_len 6 --encoder gru
    python autoencode.py --file x.txt --sweep_len 2,4,6,8,12
    python autoencode.py --dump boundaries.json --encoder gru
"""

import argparse
import itertools
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from patcher import PatcherConfig, PatchEncoder, PatchDecoder

DEFAULT_TEXT = (
    "the quick brown fox jumps over the lazy dog. she sells seashells by the "
    "seashore. how much wood would a woodchuck chuck if a woodchuck could chuck "
    "wood. a screaming comes across the sky. it has happened before but there "
    "is nothing to compare it to now. "
) * 12


def build_patches(text, patch_len=None, dump=None):
    """Return a list of patches (each a list of byte ids)."""
    if dump:
        d = json.load(open(dump))
        ids, starts, T = d["ids"], d["patch_starts"], d["num_tokens"]
        lens = [starts[k + 1] - starts[k] for k in range(len(starts) - 1)] + [T - starts[-1]]
        return [ids[starts[k]:starts[k] + lens[k]] for k in range(len(starts))]
    ids = list(text.encode("utf-8"))
    return [ids[i:i + patch_len] for i in range(0, len(ids) - patch_len + 1, patch_len)]


def batch(patch_list, device):
    """List[List[int]] -> tensors for encoder (concat+starts) and decoder (padded)."""
    B = len(patch_list)
    lens = torch.tensor([len(p) for p in patch_list], device=device)
    ids_cat = torch.tensor(list(itertools.chain.from_iterable(patch_list)),
                           dtype=torch.long, device=device)
    offs = list(itertools.accumulate((len(p) for p in patch_list)))
    starts = torch.tensor([0] + offs[:-1], dtype=torch.long, device=device)
    Lmax = int(lens.max())
    tgt = torch.zeros(B, Lmax, dtype=torch.long, device=device)
    for i, p in enumerate(patch_list):
        tgt[i, :len(p)] = torch.tensor(p, device=device)
    return ids_cat, starts, tgt, lens


def run(patches, encoder_kind, d_model, steps, batch_size, lr, device, quiet=False):
    torch.manual_seed(0)
    cfg = PatcherConfig(d_model=d_model, d_byte=64, encoder=encoder_kind,
                        d_dec=d_model, dec_layers=1)
    enc = PatchEncoder(cfg).to(device)
    dec = PatchDecoder(cfg).to(device)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=lr)

    n = int(0.9 * len(patches))
    train, val = patches[:n], patches[n:]

    def loss_acc(pl, zero_z=False):
        ids_cat, starts, tgt, lens = batch(pl, device)
        z = enc(ids_cat, starts)
        if zero_z:
            z = torch.zeros_like(z)
        logits = dec(z, tgt)
        Lmax = tgt.shape[1]
        mask = torch.arange(Lmax, device=device)[None, :] < lens[:, None]
        lo, ta = logits[mask], tgt[mask]
        loss = F.cross_entropy(lo, ta)
        acc = (lo.argmax(-1) == ta).float().mean()
        return loss, acc.item()

    enc.train(); dec.train()
    for step in range(steps):
        bl = [train[i] for i in torch.randint(len(train), (batch_size,)).tolist()]
        loss, _ = loss_acc(bl)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    enc.eval(); dec.eval()
    with torch.no_grad():
        _, val_acc = loss_acc(val)
        _, val_acc_zero = loss_acc(val, zero_z=True)
    return val_acc, val_acc_zero, (enc, dec, cfg)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="")
    p.add_argument("--text", default="")
    p.add_argument("--dump", default="")
    p.add_argument("--patch_len", type=int, default=6)
    p.add_argument("--sweep_len", default="", help="comma list e.g. 2,4,6,8,12")
    p.add_argument("--encoder", default="both", choices=["mean", "gru", "both"])
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    text = (open(args.file, encoding="utf-8").read() if args.file
            else (args.text or DEFAULT_TEXT))
    encoders = ["mean", "gru"] if args.encoder == "both" else [args.encoder]
    lens = [int(x) for x in args.sweep_len.split(",")] if args.sweep_len else [args.patch_len]

    print(f"device={args.device}  d_model={args.d_model}  steps={args.steps}  "
          f"chance≈{1/256:.4f}\n")
    print(f"{'patch_len':>9} {'encoder':>7} {'recon_acc':>10} {'zeroed_z':>9} {'vector_gain':>12}")
    for L in lens:
        patches = build_patches(text, patch_len=L, dump=(args.dump or None))
        for ek in encoders:
            acc, acc0, _ = run(patches, ek, args.d_model, args.steps,
                               args.batch_size, args.lr, args.device)
            tag = "dump" if args.dump else L
            print(f"{str(tag):>9} {ek:>7} {acc:>10.3f} {acc0:>9.3f} {acc - acc0:>+12.3f}")
        if args.dump:
            break

    # Show a couple of reconstructions with the last-trained gru codec.
    patches = build_patches(text, patch_len=lens[-1], dump=(args.dump or None))
    acc, _, (enc, dec, cfg) = run(patches, "gru", args.d_model, args.steps,
                                  args.batch_size, args.lr, args.device)
    val = patches[int(0.9 * len(patches)):][:6]
    ids_cat, starts, tgt, lensb = batch(val, args.device)
    with torch.no_grad():
        z = enc(ids_cat, starts)
        gen = dec.generate(z, lensb)
    print("\nsample reconstructions (gru codec):")
    for i, p in enumerate(val):
        orig = bytes(p).decode("utf-8", "replace")
        recon = bytes(int(b) for b in gen[i, :len(p)].tolist()).decode("utf-8", "replace")
        print(f"  {orig!r:>16}  ->  {recon!r}")


if __name__ == "__main__":
    main()
