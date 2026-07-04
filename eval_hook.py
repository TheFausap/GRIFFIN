"""
eval_hook.py -- like-for-like architectural-tax measurement.
============================================================

Scores the FROZEN flat entropy model's next-byte NLL, decomposed into
first-byte-of-patch vs within-patch, on the SAME val batches (same bytes,
same boundary mask) that the hierarchical model evaluates on.

Why: comparing hier `first`/`within` (from F.cross_entropy) against the flat
full-context model on IDENTICAL data isolates the bottleneck tax from
slice-difficulty and from intrinsic byte hardness. Do NOT compare against the
numbers analyze.py printed on a 1000-token single-file slice -- that mixes two
confounds. This runs on the real val distribution instead.

Crucial correctness point: pass the *same* first-byte mask you use to split the
hierarchical first/within, so both models are scored on exactly the same
positions.

No grad, no training. Import and call inside your existing eval_loss().
"""

import math
import torch
import torch.nn.functional as F

LN2 = math.log(2.0)


def load_flat_model(path, device, build_model):
    """
    path        : checkpoint, e.g. 'entropy_model/best.pt'
    device      : 'cuda' | 'cpu'
    build_model : callable(cfg) -> nn.Module -- reuse the EXACT constructor
                  analyze.py uses to rebuild the flat Griffin from ckpt['cfg'],
                  so the architecture matches the weights.
    Returns a frozen, eval-mode flat byte LM.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def flat_first_within(flat_model, x, first_tgt, pad_mask=None):
    """
    flat_model : frozen flat byte LM; flat_model(x) -> logits [B,S,V]
                 (or a tuple whose element 0 is logits).
    x          : [B,S] long byte ids -- the contiguous stream, i.e. the same
                 bytes the hierarchical encoder consumes before patching.
    first_tgt  : [B,S] bool -- True where TARGET byte s (s>=1) is the first byte
                 of a patch. Reuse the hier decomposition's own mask, or build
                 it from patch starts: first_tgt[b, start] = True for start>=1.
    pad_mask   : [B,S] bool, True for REAL (non-pad) bytes. Optional.

    Returns per-batch summed bits + counts (so you can aggregate across the
    whole val set before dividing). The flat model predicts target t+1 from
    position t, so target byte s maps to nll index s-1; position 0 has no
    predictor and is dropped -- exactly analyze.py's convention.
    """
    out = flat_model(x)
    logits = out[0] if isinstance(out, (tuple, list)) else out       # [B,S,V]

    logp = F.log_softmax(logits[:, :-1], dim=-1)                     # targets 1..S-1
    tgt  = x[:, 1:]                                                  # [B,S-1]
    nll  = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1) / LN2     # [B,S-1] bits

    fm    = first_tgt[:, 1:]                                         # align to targets
    valid = (pad_mask[:, 1:] if pad_mask is not None
             else torch.ones_like(fm, dtype=torch.bool))
    first_m  = fm & valid
    within_m = (~fm) & valid

    return {
        "first_bits":  nll[first_m].sum().item(),
        "first_n":     int(first_m.sum().item()),
        "within_bits": nll[within_m].sum().item(),
        "within_n":    int(within_m.sum().item()),
    }


def first_tgt_from_starts(starts_per_seq, B, S, device=None):
    """
    Build the [B,S] first-byte mask from per-sequence patch starts.
    starts_per_seq : list length B; each a list of patch-start indices (incl. 0).
    Only starts >= 1 mark a scorable first byte (position 0 has no predictor).
    """
    m = torch.zeros(B, S, dtype=torch.bool, device=device)
    for b, starts in enumerate(starts_per_seq):
        idx = [s for s in starts if 1 <= s < S]
        if idx:
            m[b, idx] = True
    return m
