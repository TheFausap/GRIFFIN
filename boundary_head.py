"""
boundary_head.py -- Stage 2 (endogenous patching) support.
============================================================

Stage 1 (dynamic.py, precompute_boundaries.py) places patch boundaries using
a SEPARATE, frozen, pretrained-then-frozen byte LM, calibrated once, offline,
against the whole corpus. Stage 2 replaces that boundary SOURCE with a small
`BoundaryHead` -- a plain `Griffin` instance trained jointly, alongside the
hierarchical model, on the same byte windows it already samples each step.

`BoundaryHead` needs no bespoke architecture: `Griffin.forward` already
matches the `model(x) -> logits[B,T,V]` contract that dynamic.py's
`next_byte_entropy`/`segment_causal` and `HierByteLM.generate_dynamic` were
built against, and analyze.py / precompute_boundaries.py's own checkpoint
loaders (`Griffin(ckpt["cfg"]); model.load_state_dict(...)`). So a
`BoundaryHead` checkpoint is automatically compatible with every diagnostic
tool Stage 1 already has -- this module only adds the pieces Stage 1 never
needed because its entropy model was static:

  * batched_entropy_mask : the live, per-training-step version of
                            dynamic.segment_causal, batched over B and
                            returning a boolean mask (not a patch list) so it
                            plugs directly into the UNMODIFIED
                            build_ragged/forward_ragged.
  * recalibrate_threshold : a cheap, sampled-batch threshold refresh for the
                             pre-freeze window, while BoundaryHead is still
                             moving and its entropy distribution is drifting.
  * build_boundaries      : precompute_boundaries.py's whole-corpus
                             compute_surprise/solve_threshold pass, extracted
                             so the freeze-time boundary computation can reuse
                             it as a library call instead of a subprocess.

Design note (see RESULTS_ANNEX.md / the Stage-2 plan): BoundaryHead trains
with its OWN optimizer, fully separate from the hierarchical model's --
folding its loss into one shared optimizer/clip_grad_norm_ call would let its
early-training loss spike distort the main model's gradient-clip norm.
"""

import math

import torch
import torch.nn.functional as F

from dynamic import cap_patch_lengths, segment_causal
from precompute_boundaries import compute_surprise, solve_threshold, stream_signature

LN2 = math.log(2)


@torch.no_grad()
def batched_entropy_mask(model, x, threshold, Lcap=32):
    """
    x : [B, S] long byte ids.
    Returns mask [B, S] bool -- True where a patch starts, using the exact
    same rule precompute_boundaries.py / dynamic.segment_causal use: byte 0
    always starts a patch, byte j (j>=1) starts one iff the entropy of
    predicting it from bytes < j exceeds `threshold`. Lcap-capped the same
    way build_ragged is, so the result plugs directly into build_ragged.
    """
    B, S = x.shape
    logits = model(x)                                    # [B, S, V]
    logp = F.log_softmax(logits, dim=-1)
    ent = -(logp.exp() * logp).sum(-1) / LN2              # [B, S]; ent[:,t] predicts byte t+1

    mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
    mask[:, 0] = True
    if S > 1:
        mask[:, 1:] = ent[:, :-1] > threshold
    if Lcap:
        mask = cap_patch_lengths(mask, Lcap)
    return mask


def recalibrate_threshold(model, x_batches, target_len):
    """
    model      : the (possibly still-training) boundary head.
    x_batches  : iterable of [B, S] long byte tensors -- reuse whichever
                 train batches the caller already sampled this round, not a
                 fresh corpus scan (that's what build_boundaries is for, at
                 freeze time only).
    target_len : desired mean patch length.

    Returns the threshold (bits) solve_threshold would pick, computed over
    these sampled windows. solve_threshold itself is used unmodified; the
    `surp[1:]` slice inside it drops one pooled entry (harmless at the pool
    sizes this is meant to run at), not a real corpus's forced position 0.
    """
    ents = []
    with torch.no_grad():
        for x in x_batches:
            logits = model(x)                             # [B, S, V]
            logp = F.log_softmax(logits, dim=-1)
            ent = -(logp.exp() * logp).sum(-1) / LN2       # [B, S]
            ents.append(ent[:, :-1].reshape(-1).cpu())     # drop each window's last (no target)
    return solve_threshold(torch.cat(ents), target_len)


def build_boundaries(model, data, target_len, block=4096, ctx=256, batch=16, device=None):
    """
    Library version of precompute_boundaries.py main()'s core: run `model`
    (any Griffin-compatible byte LM) over `data`, solve a target-mean-patch-
    length threshold, and return (mask, meta) in the same shape main() would
    have saved to an .npz -- callable mid-training at a freeze point against
    a model that just stopped moving, not only once offline at the very
    start against an already-frozen checkpoint.
    """
    device = device or next(model.parameters()).device
    surp = compute_surprise(model, data, device, block, ctx, batch)
    thr = solve_threshold(surp, target_len)
    mask = surp > thr
    mask[0] = True
    n_bnd = int(mask.sum())
    mean_len = data.numel() / n_bnd
    sig = stream_signature(data)
    meta = dict(n_bytes=sig["n_bytes"], checksum=sig["checksum"], threshold=thr,
                target_len=target_len, mean_patch_len=mean_len, n_boundaries=n_bnd,
                ctx=ctx, block=block)
    return mask, meta


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    from eval_hook import load_flat_model
    from griffin_cglru import Griffin

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    ckpt = "entropy_model/best.pt"
    if os.path.exists(ckpt):
        model = load_flat_model(ckpt, device, lambda c: Griffin(c))
        print(f"loaded {ckpt} for self-test")
    else:
        from griffin_cglru import GriffinConfig
        cfg = GriffinConfig(vocab_size=256, d_model=64, d_rnn=96, depth=3,
                             head_dim=32, window_size=32, attn_period=3)
        model = Griffin(cfg).to(device).eval()
        print("no entropy_model/best.pt found; using a fresh random Griffin for the self-test")

    # --- batched_entropy_mask ≡ segment_causal ---
    torch.manual_seed(0)
    byte_ids = torch.randint(0, 256, (200,)).tolist()
    threshold = 3.0
    patches = segment_causal(model, byte_ids, threshold, device, Lcap=32)
    starts_ref = [0]
    acc = 0
    for p in patches[:-1]:
        acc += len(p)
        starts_ref.append(acc)
    mask_ref = torch.zeros(len(byte_ids), dtype=torch.bool)
    mask_ref[torch.tensor(starts_ref)] = True

    x = torch.tensor([byte_ids], dtype=torch.long, device=device)
    mask_batched = batched_entropy_mask(model, x, threshold, Lcap=32)[0].cpu()

    assert torch.equal(mask_ref, mask_batched), \
        f"mismatch: {int((mask_ref != mask_batched).sum())} positions differ"
    print(f"batched_entropy_mask ≡ segment_causal: OK ({len(byte_ids)} bytes, "
          f"{int(mask_ref.sum())} boundaries)")

    # --- recalibrate_threshold ≡ solve_threshold on the same pooled data ---
    torch.manual_seed(1)
    x_batches = [torch.randint(0, 256, (4, 64), device=device) for _ in range(5)]
    thr_a = recalibrate_threshold(model, x_batches, target_len=6.0)

    ents = []
    with torch.no_grad():
        for xb in x_batches:
            logits = model(xb)
            logp = F.log_softmax(logits, dim=-1)
            ent = -(logp.exp() * logp).sum(-1) / LN2
            ents.append(ent[:, :-1].reshape(-1).cpu())
    thr_b = solve_threshold(torch.cat(ents), target_len=6.0)

    assert thr_a == thr_b, f"recalibrate_threshold ({thr_a}) != solve_threshold ({thr_b})"
    print(f"recalibrate_threshold ≡ solve_threshold: OK (threshold {thr_a:.3f} bits)")

    print("\nOK: boundary_head's live mask/threshold machinery matches the "
          "Stage-1 reference functions it wraps.")
