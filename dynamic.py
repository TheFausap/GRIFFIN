"""
dynamic.py -- Stage-1 dynamic (entropy-boundary) patching support.
==================================================================

Everything the hierarchical trainer needs to train on VARIABLE-length patches
whose boundaries were precomputed (by precompute_boundaries.py) from a frozen
byte-level entropy model. The design keeps the byte *window* fixed-length (S)
so the data loader is unchanged and BPC stays directly comparable to the
fixed-patch baseline; only the internal patch structure varies.

Four responsibilities:
  * load_boundary_mask   : load the corpus-aligned boolean mask, verify it
                           matches this exact byte stream (no silent misalign).
  * block_split_with_mask: the same deterministic block-interleaved split used
                           for the bytes, applied to the mask in lockstep so
                           train/val bytes and their boundaries stay aligned.
  * build_ragged         : turn a batch of byte windows + their boundary masks
                           into padded [B, Pmax, Lmax] patch tensors + masks.
  * forward_ragged       : run the hierarchical model over ragged patches with
                           a masked loss, and return the first-byte / within-
                           patch decomposition (the mandatory diagnostic).

Note on causality with padding: padded patches are always TRAILING (patch
indices >= P_b sit after every real patch in a window). The global Griffin is
causal (CG-LRU recurrence + causal local attention), so real positions never
attend to padded ones -- padded patches compute discarded garbage but cannot
corrupt real outputs. This is why no explicit masking is needed *inside* the
global model; masking only the loss suffices.
"""

import numpy as np
import torch
import torch.nn.functional as F

from patcher import prev_patch_tail


# --------------------------------------------------------------------------- #
# Mask I/O + alignment guard
# --------------------------------------------------------------------------- #
def stream_signature(data):
    """Cheap, order-sensitive signature of the byte stream for alignment checks."""
    n = data.numel()
    d = data.to(torch.int64)
    # length + a content checksum that depends on position (so permutations differ)
    idx = torch.arange(n, dtype=torch.int64)
    chk = int((d * (idx % 1009 + 1)).sum().item() % (2**61 - 1))
    return {"n_bytes": int(n), "checksum": chk}


def load_boundary_mask(path, data):
    """
    Load the boolean boundary mask and assert it aligns to `data` (the byte
    stream this trainer built). Returns a bool tensor [N] with mask[0]=True.
    """
    z = np.load(path, allow_pickle=True)
    mask = torch.from_numpy(z["mask"].astype(np.bool_))
    meta = z["meta"].item() if "meta" in z else {}
    sig = stream_signature(data)
    exp_n = int(meta.get("n_bytes", mask.numel()))
    if mask.numel() != data.numel() or exp_n != data.numel():
        raise ValueError(
            f"boundary mask length {mask.numel()} (meta n_bytes={exp_n}) != "
            f"byte stream length {data.numel()}. The corpus changed between "
            f"precompute and training -- regenerate boundaries.")
    if "checksum" in meta and meta["checksum"] != sig["checksum"]:
        raise ValueError(
            "boundary mask checksum mismatch: the byte stream differs from the "
            "one boundaries were computed on (same length, different content / "
            "file order). Regenerate boundaries against this corpus.")
    mask[0] = True  # corpus start is always a patch start
    return mask


# --------------------------------------------------------------------------- #
# Deterministic block-interleaved split, applied to bytes AND mask together
# (mirror of the split in hierarchical.main so bytes and boundaries stay aligned)
# --------------------------------------------------------------------------- #
def block_split_with_mask(data, mask, block, val_every):
    block = max(1, block)
    nb = len(data) // block
    d_blocks = data[:nb * block].view(nb, block)
    m_blocks = mask[:nb * block].view(nb, block)
    is_val = (torch.arange(nb) % val_every == val_every - 1)
    rem_d, rem_m = data[nb * block:], mask[nb * block:]
    train = torch.cat([d_blocks[~is_val].reshape(-1), rem_d])       # tail -> train
    train_m = torch.cat([m_blocks[~is_val].reshape(-1), rem_m])
    val = d_blocks[is_val].reshape(-1)
    val_m = m_blocks[is_val].reshape(-1)
    return train, train_m, val, val_m


# --------------------------------------------------------------------------- #
# ragged batch builder:  (bytes, boundary-mask) window batch -> padded patches
# --------------------------------------------------------------------------- #
def cap_patch_lengths(m, Lcap):
    """Split any patch longer than Lcap into Lcap-sized pieces.

    We insert a boundary at every Lcap-th byte MEASURED FROM THE LAST ORIGINAL
    boundary (rel % Lcap == 0), not wherever rel >= Lcap. The latter marks every
    byte past the cap as a boundary (because `last` never advances to inserted
    boundaries), shattering a long patch into 1-byte patches instead of capping
    it. Modulo from the gap start spaces the cuts exactly Lcap apart, so every
    resulting patch has length <= Lcap.
    """
    B, S = m.shape
    m = m.clone(); m[:, 0] = True
    pos = torch.arange(S, device=m.device).unsqueeze(0).expand(B, S)
    last = torch.cummax(torch.where(m, pos, torch.zeros_like(pos)), dim=1).values
    rel = pos - last
    return m | ((rel > 0) & (rel % Lcap == 0))


def build_ragged(x, m, Lcap=32):
    """
    x : [B, S] long byte ids for the sampled windows
    m : [B, S] bool boundary mask aligned to x (True = a patch starts here)
    Lcap : force a boundary at least every Lcap bytes (bounds Lmax -> bounds memory)

    Returns:
      patches : [B, Pmax, Lmax] long   (padded byte ids per patch)
      plens   : [B, Pmax]      long    (true length per patch; 0 for pad patches)
      pmask   : [B, Pmax]      bool     (True for real patches)
      bmask   : [B, Pmax, Lmax] bool    (True for real target bytes)
    The real patches of each window exactly tile [0, S): sum(plens[b]) == S.
    """
    B, S = x.shape
    device = x.device
    m = m.clone()
    m[:, 0] = True                                  # force a patch start at window pos 0
    if Lcap:
        m = cap_patch_lengths(m, Lcap)              # bound longest patch -> bound memory

    pid = torch.cumsum(m.long(), dim=1) - 1         # [B,S] patch index owning each position
    P_b = pid[:, -1] + 1                            # [B] number of patches per window
    Pmax = int(P_b.max())

    pos = torch.arange(S, device=device).unsqueeze(0).expand(B, S)
    start_of_pos = torch.cummax(torch.where(m, pos, torch.zeros_like(pos)), dim=1).values
    within = pos - start_of_pos                     # [B,S] offset within its patch
    Lmax = int(within.max()) + 1

    patches = torch.zeros(B, Pmax, Lmax, dtype=torch.long, device=device)
    bb = torch.arange(B, device=device).unsqueeze(1).expand(B, S)
    patches[bb, pid, within] = x                    # scatter bytes into (patch, offset)

    plens = torch.zeros(B, Pmax, dtype=torch.long, device=device)
    plens.scatter_add_(1, pid, torch.ones_like(pid))    # count = length per patch
    pmask = torch.arange(Pmax, device=device).unsqueeze(0) < P_b.unsqueeze(1)
    bmask = (torch.arange(Lmax, device=device).view(1, 1, Lmax) < plens.unsqueeze(-1)) \
        & pmask.unsqueeze(-1)
    return patches, plens, pmask, bmask


# --------------------------------------------------------------------------- #
# ragged forward with masked loss + first-byte / within-patch decomposition
# --------------------------------------------------------------------------- #
def forward_ragged(model, patches, plens, pmask, bmask):
    """
    Runs model.encoder -> model.global_model -> model.decoder over ragged
    patches. Returns (loss, aux) where loss is mean CE per real byte (directly
    comparable to the fixed-patch loss) and aux carries the diagnostic split.
    """
    B, Pmax, Lmax = patches.shape
    d = model.cfg.d_model
    V = model.cfg.vocab_size

    flat = patches.reshape(B * Pmax, Lmax)
    lengths = plens.reshape(B * Pmax).clamp(min=1)          # clamp: pad patches -> safe gather
    e = model.encoder.encode_batch(flat, lengths).view(B, Pmax, d)   # [B,Pmax,d]

    c = model.global_model(e)                               # causal over patches
    cond = torch.cat([model.start.expand(B, 1, -1), c[:, :-1]], dim=1)   # c through k-1

    K = model.cfg.byte_ctx_len
    prev_ctx = prev_patch_tail(patches, plens, K, model.decoder.BOS).reshape(B * Pmax, K)

    z = cond.reshape(B * Pmax, d)
    tgt = patches.reshape(B * Pmax, Lmax)
    logits = model.decoder(z, tgt, prev_ctx).view(B, Pmax, Lmax, V)

    ce = F.cross_entropy(logits.reshape(-1, V), tgt.reshape(-1),
                         reduction="none").view(B, Pmax, Lmax)

    # first byte of every real patch vs the rest
    first = torch.zeros_like(bmask)
    first[:, :, 0] = pmask                                  # offset 0 of real patches
    within = bmask & ~first

    nb = bmask.sum().clamp(min=1)
    loss = (ce * bmask).sum() / nb
    nf = first.sum().clamp(min=1)
    nw = within.sum().clamp(min=1)
    aux = {
        "loss_first": ((ce * first).sum() / nf).detach(),
        "loss_within": ((ce * within).sum() / nw).detach(),
        "frac_first": (first.sum() / nb).detach(),          # ~ 1 / mean_patch_len
        "mean_patch_len": (bmask.sum() / pmask.sum().clamp(min=1)).detach(),
    }
    return loss, aux
