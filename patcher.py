"""
patcher.py -- the patch encoder for the hierarchical byte model.
================================================================

First brick of the two-level ("Byte Latent"-style) architecture:

    bytes + boundaries  ->  ONE vector per patch

The boundaries come from analyze.py (surprise-based) or anywhere else -- this
module treats them as *given*. That deliberate separation lets us build and
test the hierarchy (encoder shapes, later decoder reconstruction) independently
of the boundary *policy* (fixed now, online/learned later).

Pipeline this feeds:
    PatchEncoder (here) -> big Griffin over the short patch sequence -> decoder
                                                                       (next brick)

Encoders provided:
    "mean" : byte-embed -> linear -> average within each patch. Order-agnostic,
             dead simple, a strong baseline and easy to verify.
    "gru"  : a small GRU over each patch's bytes; take the final state. Order-
             aware. (Swapping in a CG-LRU encoder later is a drop-in; a GRU is
             used first because it's boringly correct while we wire the shapes.)

Interface note: this first version encodes ONE sequence at a time (matching a
single analyze.py dump). Batching across sequences is a later extension; the
shapes here are written so that generalizing to [B, ...] is mechanical.
"""

import json
from dataclasses import dataclass

import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class PatcherConfig:
    vocab_size: int = 256
    d_model: int = 256        # patch-vector dim (match top-level Griffin d_model)
    d_byte: int = 128         # internal byte-embedding dim
    encoder: str = "mean"     # "mean" | "gru"
    gru_layers: int = 1
    # decoder (autoregressive byte generator conditioned on a patch vector)
    d_dec: int = 256          # decoder GRU hidden size
    dec_layers: int = 1


# --------------------------------------------------------------------------- #
# boundaries -> per-token patch id
# --------------------------------------------------------------------------- #
def boundaries_to_patch_ids(starts, T, device=None):
    """
    starts: sorted patch-start indices with starts[0] == 0.
    Returns patch_ids: [T] long, where patch_ids[t] = index of the patch owning t.

    e.g. starts=[0,3,7], T=9  ->  [0,0,0,1,1,1,1,2,2]
    """
    starts = torch.as_tensor(starts, dtype=torch.long, device=device)
    assert starts.numel() > 0 and int(starts[0]) == 0, "starts must be non-empty and begin at 0"
    mark = torch.zeros(T, dtype=torch.long, device=device)
    mark[starts] = 1                       # 1 at each patch start
    return torch.cumsum(mark, 0) - 1       # 0-based patch index per position


# --------------------------------------------------------------------------- #
# Patch encoder
# --------------------------------------------------------------------------- #
class PatchEncoder(nn.Module):
    def __init__(self, cfg: PatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_byte)
        if cfg.encoder == "mean":
            self.proj = nn.Linear(cfg.d_byte, cfg.d_model)
        elif cfg.encoder == "gru":
            self.gru = nn.GRU(cfg.d_byte, cfg.d_model,
                              num_layers=cfg.gru_layers, batch_first=True)
        else:
            raise ValueError(f"unknown encoder: {cfg.encoder!r}")
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def encode_batch(self, patches, lengths):
        """
        Batched, padded encode -- the shape all callers reduce to.
          patches : [N, Lmax] long byte ids (padding beyond `lengths` ignored)
          lengths : [N] long true patch lengths
        returns   : [N, d_model]
        """
        N, Lmax = patches.shape
        device = patches.device
        emb = self.embed(patches)                      # [N, Lmax, d_byte]
        if self.cfg.encoder == "mean":
            h = self.proj(emb)                         # [N, Lmax, d_model]
            mask = (torch.arange(Lmax, device=device)[None, :] < lengths[:, None]).to(h.dtype)
            summed = (h * mask.unsqueeze(-1)).sum(1)   # [N, d_model]
            return summed / lengths.clamp(min=1).unsqueeze(1).to(h.dtype)
        out, _ = self.gru(emb)                         # [N, Lmax, d_model]
        last = (lengths - 1).clamp(min=0).view(N, 1, 1).expand(N, 1, self.cfg.d_model)
        return out.gather(1, last).squeeze(1)          # [N, d_model]

    def forward(self, ids, starts):
        """
        Ragged interface: one contiguous byte sequence + patch-start indices.
          ids    : [T] long ; starts : sorted starts (starts[0]==0)
        returns  : [P, d_model]
        """
        ids = torch.as_tensor(ids, dtype=torch.long, device=self.embed.weight.device)
        T = ids.shape[0]
        device = ids.device
        starts_t = torch.as_tensor(starts, dtype=torch.long, device=device)
        P = starts_t.numel()
        ends = torch.cat([starts_t, torch.tensor([T], device=device)])
        lengths = ends[1:] - ends[:-1]                 # [P]
        Lmax = int(lengths.max())
        padded = torch.zeros(P, Lmax, dtype=torch.long, device=device)
        for k in range(P):
            s, L = int(starts_t[k]), int(lengths[k])
            padded[k, :L] = ids[s:s + L]
        return self.encode_batch(padded, lengths)


# --------------------------------------------------------------------------- #
# Patch decoder
# --------------------------------------------------------------------------- #
class PatchDecoder(nn.Module):
    """
    Autoregressive byte decoder conditioned on a single patch vector z.

    The patch vector both initializes the GRU hidden state AND is concatenated
    to every step's input, so the vector stays present the whole way through
    (rather than fading as the GRU runs). Teacher-forced for training; a greedy
    `generate` is provided for eyeballing reconstructions.

    This is the half that makes the bottleneck testable: if L bytes can't be
    recovered from one d_model vector, reconstruction accuracy will tell you.
    """

    def __init__(self, cfg: PatcherConfig):
        super().__init__()
        self.cfg = cfg
        self.BOS = cfg.vocab_size                         # extra start-of-patch id
        self.in_embed = nn.Embedding(cfg.vocab_size + 1, cfg.d_byte)
        self.cond = nn.Linear(cfg.d_model, cfg.d_dec * cfg.dec_layers)   # z -> h0
        self.gru = nn.GRU(cfg.d_byte + cfg.d_model, cfg.d_dec,
                          num_layers=cfg.dec_layers, batch_first=True)
        self.out = nn.Linear(cfg.d_dec, cfg.vocab_size)
        self.apply(PatchEncoder._init)

    def _h0(self, z):
        P = z.shape[0]
        h = self.cond(z).view(P, self.cfg.dec_layers, self.cfg.d_dec)
        return h.transpose(0, 1).contiguous()             # [layers, P, d_dec]

    def forward(self, z, tgt):
        """
        z   : [P, d_model] patch vectors
        tgt : [P, Lmax] padded target byte ids
        returns logits [P, Lmax, vocab] (teacher-forced; input is BOS + tgt[:-1])
        """
        P, Lmax = tgt.shape
        bos = torch.full((P, 1), self.BOS, dtype=torch.long, device=tgt.device)
        inp = torch.cat([bos, tgt[:, :-1]], dim=1)        # shift-right
        e = self.in_embed(inp)                            # [P, Lmax, d_byte]
        zc = z.unsqueeze(1).expand(-1, Lmax, -1)          # [P, Lmax, d_model]
        out, _ = self.gru(torch.cat([e, zc], dim=-1), self._h0(z))
        return self.out(out)                              # [P, Lmax, vocab]

    @torch.no_grad()
    def generate(self, z, lengths):
        """Greedy free-running reconstruction. z: [P,d_model]; lengths: [P]."""
        P = z.shape[0]
        Lmax = int(torch.as_tensor(lengths).max())
        device = z.device
        cur = torch.full((P, 1), self.BOS, dtype=torch.long, device=device)
        h = self._h0(z)
        outs = []
        for _ in range(Lmax):
            e = self.in_embed(cur)                        # [P,1,d_byte]
            step = torch.cat([e, z.unsqueeze(1)], dim=-1)
            y, h = self.gru(step, h)
            nxt = self.out(y[:, -1]).argmax(-1, keepdim=True)   # [P,1]
            outs.append(nxt)
            cur = nxt
        return torch.cat(outs, dim=1)                     # [P, Lmax]



def load_dump(path):
    with open(path) as f:
        d = json.load(f)
    return torch.tensor(d["ids"], dtype=torch.long), d["patch_starts"], d


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os

    torch.manual_seed(0)

    # Use a real dump if present, else synthesise bytes + boundaries.
    if os.path.exists("boundaries.json"):
        ids, starts, meta = load_dump("boundaries.json")
        print(f"loaded dump: {len(ids)} tokens, {len(starts)} patches "
              f"(kind={meta.get('kind')}, thr={meta.get('threshold'):.2f})")
    else:
        T = 40
        ids = torch.randint(0, 256, (T,))
        starts = [0, 5, 9, 20, 33]         # 5 patches of varying length
        print(f"synthetic: {T} tokens, {len(starts)} patches, starts={starts}")

    T = ids.shape[0]
    P = len(starts)

    # patch-id map sanity
    pid = boundaries_to_patch_ids(starts, T)
    assert pid.max().item() == P - 1 and pid[0].item() == 0
    lengths = [starts[k + 1] - starts[k] for k in range(P - 1)] + [T - starts[-1]]
    assert [(pid == k).sum().item() for k in range(P)] == lengths
    print(f"patch lengths: {lengths}")

    # --- mean encoder ---
    enc = PatchEncoder(PatcherConfig(d_model=64, d_byte=32, encoder="mean")).eval()
    with torch.no_grad():
        vecs = enc(ids, starts)
    assert vecs.shape == (P, 64), vecs.shape
    # verify pooling exactly against a manual segment mean of proj(embed(ids))
    with torch.no_grad():
        h = enc.proj(enc.embed(ids))
        manual = torch.stack([h[starts[k]:starts[k] + lengths[k]].mean(0) for k in range(P)])
    assert torch.allclose(vecs, manual, atol=1e-5), (vecs - manual).abs().max()
    print(f"mean encoder : out {tuple(vecs.shape)}  | pooling matches manual segment-mean")

    # --- gru encoder ---
    genc = PatchEncoder(PatcherConfig(d_model=64, d_byte=32, encoder="gru")).eval()
    with torch.no_grad():
        gvecs = genc(ids, starts)
    assert gvecs.shape == (P, 64), gvecs.shape
    # order-sensitivity: reversing a patch's bytes should change the GRU vector
    ids2 = ids.clone()
    ids2[starts[1]:starts[2]] = ids2[starts[1]:starts[2]].flip(0)
    with torch.no_grad():
        gvecs2 = genc(ids2, starts)
    changed = not torch.allclose(gvecs[1], gvecs2[1], atol=1e-5)
    same_else = torch.allclose(gvecs[0], gvecs2[0], atol=1e-5)
    print(f"gru encoder  : out {tuple(gvecs.shape)}  | reversing patch 1 changes it: "
          f"{changed} | other patches untouched: {same_else}")

    print("\nOK: patch encoder produces one vector per patch; interface is stable.")
