"""
hierarchical.py -- the two-level byte language model.
=====================================================

Assembles the bricks into an end-to-end model:

    bytes -> PatchEncoder -> [P patch vectors]
          -> Griffin (global, over patches, causal)
          -> [P context vectors]
          -> PatchDecoder (per patch, autoregressive over its bytes)
          -> byte logits

Causal factorization (no leakage):
    * The global Griffin is causal, so context c_k summarizes patches 0..k.
    * Patch k's bytes are decoded from c_{k-1} (context through the PREVIOUS
      patch) plus teacher-forced earlier bytes of patch k. Patch 0 uses a
      learned start vector. So every byte is predicted strictly from the past:
      earlier patches through the global model, earlier bytes within the patch
      through the decoder.

This first version uses FIXED-LENGTH patches (--patch_len). That keeps every
tensor rectangular ([B, P, L]) so batching is trivial, and -- importantly --
makes autonomous generation well defined with no end-of-patch token: each patch
is exactly L bytes. Variable / surprise-based boundaries are the next step and
need ragged batching; the encoder/decoder already support ragged inputs, so
that's an extension, not a rewrite.

The global model reuses the exact Griffin components (CG-LRU recurrent blocks +
local attention) -- so the architecture under test is unchanged, just applied
one level up, to patches instead of bytes.

Usage:
    python hierarchical.py --file corpus/pg11937.txt --patch_len 6 --steps 2000
    python hierarchical.py --text "..." --patch_len 4 --device cpu
"""

import argparse
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from griffin_cglru import (Griffin, GriffinConfig, ResidualBlock, RecurrentBlock,
                           LocalMQA, RMSNorm)
from patcher import PatcherConfig, PatchEncoder, PatchDecoder, prev_byte_window

from dynamic import (load_boundary_mask, block_split_with_mask,
                     build_ragged, forward_ragged, cap_patch_lengths,
                     load_threshold, next_byte_entropy, segment_causal)
from eval_hook import load_flat_model, flat_first_within
from boundary_head import batched_entropy_mask, recalibrate_threshold, build_boundaries

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class HierConfig:
    vocab_size: int = 256
    d_model: int = 256
    patch_len: int = 6
    # global Griffin (operates over PATCHES; window_size is in patches)
    d_rnn: int = 384
    depth: int = 6
    head_dim: int = 64
    window_size: int = 128
    attn_period: int = 3
    dropout: float = 0.0
    parallel_scan: bool = True
    # local codec
    d_byte: int = 128
    d_dec: int = 256
    encoder: str = "gru"
    dec_layers: int = 1
    byte_ctx_len: int = 8    # trailing raw bytes of the previous patch fed to the decoder

    def griffin(self):
        return GriffinConfig(
            vocab_size=self.vocab_size, d_model=self.d_model, d_rnn=self.d_rnn,
            depth=self.depth, head_dim=self.head_dim, window_size=self.window_size,
            attn_period=self.attn_period, dropout=self.dropout,
            parallel_scan=self.parallel_scan)

    def patcher(self):
        return PatcherConfig(
            vocab_size=self.vocab_size, d_model=self.d_model, d_byte=self.d_byte,
            encoder=self.encoder, d_dec=self.d_dec, dec_layers=self.dec_layers,
            byte_ctx_len=self.byte_ctx_len)


# --------------------------------------------------------------------------- #
# Global model: Griffin backbone over patch vectors (no embed / no unembed)
# --------------------------------------------------------------------------- #
class GlobalModel(nn.Module):
    def __init__(self, gcfg: GriffinConfig):
        super().__init__()
        blocks = []
        for i in range(gcfg.depth):
            temporal = LocalMQA(gcfg) if (i + 1) % gcfg.attn_period == 0 else RecurrentBlock(gcfg)
            blocks.append(ResidualBlock(gcfg, temporal))
        self.blocks = nn.ModuleList(blocks)
        self.norm_f = RMSNorm(gcfg.d_model)

    def forward(self, x):                              # [B, P, d] -> [B, P, d]
        for blk in self.blocks:
            x = blk(x)
        return self.norm_f(x)


# --------------------------------------------------------------------------- #
# Hierarchical byte LM
# --------------------------------------------------------------------------- #
class HierByteLM(nn.Module):
    def __init__(self, cfg: HierConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = PatchEncoder(cfg.patcher())
        self.global_model = GlobalModel(cfg.griffin())
        self.decoder = PatchDecoder(cfg.patcher())
        self.start = nn.Parameter(torch.zeros(1, 1, cfg.d_model))  # context before patch 0

        # Match Griffin's init regime for the global blocks.
        from griffin_cglru import CGLRU
        for m in self.global_model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for m in self.global_model.modules():
            if isinstance(m, CGLRU):
                m.reset_parameters()

    def _encode_patches(self, patches):                # patches: [B, P, L] -> [B, P, d]
        B, P, L = patches.shape
        flat = patches.reshape(B * P, L)
        lengths = torch.full((B * P,), L, dtype=torch.long, device=patches.device)
        e = self.encoder.encode_batch(flat, lengths)   # [B*P, d]
        return e.view(B, P, self.cfg.d_model)

    def forward(self, tokens):
        """tokens: [B, S] byte ids, S a multiple of patch_len. Returns (logits, loss)."""
        B, S = tokens.shape
        L = self.cfg.patch_len
        assert S % L == 0, f"sequence length {S} not a multiple of patch_len {L}"
        P = S // L
        patches = tokens.view(B, P, L)

        e = self._encode_patches(patches)              # [B, P, d]
        c = self.global_model(e)                       # [B, P, d]  (causal over patches)

        # condition for patch k = context through patch k-1 (start vector for k=0)
        cond = torch.cat([self.start.expand(B, 1, -1), c[:, :-1]], dim=1)   # [B, P, d]

        K = self.cfg.byte_ctx_len
        patch_starts = (torch.arange(P, device=tokens.device) * L).unsqueeze(0).expand(B, P)
        prev_ctx = prev_byte_window(tokens, patch_starts, K, self.decoder.BOS).reshape(B * P, K)

        z = cond.reshape(B * P, self.cfg.d_model)
        tgt = patches.reshape(B * P, L)
        logits = self.decoder(z, tgt, prev_ctx).view(B, P, L, self.cfg.vocab_size)
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), tokens.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, n_patches, device, prompt=b"", temperature=0.0, top_k=0, top_p=0.0):
        """Autonomous generation: emit n_patches new patches of L bytes each.
        Greedy (argmax) by default; pass temperature>0 (optionally with
        top_k/top_p) to sample instead -- see patcher.sample_from_logits."""
        L = self.cfg.patch_len
        K = self.cfg.byte_ctx_len
        pad_id = self.decoder.BOS
        # seed bytes (truncate prompt to a whole number of patches)
        seed = list(prompt)[: (len(prompt) // L) * L]
        out = list(seed)
        self.eval()
        for _ in range(n_patches):
            if out:
                P = len(out) // L
                patches = torch.tensor(out[: P * L], device=device).view(1, P, L)
                e = self._encode_patches(patches)
                c = self.global_model(e)
                cond = c[:, -1]                        # context through last patch
            else:
                cond = self.start[:, 0]                # [1, d]
            # trailing K bytes of the stream so far -- spans back across as
            # many previous patches as needed, not just the last one
            ctx = [pad_id] * max(0, K - len(out)) + out[-K:]
            prev_ctx = torch.tensor([ctx], dtype=torch.long, device=device)
            gen = self.decoder.generate(cond, torch.tensor([L], device=device), prev_ctx,
                                         temperature=temperature, top_k=top_k, top_p=top_p)  # [1,L]
            out.extend(int(b) for b in gen[0].tolist())
        return bytes(b & 0xFF for b in out)

    def _encode_variable(self, patches_list):
        """patches_list: list[list[int]], P patches of possibly different
        lengths. Returns e: [1, P, d] (the ragged counterpart of
        _encode_patches, for a single online generation stream)."""
        P = len(patches_list)
        Lmax = max(len(p) for p in patches_list)
        device = self.start.device
        buf = torch.zeros(P, Lmax, dtype=torch.long, device=device)
        lens = torch.zeros(P, dtype=torch.long, device=device)
        for k, p in enumerate(patches_list):
            buf[k, :len(p)] = torch.tensor(p, device=device)
            lens[k] = len(p)
        e = self.encoder.encode_batch(buf, lens)               # [P, d]
        return e.view(1, P, self.cfg.d_model)

    @torch.no_grad()
    def generate_dynamic(self, n_bytes, device, entropy_model, threshold, prompt=b"", Lcap=32,
                          temperature=0.0, top_k=0, top_p=0.0):
        """
        Autonomous generation for the DYNAMIC (entropy-boundary) model. Patch
        length isn't fixed like `generate`'s n_patches*L -- after every
        emitted byte, `entropy_model` (the SAME frozen model and `threshold`
        boundaries.npz was built from) is asked, online and causally, whether
        the byte just emitted closes the current patch. This reproduces the
        offline training-time boundary rule exactly (see dynamic.py's
        next_byte_entropy/segment_causal), so an already-trained dynamic
        checkpoint generates autonomously with no retraining. Greedy (argmax)
        by default; pass temperature>0 (optionally with top_k/top_p) to sample
        instead -- see patcher.sample_from_logits.
        """
        self.eval()
        K = self.cfg.byte_ctx_len
        pad_id = self.decoder.BOS

        closed = segment_causal(entropy_model, list(prompt), threshold, device, Lcap)
        out = list(prompt)
        cur_patch = []

        def cond_and_ctx():
            if closed:
                e = self._encode_variable(closed)
                c = self.global_model(e)
                cond = c[:, -1]                     # context through the last closed patch
            else:
                cond = self.start[:, 0]
            # trailing K bytes of the stream so far -- spans back across as
            # many previous patches as needed, not just the last closed one
            ctx = [pad_id] * max(0, K - len(out)) + out[-K:]
            prev_ctx = torch.tensor([ctx], dtype=torch.long, device=device)
            return cond, prev_ctx

        cond, prev_ctx = cond_and_ctx()
        cur, h, z = self.decoder.start_state(cond, prev_ctx)

        while len(out) < n_bytes:
            nxt, h = self.decoder.step(cur, h, z, temperature=temperature, top_k=top_k, top_p=top_p)
            b = int(nxt.item())
            out.append(b); cur_patch.append(b)
            cur = nxt

            ent = next_byte_entropy(entropy_model, out, device)
            if ent > threshold or len(cur_patch) >= Lcap:
                closed.append(cur_patch)
                cur_patch = []
                cond, prev_ctx = cond_and_ctx()
                cur, h, z = self.decoder.start_state(cond, prev_ctx)

        return bytes(x & 0xFF for x in out[:n_bytes])

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------- #
# Training harness (byte-level, fixed patches)
# --------------------------------------------------------------------------- #
def load_bytes(path, text, default):
    """Accept a single file, a folder of .txt files, or inline/default text."""
    if path and os.path.isdir(path):
        import glob
        files = sorted(glob.glob(os.path.join(path, "*.txt")))
        if not files:
            raise FileNotFoundError(f"no .txt files found in directory {path}")
        # Join with a blank line so byte offsets between files aren't glued together.
        text = "\n\n".join(open(f, encoding="utf-8", errors="replace").read() for f in files)
        raw = text.encode("utf-8")
        print(f"loaded {len(files)} file(s) from {path}/")
        return torch.tensor(list(raw), dtype=torch.long), len(raw), text
    if path:
        raw = open(path, "rb").read()
        text = raw.decode("utf-8", errors="replace")
        return torch.tensor(list(raw), dtype=torch.long), len(raw), text
    text = text or default
    raw = text.encode("utf-8")
    return torch.tensor(list(raw), dtype=torch.long), len(raw), text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--file", default="")
    p.add_argument("--text", default="")
    p.add_argument("--patch_len", type=int, default=6)
    p.add_argument("--patches", type=int, default=32, help="patches per training sequence")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_skips", type=int, default=50,
                   help="abort if this many non-finite steps are skipped")
    p.add_argument("--ckpt_interval", type=int, default=500,
                   help="save a resumable last.pt every N steps")
    p.add_argument("--resume", default="", help="path to a checkpoint to resume from")
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_block", type=int, default=65536,
                   help="block size in bytes for the interleaved val split")
    p.add_argument("--val_every", type=int, default=10,
                   help="route every Nth block to val (10 gives a 10 percent holdout)")
    p.add_argument("--boundaries", default="",
                   help="path to boundaries.npz -> dynamic entropy patching; empty = fixed")
    p.add_argument("--ckpt_tag", default="",
                   help="checkpoint suffix, e.g. _dyn (keeps the fixed best.pt safe)")
    p.add_argument("--byte_ctx_len", type=int, default=8,
                   help="trailing raw-byte window fed to the decoder as cross-patch context "
                        "(spans back across as many previous patches as needed, not just the "
                        "last one -- see patcher.prev_byte_window)")
    p.add_argument("--eval_batches", type=int, default=20,
                   help="val batches averaged per eval; bump to ~50-100 to shrink eval noise "
                        "when comparing runs whose gap is close to the per-step noise floor")
    p.add_argument("--entropy_ckpt", default="",
                   help="frozen flat byte model (e.g. entropy_model/best.pt); if set, eval "
                        "also scores it first/within on the SAME boundaries as this run -- "
                        "the real dynamic mask when --boundaries is set (the tax gate), or "
                        "a synthetic stride-patch_len mask when running fixed (the fixed "
                        "first/within baseline) -- so both land in one eval.")
    p.add_argument("--endogenous", action="store_true",
                   help="Stage 2: place boundaries via a small BoundaryHead trained jointly "
                        "instead of a precomputed --boundaries mask; mutually exclusive with "
                        "--boundaries. Pre-freeze, boundaries are computed live each step from "
                        "the (still-training) head; at --boundary_freeze_step it stops updating, "
                        "one real whole-corpus scan runs (reusing precompute_boundaries.py's own "
                        "compute_surprise/solve_threshold), and the rest of the run collapses "
                        "into the exact static-mask path --boundaries already uses.")
    p.add_argument("--boundary_target_len", type=float, default=6.0,
                   help="mean patch length the boundary head's threshold is calibrated to")
    p.add_argument("--boundary_freeze_step", type=int, default=None,
                   help="required with --endogenous. A standalone train_verdict.py + analyze.py "
                        "sweep on this corpus (small preset, batch_size=32/block_size=128) found "
                        "the boundary-quality gate passes best around step 1000, then degrades "
                        "(the same undertraining-signal-loss effect the external entropy model's "
                        "own design guards against) -- translated to this trainer's default "
                        "batch_size=16 * (patch_len*patches)=192 byte budget, that's ~1300 steps. "
                        "Retune if you change --batch_size/--patches/--patch_len.")
    p.add_argument("--boundary_recalib_interval", type=int, default=200,
                   help="pre-freeze: refresh the threshold every N steps from a handful of "
                        "freshly-sampled train batches (cheap; the whole-corpus scan only runs "
                        "once, at freeze)")
    p.add_argument("--boundary_calib_batches", type=int, default=20,
                   help="train batches sampled per threshold recalibration")
    p.add_argument("--boundary_lr", type=float, default=1e-3,
                   help="BoundaryHead has its OWN optimizer/clip_grad_norm_, fully separate from "
                        "the main model's -- folding it into one shared optimizer would let its "
                        "early-training loss spike distort the main model's gradient-clip norm")
    p.add_argument("--boundary_d_model", type=int, default=128)
    p.add_argument("--boundary_depth", type=int, default=4)
    p.add_argument("--boundary_head_dim", type=int, default=64)
    p.add_argument("--boundary_window", type=int, default=64)
    p.add_argument("--boundary_scan_block", type=int, default=1024,
                   help="freeze-time whole-corpus scan window (precompute_boundaries.py's own "
                        "default of 4096 can OOM on 8GB GPUs; 1024 is the size this corpus's own "
                        "boundaries.npz was actually built with)")
    p.add_argument("--boundary_scan_ctx", type=int, default=256)
    p.add_argument("--boundary_scan_batch", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="end-of-run sample: 0 = greedy (argmax, the old default); >0 samples "
                        "from the softmax at this temperature -- see patcher.sample_from_logits")
    p.add_argument("--top_k", type=int, default=0, help="0 = disabled; else keep top-k logits")
    p.add_argument("--top_p", type=float, default=0.0,
                   help="0 = disabled; else nucleus-sample the smallest top set with "
                        "cumulative probability >= top_p")
    p.add_argument("--Lcap", type=int, default=32,
                   help="max length any single dynamic/endogenous patch is allowed to reach "
                        "before being force-cut, regardless of the entropy threshold (bounds "
                        "memory in build_ragged). Must stay comfortably above --target_len / "
                        "--boundary_target_len -- if patch_len approaches this value, raise it, "
                        "or a growing fraction of patches will hit the cap instead of the "
                        "entropy rule and the patch-length distribution will be distorted.")
    args = p.parse_args()

    if args.endogenous:
        assert not args.boundaries, "--boundaries and --endogenous are mutually exclusive"
        assert args.boundary_freeze_step is not None, \
            "--endogenous requires --boundary_freeze_step (see its help for a starting value)"

    torch.manual_seed(args.seed)
    device = args.device
    default = ("the quick brown fox jumps over the lazy dog. she sells seashells "
               "by the seashore. a screaming comes across the sky. ") * 40

    data, n_bytes, text = load_bytes(args.file, args.text, default)
    # bytes/char to report BPC comparably to the flat byte baseline
    bytes_per_char = n_bytes / max(1, len(text))

    S = args.patch_len * args.patches

    # Deterministic block-interleaved train/val split. The old contiguous split
    # (last 10%) made val a slice of whichever book sorts last -- so train and val
    # came from different distributions. Instead, chop the stream into fixed blocks
    # and route every Nth block to val, giving an even sample across all files.
    # No RNG and no seed, so the split is identical on every run and survives
    # adding/renaming/re-chunking files.
    block = max(S, args.val_block)                    # block must hold one window
    nb = len(data) // block
    blocks = data[:nb * block].view(nb, block)
    is_val = (torch.arange(nb) % args.val_every == args.val_every - 1)
    val = blocks[is_val].reshape(-1)
    train = torch.cat([blocks[~is_val].reshape(-1), data[nb * block:]])  # tail -> train
    print(f"split: {nb} blocks x {block}B -> "
          f"train {len(train)} ({100*len(train)/len(data):.1f}%), "
          f"val {len(val)} ({100*len(val)/len(data):.1f}%)")

    DYN = bool(args.boundaries)
    ENDO = bool(args.endogenous)      # flips DYN True in-place once the boundary head freezes --
                                      # see the training loop; from then on this run IS a DYN run.
    train_m = val_m = None
    if DYN:
        bnd = load_boundary_mask(args.boundaries, data)      # verifies byte-alignment
        # split bytes AND boundaries in lockstep -- MUST match your byte-split params
        train, train_m, val, val_m = block_split_with_mask(data, bnd, block=block,
                                                            val_every=args.val_every)
        print(f"dynamic patching: {args.boundaries} "
              f"(corpus mean patch len ~{bnd.numel()/int(bnd.sum()):.2f})")
    elif ENDO:
        print(f"endogenous patching: boundary head freezes at step {args.boundary_freeze_step}, "
              f"target len {args.boundary_target_len}")

    print(f"device={device}  bytes={n_bytes}  bytes/char={bytes_per_char:.3f}  "
          f"seq_len={S} ({args.patches} patches x {args.patch_len})")
    cfg = HierConfig(d_model=args.d_model, patch_len=args.patch_len, depth=args.depth,
                     dropout=args.dropout, window_size=max(16, args.patches),
                     byte_ctx_len=args.byte_ctx_len)
    model = HierByteLM(cfg).to(device)
    print(f"params={model.num_params()/1e6:.2f}M  (global runs over {args.patches} patches, "
          f"~{args.patch_len}x shorter than {S} bytes)\n")

    flat = None
    if args.entropy_ckpt:
        flat = load_flat_model(args.entropy_ckpt, device, lambda c: Griffin(c))
        print(f"loaded frozen flat model {args.entropy_ckpt} for first/within tax comparison "
              f"({'real dynamic mask' if DYN else f'synthetic stride-{args.patch_len} mask'})\n")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.steps - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    def get_batch(split, split_m=None):
        ix = torch.randint(len(split) - S - 1, (args.batch_size,))
        x = torch.stack([split[i:i + S] for i in ix]).to(device)
        if split_m is None:
            return x
        m = torch.stack([split_m[i:i + S] for i in ix]).to(device)
        return x, m

    boundary_head = bh_opt = None
    threshold = None
    frozen = not ENDO       # True whenever there's no boundary head left to freeze
    if ENDO:
        bh_cfg = GriffinConfig(vocab_size=256, d_model=args.boundary_d_model,
                               d_rnn=int(1.5 * args.boundary_d_model), depth=args.boundary_depth,
                               head_dim=args.boundary_head_dim, window_size=args.boundary_window,
                               parallel_scan=True, dropout=0.0)
        boundary_head = Griffin(bh_cfg).to(device)
        bh_opt = torch.optim.AdamW(boundary_head.parameters(), lr=args.boundary_lr,
                                   betas=(0.9, 0.95), weight_decay=0.1)
        print(f"boundary head: {boundary_head.num_params()/1e6:.2f}M params")
        calib_batches = [get_batch(train) for _ in range(args.boundary_calib_batches)]
        threshold = recalibrate_threshold(boundary_head, calib_batches, args.boundary_target_len)
        print(f"initial threshold: {threshold:.3f} bits (target len {args.boundary_target_len})\n")

    def endo_mask(x):
        return batched_entropy_mask(boundary_head, x, threshold, Lcap=args.Lcap)

    @torch.no_grad()
    def eval_loss():
        model.eval()
        tot = first = within = mlen = 0.0
        ff_bits = ff_n = fw_bits = fw_n = 0.0    # flat model's first/within, bits (same boundaries)
        for _ in range(args.eval_batches):
            if DYN:
                x, m = get_batch(val, val_m)
                p_, pl_, pm_, bm_ = build_ragged(x, m, Lcap=args.Lcap)
                l, aux = forward_ragged(model, x, p_, pl_, pm_, bm_)
                first += aux["loss_first"].item(); within += aux["loss_within"].item()
                mlen += aux["mean_patch_len"].item()
                if flat is not None:
                    # exactly the mask build_ragged used internally (forced start + Lcap
                    # splits) -- the real boundaries hier's own first/within were scored on.
                    adj_m = m.clone(); adj_m[:, 0] = True
                    adj_m = cap_patch_lengths(adj_m, Lcap=args.Lcap)
                    r = flat_first_within(flat, x, adj_m)
            elif ENDO and not frozen:
                x = get_batch(val)
                m = endo_mask(x)
                p_, pl_, pm_, bm_ = build_ragged(x, m, Lcap=args.Lcap)
                l, aux = forward_ragged(model, x, p_, pl_, pm_, bm_)
                first += aux["loss_first"].item(); within += aux["loss_within"].item()
                mlen += aux["mean_patch_len"].item()
                if flat is not None:
                    r = flat_first_within(flat, x, m)
            else:
                x = get_batch(val)
                logits, l = model(x)
                L = args.patch_len
                B_, P_, Lp_, V_ = logits.shape
                tgt = x.view(B_, P_, L)
                ce = F.cross_entropy(logits.reshape(-1, V_), tgt.reshape(-1),
                                     reduction="none").view(B_, P_, Lp_)
                first += ce[:, :, 0].mean().item()
                within += ce[:, :, 1:].mean().item()
                if flat is not None:
                    first_tgt = torch.zeros_like(x, dtype=torch.bool)
                    first_tgt[:, ::L] = True         # synthetic fixed-stride mask (step 2)
                    r = flat_first_within(flat, x, first_tgt)
            if flat is not None:
                ff_bits += r["first_bits"]; ff_n += r["first_n"]
                fw_bits += r["within_bits"]; fw_n += r["within_n"]
            tot += l.item()
        model.train()
        n = args.eval_batches
        out = {"val": tot/n, "hier_first": first/n, "hier_within": within/n,
               "mean_patch_len": mlen/n}
        if flat is not None:
            out["flat_first"] = ff_bits / max(ff_n, 1)
            out["flat_within"] = fw_bits / max(fw_n, 1)
        return out

    best_val = float("inf")
    start_step = 0
    skipped = 0

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            print("note: checkpoint has no optimizer state; optimizer restarts fresh.")
        start_step = ck.get("step", -1) + 1
        best_val = ck.get("best_val", best_val)
        print(f"resumed from {args.resume} at step {start_step}")
        if ENDO and "boundary_head" in ck:
            boundary_head.load_state_dict(ck["boundary_head"])
            bh_opt.load_state_dict(ck["bh_opt"])
            threshold = ck["threshold"]
            frozen = ck["frozen"]
            if frozen:
                # re-derive the exact post-freeze masked split from the .npz already written
                # at freeze time, rather than rerunning the whole-corpus scan
                bnd = load_boundary_mask(f"boundaries_endo{args.ckpt_tag}.npz", data)
                train, train_m, val, val_m = block_split_with_mask(data, bnd, block=block,
                                                                    val_every=args.val_every)
                DYN = True
                print(f"resumed post-freeze (threshold {threshold:.3f}); "
                      f"reloaded boundaries_endo{args.ckpt_tag}.npz")
            else:
                print(f"resumed pre-freeze boundary head (threshold {threshold:.3f})")

    def save_ckpt(path, step):
        ck = {"model": model.state_dict(), "opt": opt.state_dict(),
              "cfg": cfg, "step": step, "best_val": best_val,
              "bytes_per_char": bytes_per_char}
        if ENDO:
            ck["boundary_head"] = boundary_head.state_dict()
            ck["boundary_cfg"] = boundary_head.cfg
            ck["bh_opt"] = bh_opt.state_dict()
            ck["threshold"] = threshold
            ck["frozen"] = frozen
        torch.save(ck, path)

    ln2 = math.log(2)
    model.train()
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if ENDO and not frozen and step == args.boundary_freeze_step:
            # Freeze: stop training the boundary head, do ONE real whole-corpus scan (reusing
            # precompute_boundaries.py's own compute_surprise/solve_threshold), then collapse
            # into the exact static-mask path --boundaries already uses for the rest of the run.
            for p_ in boundary_head.parameters():
                p_.requires_grad_(False)
            boundary_head.eval()
            bnd, bmeta = build_boundaries(boundary_head, data, args.boundary_target_len,
                                          block=args.boundary_scan_block, ctx=args.boundary_scan_ctx,
                                          batch=args.boundary_scan_batch, device=device)
            bnd_path = f"boundaries_endo{args.ckpt_tag}.npz"
            np.savez_compressed(bnd_path, mask=bnd.numpy().astype(np.bool_),
                                meta=np.array(bmeta, dtype=object))
            train, train_m, val, val_m = block_split_with_mask(data, bnd, block=block,
                                                                val_every=args.val_every)
            DYN = True
            frozen = True
            print(f"\nstep {step}: froze boundary head -> {bnd_path} "
                  f"(mean patch len {bmeta['mean_patch_len']:.2f}, "
                  f"threshold {bmeta['threshold']:.3f})\n")

        if DYN:
            x, m = get_batch(train, train_m)
            p_, pl_, pm_, bm_ = build_ragged(x, m, Lcap=args.Lcap)
            loss, _ = forward_ragged(model, x, p_, pl_, pm_, bm_)
        elif ENDO:      # not frozen yet -- boundary head still training, live per-step mask
            x = get_batch(train)
            logits_bh = boundary_head(x)
            V_bh = boundary_head.cfg.vocab_size
            bh_loss = F.cross_entropy(logits_bh[:, :-1].reshape(-1, V_bh), x[:, 1:].reshape(-1))
            bh_opt.zero_grad(set_to_none=True)
            bh_loss.backward()
            torch.nn.utils.clip_grad_norm_(boundary_head.parameters(), 1.0)
            bh_opt.step()

            if step % args.boundary_recalib_interval == 0:
                calib_batches = [get_batch(train) for _ in range(args.boundary_calib_batches)]
                threshold = recalibrate_threshold(boundary_head, calib_batches,
                                                  args.boundary_target_len)

            with torch.no_grad():
                logp = F.log_softmax(logits_bh.detach(), dim=-1)
                ent = -(logp.exp() * logp).sum(-1) / ln2
                m = torch.zeros_like(x, dtype=torch.bool)
                m[:, 0] = True
                if x.shape[1] > 1:
                    m[:, 1:] = ent[:, :-1] > threshold

            p_, pl_, pm_, bm_ = build_ragged(x, m, Lcap=args.Lcap)
            loss, _ = forward_ragged(model, x, p_, pl_, pm_, bm_)
        else:
            _, loss = model(get_batch(train))

        # NaN/Inf guard: skip the poisoning step instead of dying.
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped <= 5 or skipped % 20 == 0:
                print(f"step {step:4d} | non-finite loss -> step skipped (total {skipped})")
            if skipped > args.max_skips:
                print(f"aborting: {skipped} skips exceeds --max_skips. "
                      f"Lower --lr or raise --warmup.")
                break
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gnorm):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped <= 5 or skipped % 20 == 0:
                print(f"step {step:4d} | non-finite grad -> step skipped (total {skipped})")
            if skipped > args.max_skips:
                print(f"aborting: {skipped} skips exceeds --max_skips.")
                break
            continue
        opt.step()

        if step % args.eval_interval == 0 or step == args.steps - 1:
            m = eval_loss()
            v = m["val"]
            bpb = v / ln2; bpc = bpb * bytes_per_char
            flag = ""
            if v < best_val:
                best_val = v; save_ckpt(f"best{args.ckpt_tag}.pt", step); flag = "  <- best (saved)"
            vf, vw = m["hier_first"] / ln2, m["hier_within"] / ln2
            extra = f" | hier first {vf:.3f} within {vw:.3f} b/byte"
            if DYN or (ENDO and not frozen):
                extra += f" | len {m['mean_patch_len']:.2f}"
                if ENDO and not frozen:
                    extra += f" | thr {threshold:.2f}"
            else:
                extra += f" | len {args.patch_len} (fixed)"
            if "flat_first" in m:
                ff, fw = m["flat_first"], m["flat_within"]
                extra += f" | flat first {ff:.3f} within {fw:.3f}"
                extra += f" | tax first {vf - ff:+.3f} within {vw - fw:+.3f}"
            print(f"step {step:4d} | lr {lr_at(step):.2e} | train {loss.item():.3f} "
                  f"| val {v:.3f} | bits/byte {bpb:.3f} | BPC {bpc:.3f}{flag}{extra}")

        if args.ckpt_interval and step > 0 and step % args.ckpt_interval == 0:
            save_ckpt(f"last{args.ckpt_tag}.pt", step)

    if DYN or ENDO:
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            print("\n(skipping sample: non-finite weights. Resume from best.pt / last.pt "
                  "at a lower --lr.)")
            return
        if ENDO:
            # boundary_head/threshold exist from the initial calibration onward, frozen or not
            state = "frozen" if frozen else "still training, pre-freeze"
            entropy_model, thr, src = boundary_head, threshold, f"endogenous boundary head, {state}"
        elif flat is not None:
            entropy_model, thr, src = flat, load_threshold(args.boundaries), args.entropy_ckpt
        else:
            print("\n(dynamic mode: pass --entropy_ckpt to also enable autonomous generation "
                  "-- it reuses that frozen model + boundaries.npz's threshold online.)")
            return
        print(f"\n--- sample (autonomous, dynamic patches, threshold {thr:.2f} bits, {src}, "
              f"temperature {args.temperature} top_k {args.top_k} top_p {args.top_p}) ---")
        print(repr(model.generate_dynamic(240, device, entropy_model, thr, prompt=b"The ",
                                          Lcap=args.Lcap, temperature=args.temperature,
                                          top_k=args.top_k, top_p=args.top_p)))
        return

    # ------- sample (skip if the run diverged) -------
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        print("\n(skipping sample: non-finite weights. Resume from best.pt / last.pt "
              "at a lower --lr.)")
        return
    print(f"\n--- sample (autonomous, fixed-length patches, temperature {args.temperature} "
          f"top_k {args.top_k} top_p {args.top_p}) ---")
    print(repr(model.generate(40, device, prompt=b"The ",
                              temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)))


if __name__ == "__main__":
    main()
