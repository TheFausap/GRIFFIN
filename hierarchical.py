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

import torch
import torch.nn as nn
import torch.nn.functional as F

from griffin_cglru import (Griffin, GriffinConfig, ResidualBlock, RecurrentBlock,
                           LocalMQA, RMSNorm)
from patcher import PatcherConfig, PatchEncoder, PatchDecoder, prev_patch_tail

from dynamic import (load_boundary_mask, block_split_with_mask,
                     build_ragged, forward_ragged, cap_patch_lengths)
from eval_hook import load_flat_model, flat_first_within

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

        plens = torch.full((B, P), L, dtype=torch.long, device=tokens.device)
        prev_ctx = prev_patch_tail(patches, plens, self.cfg.byte_ctx_len,
                                    self.decoder.BOS).reshape(B * P, self.cfg.byte_ctx_len)

        z = cond.reshape(B * P, self.cfg.d_model)
        tgt = patches.reshape(B * P, L)
        logits = self.decoder(z, tgt, prev_ctx).view(B, P, L, self.cfg.vocab_size)
        loss = F.cross_entropy(logits.reshape(-1, self.cfg.vocab_size), tokens.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, n_patches, device, prompt=b"", temperature=1.0):
        """Autonomous generation: emit n_patches new patches of L bytes each."""
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
                last_patch = out[-L:]                  # the just-completed patch
            else:
                cond = self.start[:, 0]                # [1, d]
                last_patch = []                         # no previous patch yet
            ctx = [pad_id] * max(0, K - len(last_patch)) + last_patch[-K:]
            prev_ctx = torch.tensor([ctx], dtype=torch.long, device=device)
            gen = self.decoder.generate(cond, torch.tensor([L], device=device), prev_ctx)  # [1,L]
            out.extend(int(b) for b in gen[0].tolist())
        return bytes(b & 0xFF for b in out)

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
    p.add_argument("--entropy_ckpt", default="",
                   help="frozen flat byte model (e.g. entropy_model/best.pt); if set, eval "
                        "also scores it first/within on the SAME boundaries as this run -- "
                        "the real dynamic mask when --boundaries is set (the tax gate), or "
                        "a synthetic stride-patch_len mask when running fixed (the fixed "
                        "first/within baseline) -- so both land in one eval.")
    args = p.parse_args()

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
    train_m = val_m = None
    if DYN:
        bnd = load_boundary_mask(args.boundaries, data)      # verifies byte-alignment
        # split bytes AND boundaries in lockstep -- MUST match your byte-split params
        train, train_m, val, val_m = block_split_with_mask(data, bnd, block=65536, val_every=10)
        print(f"dynamic patching: {args.boundaries} "
              f"(corpus mean patch len ~{bnd.numel()/int(bnd.sum()):.2f})")

    print(f"device={device}  bytes={n_bytes}  bytes/char={bytes_per_char:.3f}  "
          f"seq_len={S} ({args.patches} patches x {args.patch_len})")
    cfg = HierConfig(d_model=args.d_model, patch_len=args.patch_len, depth=args.depth,
                     dropout=args.dropout, window_size=max(16, args.patches))
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

    @torch.no_grad()
    def eval_loss():
        model.eval()
        tot = first = within = mlen = 0.0
        ff_bits = ff_n = fw_bits = fw_n = 0.0    # flat model's first/within, bits (same boundaries)
        for _ in range(20):                      # bump to ~50 for the real comparison
            if DYN:
                x, m = get_batch(val, val_m)
                p_, pl_, pm_, bm_ = build_ragged(x, m)
                l, aux = forward_ragged(model, p_, pl_, pm_, bm_)
                first += aux["loss_first"].item(); within += aux["loss_within"].item()
                mlen += aux["mean_patch_len"].item()
                if flat is not None:
                    # exactly the mask build_ragged used internally (forced start + Lcap
                    # splits) -- the real boundaries hier's own first/within were scored on.
                    adj_m = m.clone(); adj_m[:, 0] = True
                    adj_m = cap_patch_lengths(adj_m, Lcap=32)
                    r = flat_first_within(flat, x, adj_m)
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
        out = {"val": tot/20, "hier_first": first/20, "hier_within": within/20,
               "mean_patch_len": mlen/20}
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

    def save_ckpt(path, step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "cfg": cfg, "step": step, "best_val": best_val,
                    "bytes_per_char": bytes_per_char}, path)

    ln2 = math.log(2)
    model.train()
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        if DYN:
            x, m = get_batch(train, train_m)
            p_, pl_, pm_, bm_ = build_ragged(x, m)
            loss, _ = forward_ragged(model, p_, pl_, pm_, bm_)
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
            extra += f" | len {m['mean_patch_len']:.2f}" if DYN else f" | len {args.patch_len} (fixed)"
            if "flat_first" in m:
                ff, fw = m["flat_first"], m["flat_within"]
                extra += f" | flat first {ff:.3f} within {fw:.3f}"
                extra += f" | tax first {vf - ff:+.3f} within {vw - fw:+.3f}"
            print(f"step {step:4d} | lr {lr_at(step):.2e} | train {loss.item():.3f} "
                  f"| val {v:.3f} | bits/byte {bpb:.3f} | BPC {bpc:.3f}{flag}{extra}")

        if args.ckpt_interval and step > 0 and step % args.ckpt_interval == 0:
            save_ckpt(f"last{args.ckpt_tag}.pt", step)

    if DYN:
        print("\n(dynamic mode: autonomous generation is Stage 2 -- skipping sample.)")
        return

    # ------- sample (skip if the run diverged) -------
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        print("\n(skipping sample: non-finite weights. Resume from best.pt / last.pt "
              "at a lower --lr.)")
        return
    print("\n--- sample (autonomous, fixed-length patches) ---")
    print(repr(model.generate(40, device, prompt=b"The ")))


if __name__ == "__main__":
    main()
