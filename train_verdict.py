"""
Train the Griffin / CG-LRU model on 'the-verdict.txt' (character level).
========================================================================

A minimal, dependency-free training loop meant to run on modest hardware
(e.g. a single GTX 1070). No tokenizer download: we build a char vocab from
the file itself. The corpus is tiny (~20 KB), so the model *will* overfit --
that is the intended demo. You should see train loss fall well below val loss
and the samples start to read like Wharton.

Usage:
    python train_verdict.py                 # trains, prints loss, samples
    python train_verdict.py --device cpu    # force CPU

Pascal-era GPU notes:
    * fp32 only. fp16 runs ~1/64 speed on GP104 -- do NOT enable autocast.
    * One card is plenty at this size; no need for DataParallel.
    * parallel_scan=True (default) minimizes CUDA launch overhead in the scan.
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F

from griffin_cglru import Griffin, GriffinConfig
from tokenizer import build_tokenizer


# --------------------------------------------------------------------------- #
# Data: read a file or a folder of .txt, tokenize with the chosen tokenizer
# --------------------------------------------------------------------------- #
def load_data(path, kind):
    if os.path.isdir(path):
        import glob
        files = sorted(glob.glob(os.path.join(path, "*.txt")))
        if not files:
            raise FileNotFoundError(f"no .txt files found in directory {path}")
        text = "\n\n".join(open(f, encoding="utf-8").read() for f in files)
        print(f"loaded {len(files)} file(s) from {path}/")
    else:
        text = open(path, encoding="utf-8").read()

    tok = build_tokenizer(kind, text)
    if kind == "byte":
        # Fast, no giant Python list: view the raw UTF-8 buffer as bytes.
        data = torch.frombuffer(bytearray(text.encode("utf-8")), dtype=torch.uint8).long()
    else:
        data = torch.tensor(tok.encode(text), dtype=torch.long)

    # tokens-per-character makes loss comparable across tokenizers:
    #   char -> 1.0, byte -> bytes/char, bpe -> <1. BPC = nats/token * tpc / ln2.
    tokens_per_char = len(data) / max(1, len(text))
    return data, tok, tokens_per_char


def get_batch(data, block_size, batch_size, device):
    # Random windows; target is input shifted by one.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------- #
# Loss / eval
# --------------------------------------------------------------------------- #
def loss_fn(model, x, y):
    logits = model(x)                                  # [B,T,vocab]
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def estimate_loss(model, splits, block_size, batch_size, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, block_size, batch_size, device)
            losses[k] = loss_fn(model, x, y).item()
        out[name] = losses.mean().item()
    model.train()
    return out


# --------------------------------------------------------------------------- #
# Model size presets (all keep head_dim=128-friendly shapes; d_rnn even)
# 'tiny' is the original demo model; 'small'/'medium' suit a multi-MB corpus.
# --------------------------------------------------------------------------- #
PRESETS = {
    # name       d_model d_rnn depth head_dim window
    "tiny":   dict(d_model=256, d_rnn=384,  depth=6,  head_dim=64,  window_size=64),
    "small":  dict(d_model=384, d_rnn=512,  depth=9,  head_dim=64,  window_size=128),
    "medium": dict(d_model=512, d_rnn=768,  depth=12, head_dim=64,  window_size=256),
}


# --------------------------------------------------------------------------- #
# Main
# Sampling is inlined at the end of main(): no KV cache, we just recompute the
# (cropped) context each step. Slow in principle, fine for short demo samples.
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="the-verdict.txt")
    p.add_argument("--tokenizer", default="byte", choices=["byte", "char"])
    p.add_argument("--size", default="tiny", choices=list(PRESETS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--max_skips", type=int, default=50,
                   help="abort if this many non-finite steps are skipped (persistent NaN)")
    p.add_argument("--ckpt_interval", type=int, default=500,
                   help="save a resumable checkpoint (last.pt) every N steps")
    p.add_argument("--bpc_iters", type=int, default=200,
                   help="batches for the final bits-per-char estimate")
    p.add_argument("--resume", default="",
                   help="path to a last.pt checkpoint to resume from")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    print(f"device: {device}")

    if not os.path.exists(args.data):
        raise FileNotFoundError(f"{args.data} not found (run from the folder containing it)")

    data, tok, tokens_per_char = load_data(args.data, args.tokenizer)
    vocab_size = tok.vocab_size
    decode = tok.decode
    n = int(0.9 * len(data))
    splits = {"train": data[:n], "val": data[n:]}
    print(f"tokenizer={tok.kind}  tokens={len(data)}  vocab={vocab_size}  "
          f"tokens/char={tokens_per_char:.3f}  "
          f"train={len(splits['train'])}  val={len(splits['val'])}")

    # nats/token -> bits/character (tokenizer-independent comparison axis)
    to_bpc = lambda nats: nats * tokens_per_char / math.log(2)

    # Config from the chosen size preset.
    cfg = GriffinConfig(
        vocab_size=vocab_size,
        parallel_scan=True,
        dropout=args.dropout,
        **PRESETS[args.size],
    )
    model = Griffin(cfg).to(device)
    print(f"preset={args.size}  params={model.num_params()/1e6:.2f}M  heads={cfg.num_heads}  "
          f"d_model={cfg.d_model} d_rnn={cfg.d_rnn} depth={cfg.depth} window={cfg.window_size}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.95), weight_decay=args.weight_decay)

    def lr_at(step):
        # linear warmup then cosine decay to 10% of peak
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        prog = (step - args.warmup) / max(1, args.max_iters - args.warmup)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    # prompt encoder for sampling (uses the chosen tokenizer)
    prompt_ids = lambda s: tok.encode(s) or [0]

    best_val = float("inf")
    start_step = 0
    skipped = 0

    # Optional resume from a checkpoint. Tolerant of older/partial files:
    # loads weights always; restores optimizer state and step only if present.
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        else:
            print("note: checkpoint has no optimizer state (older file); "
                  "optimizer restarts fresh.")
        start_step = ck.get("step", -1) + 1
        best_val = ck.get("best_val", best_val)
        shown = f"{best_val:.3f}" if best_val < float("inf") else "n/a"
        print(f"resumed from {args.resume} at step {start_step} (best_val={shown})")

    def save_ckpt(path, step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "cfg": cfg, "tok": tok.state(), "step": step,
                    "best_val": best_val}, path)

    model.train()
    for step in range(start_step, args.max_iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        x, y = get_batch(splits["train"], args.block_size, args.batch_size, device)
        loss = loss_fn(model, x, y)

        # NaN/Inf guard: a single bad batch can produce a non-finite loss whose
        # gradients would poison every weight. Skip the step instead of dying.
        if not torch.isfinite(loss):
            skipped += 1
            opt.zero_grad(set_to_none=True)
            if skipped <= 5 or skipped % 20 == 0:
                print(f"step {step:4d} | non-finite loss -> step skipped "
                      f"(total skipped: {skipped})")
            if skipped > args.max_skips:
                print(f"aborting: {skipped} skipped steps exceeds --max_skips "
                      f"({args.max_skips}). Lower --lr or --warmup harder.")
                break
            continue

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % args.eval_interval == 0 or step == args.max_iters - 1:
            m = estimate_loss(model, splits, args.block_size, args.batch_size, device)
            flag = ""
            if m["val"] < best_val:
                best_val = m["val"]
                save_ckpt("best.pt", step)
                flag = "  <- best (saved)"
            print(f"step {step:4d} | lr {lr_at(step):.2e} | "
                  f"train {m['train']:.3f} | val {m['val']:.3f} | "
                  f"bpc {to_bpc(m['val']):.3f}{flag}")

        # Periodic resumable checkpoint (survives a later NaN / crash / Ctrl-C).
        if args.ckpt_interval and step > 0 and step % args.ckpt_interval == 0:
            save_ckpt("last.pt", step)

    # ------- final bits-per-char on the best checkpoint -------
    if os.path.exists("best.pt"):
        ck = torch.load("best.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        print(f"\nloaded best.pt (step {ck.get('step', '?')}) for final eval")
    mf = estimate_loss(model, splits, args.block_size, args.batch_size,
                       device, iters=args.bpc_iters)
    approx_tokens = args.bpc_iters * args.batch_size * args.block_size
    print(f"FINAL | val {mf['val']:.3f} nats/token | "
          f"BPC {to_bpc(mf['val']):.3f} bits/char  "
          f"(~{approx_tokens:,} tokens, tokenizer={tok.kind})")

    # ------- sample (from the best checkpoint just loaded) -------
    weights_ok = all(torch.isfinite(p).all() for p in model.parameters())
    if not weights_ok:
        print("\n(skipping end-of-run sample: weights are non-finite. "
              "Use best.pt / last.pt with sample.py, and rerun with a lower --lr.)")
        return

    print("\n--- sample ---")
    ids = torch.tensor([prompt_ids("I ")], dtype=torch.long, device=device)
    with torch.no_grad():
        model.eval()
        for _ in range(400):
            ctx = ids[:, -args.block_size:]
            logits = model(ctx)[:, -1, :] / 0.8
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            ids = torch.cat([ids, nxt], dim=1)
    print(decode(ids[0].tolist()))


if __name__ == "__main__":
    main()
