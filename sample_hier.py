"""
sample_hier.py -- generate from a trained hierarchical (patch-based) checkpoint
with real sampling (temperature / top-k / top-p), no retraining needed.

Auto-detects which patching regime the checkpoint belongs to:
  - Stage 2 (endogenous): the checkpoint carries its own boundary_head/
    threshold (saved by hierarchical.py's --endogenous training path), so
    nothing else is needed.
  - Stage 1 (external dynamic): pass --entropy_ckpt and --boundaries so the
    frozen flat model + its calibrated threshold can be reloaded.
  - Fixed: neither of the above -- uniform stride-patch_len generation.

Usage:
    python sample_hier.py --ckpt best.pt --prompt "The " --n 400 --temperature 0.8 --top_k 40
    python sample_hier.py --ckpt best_dyn.pt --entropy_ckpt entropy_model/best.pt \
        --boundaries boundaries.npz --n 400 --temperature 0.8 --top_p 0.9
    python sample_hier.py --ckpt best_endo.pt --n 400 --temperature 0.8 --top_k 40
    python sample_hier.py --ckpt best_endo.pt --n 400 --temperature 0    # greedy, for comparison
"""

import argparse

import torch

from griffin_cglru import Griffin, GriffinConfig  # noqa: F401 (GriffinConfig needed for unpickle)
from hierarchical import HierByteLM, HierConfig    # noqa: F401 (HierConfig needed for unpickle)
from eval_hook import load_flat_model
from dynamic import load_threshold


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="The ")
    p.add_argument("--n", type=int, default=400, help="bytes to generate beyond the prompt")
    p.add_argument("--temperature", type=float, default=0.8,
                   help="0 = greedy (argmax); >0 samples from the softmax at this temperature")
    p.add_argument("--top_k", type=int, default=0, help="0 = disabled; else keep top-k logits")
    p.add_argument("--top_p", type=float, default=0.0,
                   help="0 = disabled; else nucleus-sample the smallest top set with "
                        "cumulative probability >= top_p")
    p.add_argument("--entropy_ckpt", default="",
                   help="frozen flat model, required for a Stage-1 (external) dynamic "
                        "checkpoint; not needed for fixed or Stage-2 (endogenous) checkpoints")
    p.add_argument("--boundaries", default="",
                   help="boundaries.npz, for a Stage-1 checkpoint's calibrated threshold "
                        "(Stage-2 checkpoints carry their own, this is ignored for those)")
    p.add_argument("--Lcap", type=int, default=32, help="max patch length (dynamic/endogenous only)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    model = HierByteLM(ckpt["cfg"]).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prompt = args.prompt.encode("utf-8")
    sample_kwargs = dict(temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)

    if "boundary_head" in ckpt:
        mode = "Stage 2 (endogenous)"
        boundary_head = Griffin(ckpt["boundary_cfg"]).to(args.device)
        boundary_head.load_state_dict(ckpt["boundary_head"])
        boundary_head.eval()
        threshold = ckpt["threshold"]
        if not ckpt.get("frozen", True):
            print("note: checkpoint's boundary head was saved pre-freeze -- "
                  "generation uses the not-yet-final threshold.")
        out = model.generate_dynamic(len(prompt) + args.n, args.device, boundary_head, threshold,
                                     prompt=prompt, Lcap=args.Lcap, **sample_kwargs)
    elif args.entropy_ckpt:
        mode = "Stage 1 (external dynamic)"
        assert args.boundaries, "--boundaries is required alongside --entropy_ckpt"
        flat = load_flat_model(args.entropy_ckpt, args.device, lambda c: Griffin(c))
        threshold = load_threshold(args.boundaries)
        out = model.generate_dynamic(len(prompt) + args.n, args.device, flat, threshold,
                                     prompt=prompt, Lcap=args.Lcap, **sample_kwargs)
    else:
        mode = "fixed"
        L = model.cfg.patch_len
        n_patches = -(-args.n // L)   # ceil
        out = model.generate(n_patches, args.device, prompt=prompt, **sample_kwargs)

    print(f"--- {mode} | temperature {args.temperature} top_k {args.top_k} top_p {args.top_p} ---")
    print(out.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
