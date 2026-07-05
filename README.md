# GRIFFIN — Hierarchical Byte-Level Language Modeling with Dynamic Patching

A from-scratch PyTorch implementation of Google DeepMind's
[Griffin](https://arxiv.org/abs/2402.19427) (gated linear recurrence +
local attention) extended into a **two-level byte-level language model**:
a Griffin backbone runs causally over compressed *patch* vectors instead of
raw bytes, with an encoder/decoder wrapping it to compress bytes into
patches and decode them back out. The empirical question this repo answers:
does cutting the byte stream at the model's own points of surprise (dynamic
patching) beat uniform fixed-length patches — and does it matter whether
those cut points come from a frozen external model or one trained jointly
with the rest of the system?

**Headline result:** yes — a small boundary-placement head trained *jointly*
with the hierarchical model ("endogenous" patching) beats both a
fixed-stride baseline and dynamic patching driven by an external frozen
entropy model, on bits-per-character, confirmed across two random seeds.
See [`RESULTS_ANNEX.md`](RESULTS_ANNEX.md) for the full experimental
narrative and [`paper/main.pdf`](paper/main.pdf) for a short write-up with
an architecture diagram.

## How the pieces fit together

```
byte stream
     │
     ▼
boundary policy          -- frozen external entropy model, OR a small
(patch segmentation)        BoundaryHead trained jointly, then frozen
     │
     ▼
PatchEncoder              -- bytes of one patch -> one vector
     │
     ▼
Griffin backbone           -- THE SAME architecture as the paper's local-
(unchanged scaffolding,       attention + RG-LRU/CG-LRU residual blocks,
 CG-LRU recurrence)            just applied one level up, over patch vectors
     │
     ▼
PatchDecoder                -- previous patch's context vector + trailing
(+ cross-patch context)        raw bytes of prior patch(es) -> byte logits
     │
     ▼
byte logits
```

Griffin's residual/local-attention block (`griffin_cglru.py`) is reused
*unchanged* in two different roles: as the boundary-placing model, and as
the global model running over patch vectors. Everything else — the patch
encoder/decoder, the two boundary policies, and the cross-patch decoder
context — is new. See `paper/main.pdf` for the full diagram and a
correction worth knowing up front: this implementation uses the *complex*
CG-LRU recurrence described in the paper's Appendix B, not the real-valued
RG-LRU the paper's own reported experiments actually used.

## Repo layout

| File | Role |
|---|---|
| `griffin_cglru.py` | The Griffin block (CG-LRU recurrence + blockwise local-window MQA). Used both as the flat byte model and as the hierarchical model's global model. |
| `tokenizer.py` | Char / byte tokenizers. |
| `train_verdict.py` | Trains a **flat** Griffin — the char-level anchor, and the byte-level entropy model used to place boundaries. |
| `hierarchical.py` | Trains the **two-level** model: fixed-stride patches, external dynamic patches (`--boundaries`), or endogenous patches (`--endogenous`). |
| `analyze.py` | Next-byte entropy → boundary placement, sweep, heatmap. The boundary-quality go/no-go gate. |
| `precompute_boundaries.py` | Offline whole-corpus boundary mask + threshold from a frozen entropy model, for the external-dynamic regime. |
| `dynamic.py` | Ragged-batching support for variable-length patches, and the online (generation-time) boundary-decision helpers. |
| `boundary_head.py` | The endogenous boundary-placement machinery: live per-step masking, threshold recalibration, freeze-time whole-corpus rescan. |
| `patcher.py` | `PatchEncoder` / `PatchDecoder`, including the cross-patch byte-context mechanism and temperature/top-k/top-p sampling. |
| `eval_hook.py` | Scores a frozen flat model's first/within NLL on the *same* batches the hierarchical model evaluates — the architectural-tax measurement. |
| `autoencode.py` | Patch-codec sanity check (can `L` bytes be recovered from one vector?). |
| `sample.py` | Sampling from a flat (`train_verdict.py`) checkpoint. |
| `sample_hier.py` | Sampling from a hierarchical checkpoint — auto-detects fixed / external-dynamic / endogenous. |
| `REPRODUCE.md` | The detailed, step-by-step experimental protocol (read this for the full "why"). |
| `RESULTS_ANNEX.md` | The complete results narrative across every round of the study. |
| `paper/` | A short LaTeX write-up citing the source paper directly, with an architecture diagram. |

## Requirements

- Python 3.10+
- PyTorch (tested on 2.x; a CUDA build if you have a GPU — this also runs on CPU for small smoke tests)
- NumPy

```bash
pip install torch numpy
```

No other dependencies. Built and validated on modest hardware (8 GB GPUs);
`griffin_cglru.py`'s local attention is a blockwise O(sequence-length ×
window) implementation, not the naive O(sequence-length²) — it doesn't
need much VRAM even at long context.

## Quickstart

Put some UTF-8 `.txt` files (e.g. a handful of Project Gutenberg books) in
a `corpus/` directory, then:

### 1. Train the byte-level entropy model

This is the model that will place patch boundaries. Train it in its own
directory so it never collides with the hierarchical model's checkpoints,
and stop it early — a fully-converged model smooths away the sharp
surprise signal a boundary policy needs.

```bash
mkdir -p entropy_model
cd entropy_model
python ../train_verdict.py --data ../corpus --tokenizer byte --size small --max_iters 2000
cd ..
```

### 2. Sanity-check its boundaries (go/no-go gate)

```bash
python analyze.py --ckpt entropy_model/best.pt --file corpus/<one_book>.txt \
    --sweep --color --percentile 85 --top 12
```

Boundaries should land on word/sentence onsets, and the highest-surprise
positions should be word-initial bytes. If they don't, check these in order
— the first is by far the most common cause:

1. **Wrong operating point, not a bad model.** Mean patch length is a pure
   function of `--percentile`: `len = 100 / (100 - percentile)`. To match a
   fixed baseline at `--patch_len 6` you need `len ≈ 6`, i.e. `--percentile`
   around **84–85**, not the `analyze.py` default of 70 (`len ≈ 3.3`, which
   will look badly over-fragmented even from a perfectly good entropy
   model). Try the matching percentile before touching the model at all.
2. **Boundaries look smooth / cut in arbitrary places → likely overtrained.**
   Lower `--max_iters` and retrain. This isn't a guess: a controlled sweep
   in this repo's own history (`RESULTS_ANNEX.md`) found mid-word
   fragmentation rising monotonically from 10.6% at 200 steps to 15.6% at
   8000 steps on the same corpus/preset — quality peaked early, then
   degraded steadily with more training. If your run used more than a
   couple thousand steps, that's the first thing to cut back. That sweep
   used `train_verdict.py`'s own defaults (`--batch_size 32 --block_size
   128`); if you changed either, translate by *bytes seen*, not raw steps —
   see "How many steps do I need?" below.
3. **Boundaries look noisy / don't track words at all → likely
   undertrained or too small.** Raise `--max_iters` a bit, or move up a
   size preset (`--size tiny` → `small` → `medium`) — capacity has to be
   sufficient before "stop early" becomes the right lever; `entropy_model`
   in this repo's own experiments used `--size small` (9 layers, 384-dim),
   not `tiny`.

Re-run step 2 after any change — it's cheap and it's the actual gate, not a
one-time formality.

#### How many steps do I need?

`--steps`/`--max_iters` alone doesn't tell you how much data the model has
seen — that also depends on batch size and sequence length. The quantity
that's actually comparable across different settings is **bytes of
gradient exposure**:

```
bytes_per_step = batch_size × sequence_length
total_bytes    = steps × bytes_per_step
```

where `sequence_length` is `block_size` for `train_verdict.py`, or
`patch_len × patches` for `hierarchical.py`. Training samples random
windows *with replacement*, not epoch by epoch, so `total_bytes ÷
corpus_size` is an **equivalent epoch count**, not an exact count of unique
bytes seen — but it's the right number to compare across runs with
different batch sizes or sequence lengths.

Worked example — this repo's own `hierarchical.py` defaults
(`batch_size=16`, `patch_len=6`, `patches=32` → 3072 bytes/step) at
`--steps 32000`:

```
32000 × 3072 = 98,304,000 bytes of exposure
```

Against the ~39.6M-byte corpus used throughout this project's own
experiments, that's **~2.5 equivalent epochs** — the actual scale every
result in `RESULTS_ANNEX.md` was measured at. If your corpus is a
different size, scale `--steps` so `total_bytes` lands wherever you want
relative to your own corpus size, e.g. for ~3 epochs:
`steps = 3 × corpus_size_bytes ÷ (batch_size × patch_len × patches)`.

The same formula is what translates the entropy-model sweep above between
different `--batch_size`/`--block_size` choices: that sweep's "~1000 steps"
sweet spot is really **4,096,000 bytes of exposure** (`1000 × 32 × 128`);
if you use different batch/block sizes, divide 4,096,000 by
`your_batch_size × your_block_size` to get the equivalent step count for
your settings, rather than reusing "1000" literally.

### 3. Train the fixed-stride baseline

```bash
python hierarchical.py --file corpus --patch_len 6 --patches 32 \
    --d_model 256 --depth 6 --steps 32000 --lr 5e-4 --warmup 500 \
    --eval_interval 200 --eval_batches 100 \
    --entropy_ckpt entropy_model/best.pt
```

### 4. Train the external-dynamic regime (Stage 1)

Precompute boundaries from the frozen entropy model, then train against
that fixed mask:

```bash
python precompute_boundaries.py --corpus corpus --ckpt entropy_model/best.pt \
    --target_len 6 --out boundaries.npz --block 1024 --batch 8

python hierarchical.py --file corpus --patch_len 6 --patches 32 \
    --d_model 256 --depth 6 --steps 32000 --lr 5e-4 --warmup 500 \
    --eval_interval 200 --eval_batches 100 \
    --boundaries boundaries.npz --entropy_ckpt entropy_model/best.pt --ckpt_tag _dyn
```

### 5. Train the endogenous regime (Stage 2)

No precompute step — a small boundary head trains jointly, then freezes
partway through and rescans the corpus once:

```bash
python hierarchical.py --file corpus --patch_len 6 --patches 32 \
    --d_model 256 --depth 6 --steps 32000 --lr 5e-4 --warmup 500 \
    --eval_interval 200 --eval_batches 100 \
    --endogenous --boundary_freeze_step 1300 \
    --entropy_ckpt entropy_model/best.pt --ckpt_tag _endo
```

### 6. Generate text from any of the three checkpoints

`sample_hier.py` auto-detects which regime a checkpoint belongs to:

```bash
python sample_hier.py --ckpt best.pt        --prompt "The " --n 400 --temperature 0.8 --top_k 40
python sample_hier.py --ckpt best_dyn.pt    --entropy_ckpt entropy_model/best.pt --boundaries boundaries.npz \
                       --prompt "The " --n 400 --temperature 0.8 --top_p 0.9
python sample_hier.py --ckpt best_endo.pt   --prompt "The " --n 400 --temperature 0.8 --top_k 40
```

Use `--temperature 0` for the old deterministic greedy behavior; greedy
decoding tends to fall into repetition loops on a model this size, so
sampling is recommended for anything meant to be read.

## Scaling up

Everything above runs at a deliberately small scale (~7M-parameter
hierarchical model, `patches=32` × `patch_len=6` = 192 bytes of context per
training window). If you want a bigger model or more context, here's
exactly what's exposed and what isn't.

**Global (hierarchical) model width/depth** — `--d_model` and `--depth` are
both CLI flags. One constraint: `d_model` must be a multiple of `head_dim`
(64 by default), so valid values are 128, 192, 256, 320, 384, ... A gap
worth knowing: the global model's recurrent width (`d_rnn`, default 384) is
**not** wired to a CLI flag and won't auto-scale with `--d_model` — the
paper's own models keep `d_rnn ≈ 4/3 × d_model`, so if you push `d_model`
well past 256 without also raising `d_rnn`, you're skewing that ratio. To
change it, edit `HierConfig`'s `d_rnn` default directly in
`hierarchical.py`.

**More context** — raise `--patches` and/or `--patch_len` (the training
window is `patches × patch_len` bytes). This used to hit an O(sequence²)
memory wall in `LocalMQA`'s attention; that's fixed now (blockwise,
O(sequence × window)), so pushing `--patches` well past 32 is far cheaper
than it used to be. The window itself auto-tracks `--patches`
(`window_size = max(16, patches)`), so you don't need to set it separately.

**Boundary head capacity (Stage 2 only)** — `--boundary_d_model`,
`--boundary_depth`, `--boundary_head_dim`, `--boundary_window` are all CLI
flags, and this one's recurrent width *does* auto-scale (`d_rnn = 1.5 ×
boundary_d_model`), so no manual edit needed there. Keep it modest relative
to the main model, though — it runs a full forward+backward pass every
single pre-freeze training step, on top of the main model's own step.

**Entropy model / char anchor capacity** — controlled by `--size` in
`train_verdict.py` (`tiny` / `small` / `medium`; see `PRESETS` in that file
for the exact numbers), not raw `d_model`/`depth` flags.

**Batch size and hardware** — `--batch_size` is the usual lever if you're
VRAM-bound rather than compute-bound. Two practical notes from this repo's
own history: this codebase has **no multi-GPU support** (no DDP, no model
parallelism) — two GPUs today means two independent runs side by side, not
double the VRAM for one run. And the lowest-effort path to a genuinely
bigger model is a single bigger card (e.g. Colab) rather than squeezing
scale out of 8 GB: nothing here assumes a specific device, so the same
commands run unchanged, just with larger `--d_model`/`--depth`/`--patches`.

## Where to go next

- **`REPRODUCE.md`** — the full protocol: environment notes, the
  block-interleaved train/val split and why it matters, numerical-stability
  gotchas, and every decision gate in detail.
- **`RESULTS_ANNEX.md`** — the complete results narrative, round by round,
  including the two-seed confirmation of the headline finding and every
  dead end investigated along the way.
- **`paper/main.pdf`** — a short write-up citing
  [the source paper](https://arxiv.org/abs/2402.19427) directly, with a
  diagram of exactly where this project's additions sit relative to the
  original architecture.

## Citation

This project builds on:

> De, Smith, Fernando, Botev, Cristian-Muraru, Gu, Haroun, Berrada, Chen,
> Srinivasan, Desjardins, Doucet, Budden, Teh, Pascanu, De Freitas, Gulcehre.
> "Griffin: Mixing Gated Linear Recurrences with Local Attention for
> Efficient Language Models." arXiv:2402.19427 (2024). Google DeepMind.
