# Reproducing the Hierarchical Byte-LM Entropy-Patching Experiment

A step-by-step protocol for reproducing the study of whether **surprise-based
dynamic patch boundaries** can match or beat **fixed-stride patching** in a
two-level Griffin / CG-LRU byte language model, while keeping the global
sequence compressed.

> **Central question.** Does CG-LRU recurrence make byte-level modeling
> competitive, and can boundaries cut at the model's own high-entropy positions
> close the quality gap versus uniform fixed-stride patches at matched mean
> patch length?

---

## 0. What you're comparing

Two ways of grouping a byte stream into patches, *same architecture otherwise*:

- **Fixed-patch** — uniform stride `L` (e.g. 6). Every patch is exactly `L`
  bytes. Rectangular tensors, trivial batching. This is the reproducible
  baseline.
- **Entropy (dynamic) patch** — variable-length patches cut wherever a frozen
  byte-level model's next-byte predictive entropy exceeds a threshold. Boundaries
  land on the highest-information bytes by construction.

The honest comparison metric is **total BPC** on held-out data (cut-invariant).
But BPC alone hides *why* one wins, because entropy patching redistributes
prediction difficulty onto boundary bytes. So every run is decomposed into
**first-byte-of-patch** vs **within-patch** loss. This decomposition is
mandatory, not optional.

---

## 1. Environment

- **GPUs:** built and validated on 2× GTX 1070 (Pascal, 8 GB).
- **Precision:** **fp32 only.** fp16 is slower than fp32 on Pascal — do not use
  autocast/AMP here.
- **Parallel scan:** the CG-LRU associative scan (`parallel_scan=True`) is
  GPU-only; on CPU use the sequential path.
- **Config constraint:** `d_model` must be divisible by `head_dim` (default 64).
  Valid `d_model`: 128, 192, 256, 320, 384, …
- **Deps:** Python 3.10+, PyTorch (CUDA build matching your driver), NumPy.

```
pip install torch numpy
```

---

## 2. Files

| File | Role |
|------|------|
| `griffin_cglru.py` | Griffin block: CG-LRU recurrent layer + local MQA. `Griffin(cfg)` returns logits `[B,T,V]` (tied embeddings). Used at both levels. |
| `tokenizer.py` | Char / byte tokenizers; `tokens_per_char` for cross-tokenizer BPC. |
| `train_verdict.py` | Trains a **flat** Griffin (char baseline **and** the byte entropy model). |
| `hierarchical.py` | Trains the **two-level** model (fixed-patch baseline; dynamic is the extension). |
| `analyze.py` | Next-byte entropy → boundary placement, sweep, heatmap, `--dump` of masks. The go/no-go gate. |
| `patcher.py` | `PatchEncoder` / `PatchDecoder`; consumes `boundaries.json`. Policy-agnostic (treats boundaries as given). |
| `autoencode.py` | Patch autoencoder sanity check (can `L` bytes be recovered from one vector?). |
| `sample.py` | Autonomous generation from a trained model. |
| `eval_hook.py` | Scores the frozen flat model's first/within NLL on the **same** batches the hierarchical model evaluates — the like-for-like tax measurement. |

---

## 3. Metrics & formulas

- **BPC** (bits per character), the cross-tokenizer axis:
  `BPC = nats_per_token × tokens_per_char / ln(2)`.
  For a byte model, `tokens_per_char = bytes/char`.
- **bits/byte** = `val_loss_nats / ln(2)`.
- **first-byte loss** — mean NLL (bits) of the first byte of each patch,
  predicted from prior-patch context only.
- **within-patch loss** — mean NLL (bits) of the remaining bytes.
- **architectural tax** — `hier_loss − flat_full_context_loss` on the *same*
  positions. The flat model has full byte-level left context, so its loss is the
  reference floor; the gap is what the patch bottleneck costs.

> Aggregate BPC alone cannot tell you whether entropy patching helped. Always
> report the first/within split alongside it.

---

## 4. Corpus & split

1. Put Gutenberg `.txt` files under `corpus/` (the study used ~39 M bytes across
   several books). A single file also works.
2. **Split must be block-interleaved, not contiguous.** Divide the byte stream
   into fixed blocks of size ≥ the training sequence length, route every Nth
   block to validation, no RNG. A contiguous tail split makes train and val
   sample *different* books and inflates the apparent train/val gap by ~0.4 —
   an artifact, not overfitting.
   - ⚠️ The current `hierarchical.py` ships a contiguous `data[:0.9]/data[0.9:]`
     split (see `main()`). Replace it with the block-interleaved split before
     trusting any val number.

---

## 5. Pipeline

Run in order. Each stage gates the next.

### 5.1 Flat baselines (`train_verdict.py`)

Establish the char-level anchor and train the byte entropy model that will place
boundaries.

```bash
# char-level anchor (reference point; expect ~BPC 1.74)
python train_verdict.py --data corpus/ --kind char --steps <N>

# byte entropy model — the boundary placer.
# Train it in its OWN directory so it never overwrites the hierarchical best.pt,
# and DELIBERATELY UNDERTRAIN it: a fully converged model smooths away the
# boundary signal. Stop while its entropy still varies sharply at word onsets.
mkdir -p entropy_model
python train_verdict.py --data corpus/ --kind byte --steps <fewer> --out entropy_model/best.pt
```

> Undertraining is intentional. The entropy model's job is to be *sharply
> surprised* at real boundaries, not to be the best possible LM.

### 5.2 Fixed-patch hierarchical baseline (`hierarchical.py`)

The reproducible anchor for the whole study.

```bash
python hierarchical.py --file corpus/<book>.txt --patch_len 6 --patches 32 \
    --d_model 256 --depth 6 --steps <N> --lr 5e-4 --warmup 500
```

Record **val loss, bits/byte, BPC**, and (with the eval-hook patch applied, see
5.4) **hier first/within**. Reference from the study: val ≈ 1.108, BPC ≈ 1.606,
beating the char anchor.

> **Numerical-stability note.** The CG-LRU normalization term
> `sqrt(clamp(1 − mag_sq, min=…))` must floor at `1e-6`, **not** `0.0`, or you
> hit a gradient singularity and NaNs. A secondary guard checks `isfinite(gnorm)`
> before `opt.step()`. Both are in the shipped code; keep them.

### 5.3 Go/no-go gate — boundary quality (`analyze.py`)

Confirm the entropy model places boundaries at linguistically sensible positions
*before* committing to dynamic training.

```bash
python analyze.py --ckpt entropy_model/best.pt --file corpus/<book>.txt \
    --sweep --color --dump boundaries.json --percentile 85
```

Read:
- **Match the operating point.** Mean patch length is a pure function of the
  percentile: `len = 100/(100 − p)`. To match a fixed baseline at stride 6 you
  need `len ≈ 6`, i.e. **p84–85**, *not* the p70 default (len 3.3). The
  text-dependent number in the sweep is the `threshold(bits)` column; the
  `mean_patch_len` column is a tautology.
- **Boundary view.** Boundaries should open at word/sentence onsets; function
  words should be captured as whole-word patches. Expect minor residual
  fragmentation of content-word onsets and of ALL-CAPS/heading text.
- **`--dump` at the matched percentile.** The dumped `boundaries.json` must be at
  the granularity you'll train on, and from the *same* entropy checkpoint that
  feeds `patcher.py`.

**Pass criterion:** boundaries track onsets; top surprises are word-initial
bytes. If instead they're junk, fix/retrain the entropy model before proceeding.

### 5.4 First/within tax measurement (`eval_hook.py`, wired into `hierarchical.py`)

`hierarchical.py` already imports `eval_hook` and accepts `--entropy_ckpt`. Pass
a frozen flat checkpoint and every eval reports the hierarchical first/within
**and** the frozen flat model's first/within on the *same* boundaries:

```bash
# dynamic run -- flat is scored on the REAL boundary mask (the tax gate)
python hierarchical.py --file corpus/<book>.txt --patch_len 6 --patches 32 \
    --boundaries boundaries.npz --steps <N> --entropy_ckpt entropy_model/best.pt

# fixed run -- flat is scored on a synthetic stride-patch_len mask
# (the fixed-stride first/within baseline, for a like-for-like comparison)
python hierarchical.py --file corpus/<book>.txt --patch_len 6 --patches 32 \
    --steps <N> --entropy_ckpt entropy_model/best.pt
```

New per-eval line:

```
… | hier first F within W b/byte | len L | flat first f within w | tax first (F−f) within (W−w)
```

(the fixed run has no boundary mask to score hier's own first/within against, so
it prints only `flat first/within` -- that's the fixed-stride baseline itself.)

Interpretation:
- `flat first/within` are the **full-context reference floors**.
- `tax within` over ~85% of bytes dominates the aggregate gap far more than
  `tax first` over ~15%. Watch the **within** tax specifically: if it is large,
  the decoder is leaking on *easy* bytes (it sees prior patches only through the
  single summary vector, never their raw bytes), and the fix is real cross-patch
  byte context, not a boundary tweak.
- Compare the dynamic run's `hier within` against the fixed run's `flat within`
  (both on stride/boundary-scored full-context floors) for the like-for-like
  dynamic-vs-fixed within comparison -- this is the piece that was never
  measured before.

> Caveat: read the tax as a *floor*. The entropy/flat model is deliberately
> undertrained (overstates surprise) and short-window batches start cold; a
> better full-context model would lower the reference and widen the true tax.

### 5.5 Dynamic experiment (extension)

The dynamic trainer is the extension of `hierarchical.py`; it needs ragged
batching plus the boundary masks from 5.3. Two stages:

- **Stage 1 — external boundaries.** Freeze the entropy model, precompute
  corpus-aligned masks (`analyze.py --dump` at the matched percentile), pad
  variable patches to rectangular batches with masking, train. Change *only* the
  boundary rule vs the fixed baseline. Report BPC + first/within.
- **Stage 2 — endogenous patching.** Add a next-byte head on the encoder's
  per-byte recurrent states so boundary decisions come from the model's own
  uncertainty and work identically at generation time.

Run the 5.4 decomposition on the **same val data the model evaluates on** (not a
hand-picked slice) so `flat_within` vs `hier_within` is purely architectural.

---

## 6. Decision gates

| Gate | Check | If it fails |
|------|-------|-------------|
| Stability | No NaNs; `isfinite(gnorm)` guard active | Restore the `1e-6` floor; lower `--lr` / raise `--warmup`. |
| Split integrity | Train/val sample the same distribution (block-interleaved) | Fix the split before reading any gap as overfitting. |
| Boundary quality (5.3) | Boundaries at word/sentence onsets at matched `len` | Retrain/retune the entropy model or threshold. |
| Where the tax lives (5.4) | `within` tax small vs `first` tax | If `within` tax dominates on matched val → decoder needs cross-patch byte context (bigger change). If it's a slice artifact → boundary-side fixes suffice. |

---

## 7. Principles baked into this protocol

- **Change one variable at a time.** Dynamic vs fixed differ only in the boundary
  rule; keep corpus, split, entropy checkpoint, threshold, and LR schedule fixed
  across the compared runs.
- **Diagnose before fixing.** Separate the diagnosis (e.g. NaN root cause, tax
  location) from the fix.
- **Distribution integrity before overfitting hypotheses.** Verify the split
  samples one distribution before interpreting any train/val gap.
- **Deliberate undertraining of the entropy model** — preserves boundary signal.
- **Aggregate metrics obscure.** BPC is necessary but never sufficient here;
  always pair it with the first/within decomposition.
- **Compare like-for-like.** Never compare dynamic-within to fixed-*total*;
  measure fixed-within on the same positions (5.4).

---

## 8. Findings so far (provisional)

- Fixed baseline: BPC ≈ 1.606 (beats char anchor 1.74).
- Dynamic (entropy) baseline, converged (LR at floor): BPC ≈ 1.94,
  first ≈ 3.94 / within ≈ 1.565, len ≈ 6.4. So the ~0.33 BPC gap is real, not
  undertraining.
- Boundary gate passes at p85 (threshold ≈ 3.0 bits, len ≈ 6.6): onsets, mostly
  whole-word patches.
- On an in-distribution slice, the flat full-context model scores first ≈ 3.54 /
  within ≈ 0.99. Decomposing the gap suggests the **within** bytes, not the
  boundary bytes, carry most of the architectural tax — pointing at the decoder's
  lack of cross-patch byte context rather than at the boundary policy.
- **Unconfirmed:** that within-tax is measured on a small single-file slice
  against an undertrained reference. Confirm on matched val (5.4) before acting
  on it.
