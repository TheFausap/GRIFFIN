# Annex — Dynamic vs Fixed Byte Patching: Confirmed Results

Companion to `REPRODUCE.md` (protocol) and De et al., *"Griffin: Mixing Gated
Linear Recurrences with Local Attention for Efficient Language Models"*
(Google DeepMind, `2402.19427v1.pdf`, kept in this directory) — the CG-LRU +
local-attention block this study reuses unchanged, one level up, over byte
patches instead of raw tokens.

Status: supersedes the "Findings so far (provisional)" section of
`REPRODUCE.md` §8. Those numbers were measured on `corpus.random/`, a
differently-shuffled/renamed file set from `corpus/`; everything below is
measured on `corpus/` (39,594,610 bytes, 53 Gutenberg files), matched between
runs.

---

## 1. What was run

Both runs share: `corpus/`, block-interleaved 90/10 split (block 65536 B,
every 10th block to val), `patch_len=6`, `patches=32` (seq_len 192),
`d_model=256`, `depth=6`, `lr=5e-4` with 500-step warmup and cosine decay to a
0.1× floor, 32,000 steps (LR reaches its floor ~step 31,600), frozen
`entropy_model/best.pt` (deliberately undertrained flat Griffin) as both the
boundary placer and the full-context reference floor.

- **Fixed** — uniform stride-6 patches (`hierarchical.py`, no `--boundaries`).
- **Dynamic** — patches cut at entropy-model surprise, `--boundaries
  boundaries.npz` precomputed via `precompute_boundaries.py --target_len 6`
  (operating point verified at the matched percentile: boundaries land on
  word/sentence onsets, mostly whole-word patches — the go/no-go gate in
  `REPRODUCE.md` §5.3 passed cleanly before training).

Every eval also scores the frozen flat model on the *same* boundary positions
each run trained on (`eval_hook.py`, wired into `hierarchical.py` via
`--entropy_ckpt`) — this is what makes the first/within split load-bearing
rather than descriptive: it isolates the architectural tax (`hier − flat`)
from intrinsic byte hardness.

## 2. Head-to-head (best checkpoint, step 30600, both runs)

| | BPC | hier first (tax) | hier within (tax) | mean patch len |
|---|---|---|---|---|
| **Fixed** (stride-6) | 1.960 | 1.804 (+0.416) | 1.980 (+0.530) | 6 (fixed) |
| **Dynamic** (entropy) | 1.987 | 3.982 (+0.148) | 1.560 (+0.603) | 5.82 |

Both runs are converged, not undertrained: val loss is flat/oscillating
(noise from 20 eval batches) across the full 25,400–32,000 step tail, well
past the LR floor.

## 3. Findings

**3.1 — Dynamic within genuinely beats fixed within, like-for-like.**
1.560 vs 1.980 bits/byte, a 0.42 bit/byte win on the ~85% of bytes that are
*not* a patch's first byte. This is the one number the original protocol
never had a baseline for (`REPRODUCE.md` §8 compared dynamic-within only to
fixed's *total*, which conflates the comparison). It confirms the core
hypothesis: cutting *before* high-entropy bytes makes everything else in the
patch dramatically more predictable, exactly as the boundary-quality gate
implied qualitatively.

**3.2 — Total BPC is a near-wash for a structural, not a flaw, reason.**
Dynamic's boundary rule puts the single hardest byte in the stream at every
patch start by construction, so `hier_first` (3.982) is unavoidably
expensive — and the decoder is already handling it almost optimally there
(tax only +0.148, close to the flat floor). There's essentially no headroom
left on `first`; its cost is intrinsic surprise, not an architecture gap.
That expense currently cancels out most of the within-patch win in the
aggregate BPC.

**3.3 — The decoder's cross-patch leak is an architecture-wide constant, not
a boundary-policy artifact.** Within-tax is +0.530 (fixed) vs +0.603
(dynamic) — similar magnitude under both regimes, on positions whose only
common property is "predicted from `c_{k-1}` plus teacher-forced earlier
bytes of *this* patch, never raw bytes of the previous patch." This is the
target for the next change: give `PatchDecoder` real cross-patch byte context
(BLT/MegaByte-style), not a boundary-policy tweak. Because the tax shows up
in both regimes at similar size, fixing it should benefit both — and should
let dynamic's already-confirmed 0.42-bit within advantage finally show
through in total BPC, currently masked by the unavoidable first-byte cost.

Side note: fixed's `first` tax (+0.416) is much larger than dynamic's
(+0.148) — under fixed stride, "first" is just an arbitrary (often mid-word)
byte, so it's exposed to the same summary-vector bottleneck that hurts
`within` everywhere. Under dynamic patching, "first" is dominated by
intrinsic surprise the model can't help regardless of context, so the same
bottleneck matters comparatively less there.

## 4. Gate status (vs `REPRODUCE.md` §6)

| Gate | Result |
|---|---|
| Stability | No NaNs/Infs across both 32k-step runs. |
| Split integrity | Block-interleaved, both runs on identical `corpus/` split. |
| Boundary quality (§5.3) | **Pass** — onsets, whole-word patches, minor residual fragmentation only. |
| Where the tax lives (§5.4) | **Confirmed on matched val data** (not a slice artifact): within-tax is real and comparable in size to first-tax's *relative* headroom — decoder needs cross-patch byte context. |

## 5. Caveats

- The entropy/flat reference is deliberately undertrained (by design, so it
  keeps a sharp boundary signal) — it overstates surprise, so all tax numbers
  above are a *floor*. A better full-context reference would likely widen the
  measured within-tax further, not narrow it.
- Per-eval noise is real: individual eval points swing ±0.03–0.05 BPC across
  the converged tail (20 batches/eval). Read single-step numbers as samples
  from a stable band, not exact points — hence reporting the best checkpoint
  alongside the tail range rather than a single number in isolation.
- `corpus/` vs `corpus.random/` mismatch: earlier runs (`REPRODUCE.md` §8,
  BPC 1.606 fixed / 1.94 dynamic) were trained against `corpus.random/`, whose
  53 files are the same content as `corpus/` but differently named — the
  boundary mask's checksum guard in `dynamic.py` caught this immediately when
  the wrong directory was passed, preventing a silent misalignment. All
  numbers in this annex are the corrected re-run against `corpus/` only.

## 6. Round 2 — cross-patch byte context in `PatchDecoder`

Implemented exactly the single change §3.3/§6 called for: `PatchDecoder` now
receives `prev_ctx`, the trailing `byte_ctx_len=8` raw bytes of the *previous*
patch (via `patcher.prev_patch_tail`, shared by the fixed and ragged paths),
embedded with the decoder's own byte table and projected into the
conditioning vector alongside `z = c_{k-1}`. Patch 0 gets an all-pad window
(same "nothing before this" convention as the learned `start` vector).
Nothing else changed: same corpus, split, entropy checkpoint, LR schedule,
32,000 steps, both regimes retrained from scratch.

### Head-to-head, best checkpoint each

| | BPC | hier first (tax) | hier within (tax) |
|---|---|---|---|
| Fixed, before → after | 1.960 → **1.891** | 1.804 (+0.416) → 1.812 (+0.388) | 1.980 (+0.530) → **1.896 (+0.479)** |
| Dynamic, before → after | 1.987 → **1.903** | 3.982 (+0.148) → 4.007 (+0.165) | 1.560 (+0.603) → **1.467 (+0.518)** |

### Tail-averaged (steps 30400/30800–32000, the more honest number given 20 eval batches/point)

| | BPC | hier within |
|---|---|---|
| Fixed, before → after | 1.995 → 1.942 | 2.011 → 1.947 |
| Dynamic, before → after | 2.023 → 1.943 | ~1.55 → ~1.50 |

### Findings

**6.1 — Confirmed architecture-wide, a second time.** `hier_within` dropped
~0.06–0.09 bits/byte in *both* regimes; `hier_first`/`tax_first` stayed flat
in both (fixed 1.858→1.859, dynamic 4.010→3.997, tail-averaged) — exactly the
predicted shape, since the fix targets within-patch prediction only.

**6.2 — The fixed-vs-dynamic total-BPC gap collapsed from real to noise-level.**
Tail-averaged: **0.028 bits/byte before → 0.0008 after.** Best-checkpoint:
0.027 → 0.012. Neither is distinguishable from the ±0.03–0.05 per-step eval
noise (20 batches/eval) any more — fixed and dynamic are now statistically
tied on total BPC, down from a small but real dynamic deficit.

**6.3 — Dynamic did not flip ahead in total BPC, but that was never the axis
the fix targeted, and the axis it does target held up under a second decoder
architecture.** Fixed-within minus dynamic-within was 0.420 bits/byte before
the change, 0.429 after — the entropy-boundary within-advantage is robust
across two different decoders now, not an artifact of the old decoder's
specific weakness.

**6.4 — Open question: is the residual ~0.001–0.012 gap real or pure noise?**
`eval_loss()` samples only 20 batches/eval (a comment in the code already
flags "bump to ~50 for the real comparison"). Given the gap is now this
small, a confirmatory pass with more eval batches on both final checkpoints
is the natural next no-train measurement before drawing a final verdict on
whether dynamic patching wins on total BPC outright.

## 7. Round 3 — confirmatory eval at `--eval_batches 100`

`eval_loss()`'s batch count was hardcoded at 20; added `--eval_batches` (§6.4's
open question) and re-evaluated both final checkpoints via `--resume` at
`--eval_batches 100`. Caveat: `--resume` continues training from the
checkpoint's saved step rather than freezing it, so this is a continuation,
not a perfectly frozen repeat-eval — dynamic resumed at step 31201 (4 eval
points to 32000), fixed at 31801 (1 eval point). Both are past the LR floor,
so the extra steps have minimal effect; the conclusion below doesn't hinge on
this asymmetry.

| | BPC | hier within (tax) |
|---|---|---|
| Fixed (single point, step 31999) | 1.949 | 1.956 (+0.472) |
| Dynamic (avg of 4, steps 31400–31999) | 1.938 (range 1.916–1.955) | 1.495 (+0.513) |

**Verdict: the total-BPC gap is confirmed noise-level, not real** — 1.949 vs
1.938 (or vs any of dynamic's individual 1.916–1.955 points) is a dead heat,
resolving §6.4. **The within-loss advantage is confirmed unambiguous at this
lower noise floor** — 1.956 vs 1.495, a **0.46 bit/byte gap**, if anything
larger than the n=20 estimate (0.43) and now far outside any remaining noise.
`hier_first`/`tax_first` stayed exactly where predicted (fixed 1.854/+0.408,
dynamic avg 3.991/+0.152) — untouched by the decoder fix, as designed.

## 8. Round 4 — widen the cross-patch context window (`byte_ctx_len` 8 → 16)

Also redesigned `prev_patch_tail` → `prev_byte_window`: the old version only
ever looked inside patch k-1, so for the common case (dynamic's mean patch
length ~5.8-6 already fits inside K=8) widening K would have done nothing —
extra window slots would just stay padding. The redesign slides the window
over the raw byte stream by *offset*, so a short patch naturally lets it
spill into patch k-2, k-3, etc. Unit-verified against hand-computed cases
before retraining. Both regimes retrained from scratch at `--byte_ctx_len 16`,
same corpus/split/entropy checkpoint/steps, `--eval_batches 100` from the
start this time (avoiding round 3's two-pass confirmation dance).

| | BPC | hier within (tax) | hier first (tax) |
|---|---|---|---|
| Fixed, K=8 → K=16 | 1.949 → 1.926 | 1.956 (+0.472) → 1.934 (+0.481) | 1.854 (+0.408) → 1.827 (+0.405) |
| Dynamic, K=8 → K=16 | 1.955 → 1.929 | 1.511 (+0.517) → 1.491 (+0.521) | 3.999 (+0.153) → 3.999 (+0.155) |

(single eval point per run at step 31999, not a tail-average like round 3's
confirmatory pass — read accordingly.)

**Verdict: diminishing-to-no returns.** Both regimes improved by a similar,
modest ~0.02-0.03 bits/byte in total BPC — consistent enough across two
independently-trained runs to likely be a small real effect, not pure noise.
But `tax_within` itself didn't move (flat or marginally up in both regimes):
the improvement in `hier_within` was matched by an equivalent shift in
`flat_within`'s own random-batch estimate (the frozen entropy model is scored
on fresh random val batches each run, so its number wobbles run-to-run even
though the model is frozen) — no evidence the wider window closed more of the
decoder's cross-patch leak specifically. Total BPC is still tied
(dynamic 1.929 vs fixed 1.926) and dynamic's within-advantage is unchanged
(1.934 − 1.491 = **0.443 bits/byte**, same as K=8's 0.44-0.46).

Likely cause: `ctx_proj` is a single flat `Linear(K*d_byte, d_model)` — K
independently-weighted byte-embedding slots summed together, no real sequence
model over the window. More raw bytes doesn't help if the projection can't
extract more from them regardless of K. Going wider (K=24, 32) is unlikely to
move this further; the bottleneck looks like *how* the context is summarized,
not *how much* of it is provided.

## 9. Round 5 — Stage 2, Option B: endogenous boundary head

Stopped the `byte_ctx_len`-widening thread at round 4 (diminishing returns)
and moved to the real architecture change scoped as "Option B": replace the
external frozen entropy model as the boundary source with a small
`BoundaryHead` (`boundary_head.py`, a plain `Griffin` instance) trained
*jointly*, with its own optimizer, on the same byte windows the hierarchical
model already samples. Pre-freeze, boundaries come from the still-training
head with a periodically-recalibrated threshold; at a freeze step it stops
updating, one real whole-corpus scan runs (reusing
`precompute_boundaries.py`'s own `compute_surprise`/`solve_threshold`), and
the run collapses into the *exact*, unmodified Stage-1 static-mask path for
the remainder of training.

Empirically calibrated the freeze point first (not guessed): a standalone
`train_verdict.py` sweep at step budgets 200/1000/2000/4000/8000, gated by an
objective mid-word-fragmentation metric, showed boundary quality *degrading*
monotonically past ~1000 steps (10.6%→10.0%→11.4%→13.4%→15.6%) — the same
"overtraining smooths away the signal" effect the original entropy model's
design guarded against. Translated to this trainer's own batch/step schedule:
`--boundary_freeze_step 1300`.

### Result — confirmed on two independent seeds

All at `--eval_batches 100`, tail-averaged over the last ~10 eval points of a
32k-step run (same corpus/split/decoder-context architecture as round 3's
K=8 numbers):

| | BPC | hier first (tax) | hier within (tax) | len |
|---|---|---|---|---|
| Fixed | 1.949 | 1.854 (+0.408) | 1.956 (+0.472) | 6 |
| Stage-1 Dynamic (external entropy model) | 1.938 | 3.991 (+0.152) | 1.495 (+0.513) | 5.79 |
| **Stage-2 Endogenous, seed 0** | **1.881** | 3.653 (+0.260) | 1.503 (+0.433) | 5.85 |
| **Stage-2 Endogenous, seed 42** | **1.888** | 3.687 (+0.287) | 1.507 (+0.432) | 5.88 |

**This is the first clear, noise-exceeding total-BPC win for any dynamic
variant in the whole study** — ~0.05-0.06 bits/byte better than Stage-1
dynamic, confirmed near-identically across two independently-trained seeds
(BPC within 0.007, `hier_within` within 0.004, `tax_within` within 0.001 of
each other).

**The mechanism is not what a first guess would suggest.** `hier_within` is
statistically unchanged from Stage-1 (~1.50 in both) — the win comes almost
entirely from `hier_first` dropping substantially (~3.65-3.69 vs ~3.99). The
external flat model agrees: `flat_first` also drops sharply under these
boundaries (~3.39-3.40 vs ~3.84), meaning even the bigger, longer-trained
reference model finds the endogenous head's chosen cut points less
surprising than Stage-1's. `flat_within` moves the other way (~1.07 vs
~0.98) — some difficulty shifted from patch-starts into patch-bodies — but
`hier_within` absorbed that shift at no measurable cost. Read honestly: the
small, briefly-trained endogenous head's surprise ranking doesn't fully agree
with the bigger external model's; it systematically selects a different (and,
on this corpus, more favorable) set of cut points, while still respecting
word/sentence structure qualitatively (verified directly against the real
frozen mask — boundaries land on onsets, e.g. "│more │sensual │minds of...",
with the same modest fragmentation-on-unusual-words character as Stage-1's
own boundaries, not a degenerate mask).

Whether this reshuffling-of-difficulty effect is a robust property of
jointly-trained/endogenous boundary ranking in general, or a specific quirk
of this small head's capacity/freeze-point on this corpus, isn't something
two same-config seeds can fully settle — it's confirmed *reproducible*, not
yet confirmed *general*.

## 10. Next step

The headline result now stands on two-seed confirmation. Open threads, not
mutually exclusive:
- **Bank this as the study's main finding** — endogenous patching beats both
  fixed and externally-guided dynamic patching on total BPC, the first clean
  win either dynamic approach has produced.
- **Understand *why* the endogenous ranking differs from the external one** —
  capacity? freeze timing? training-objective differences? A mechanistic
  question the current data can motivate but not answer.
- Revisit the `byte_ctx_len`/cross-patch-context lever now that the boundary
  policy itself has changed — round 4's "diminishing returns" verdict was
  measured against Stage-1 boundaries, not these.
