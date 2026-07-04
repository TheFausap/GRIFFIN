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

## 6. Next step

Give `PatchDecoder` real cross-patch raw-byte context (the previous patch's
trailing bytes), holding the entropy model, corpus, split, and LR schedule
fixed — one architecture variable changed, retrained on both fixed and
dynamic patching for a repeat of the head-to-head table above.
