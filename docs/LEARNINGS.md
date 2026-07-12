# Learnings for mantissa

Channel to the mantissa Chief Architect. Every entry must be a
**measurable idea**: observation → hypothesis → experiment → expected
win. No vibes; numbers or a plan to get them.

---

## L1. Per-call FFI overhead dominates small-layer training (hypothesis)

- **Observation (architect, design-time)**: a perceptron step on d≤60
  features is ~100 FLOPs, but each `tk.train_step` crosses ctypes once
  per sample. Microseconds of FFI around nanoseconds of math.
- **Idea**: a batched epoch entry point in C, e.g.
  `tk_train_epoch_f32(W, bias, X, Y, n_samples, out_dim, in_dim, act, lr,
  perm)` — one crossing per epoch instead of per sample.
- **Measure**: Dev B's fit-time numbers give the baseline; prototype the C
  loop and compare steps/s on banknote (1372×4) and sonar (208×60).
  Expected: ≥10× on these shapes if FFI-bound, ~1× if not — either result
  is information.

## L2. `linear_forward` returns a boxed Python list

- **Observation**: the binding materializes `list(yc)` on every call; the
  batch-predict trick (X-as-W GEMV) returns n floats that we immediately
  re-wrap in numpy.
- **Idea**: accept an optional preallocated float32 `out=` buffer (numpy /
  `array('f')`) and skip list construction, mirroring the zero-copy input
  path.
- **Measure**: batch predict on banknote test split (343×4), median of
  1000 calls, list path vs out-buffer path.

## L3. STEP/SIGN gradient is defined as 0

- **Observation**: `tk_act_grad_scalar` returns 0 for STEP/SIGN, so
  `train_step(act=STEP)` is a silent no-op — we hit this designing the
  perceptron rule.
- **Idea (API, not perf)**: worth a one-line note in mantissa's USAGE.md
  ("step/sign are inference-only activations for training entry points"),
  or an assert in debug builds. Zero cost, saves the next integrator an
  afternoon.

## L4. Measured: FFI overhead confirms L1 — we lose fit time to per-sample crossings

- **Observation (Dev B, `bench/speed.py`, banknote 1372×4, 100-epoch cap,
  15 interleaved repeats, medians, Apple M4)**: median fit times —
  - `ours (perceptron)`  537 ms  (one `linear_forward` crossing / sample)
  - `ours (delta)`       661 ms  (one `train_step` crossing / sample)
  - hand-rolled numpy     52 ms  (same rule, stays in Python/numpy)
  - pure Python           40 ms  (same rule, no numpy)
  - scikit-learn SGD     1.4 ms  (Cython, one crossing / fit)
  We are **~10–13× slower than the naive in-process loops** doing the
  *identical* mistake-driven rule, and ~380× slower than scikit-learn. The
  arithmetic is ~100 FLOPs/sample; the gap is pure `ctypes` call overhead,
  exactly as L1 predicted. Confirmed across all five datasets (see plot).
- **Contrast — the crossing count is the whole story**: batch `predict` makes
  *one* crossing for the entire test set (X-as-W GEMV) and lands at 0.029 ms,
  beating scikit-learn (0.047 ms). Same engine, ~1 call vs n calls.
- **Expected win**: the `tk_train_epoch_f32` batched entry point from L1
  would cut banknote from n·100 ≈ 137k crossings to 100. If FFI-bound (it is),
  fit should collapse from ~537 ms toward the ~40–50 ms in-process baseline —
  a ≥10× win — and likely undercut it once the epoch loop is C, not Python.
  This is the single highest-leverage change for training throughput.

## L5. Measured: engine is memory-lean — 3.5× smaller RSS than scikit-learn

- **Observation (Dev B, peak RSS in a fresh subprocess, import + one fit,
  banknote)**: `ours` 26.7 MB vs `scikit-learn` 93.4 MB; hand-rolled numpy
  and pure Python also ~26.6 MB. The mantissa dylib adds only ~70 KB over the
  interpreter+numpy floor.
- **Idea**: this is a genuine selling point worth *guarding*, not just noting.
  A CI check that fails if `import mantissa` + a fit exceeds, say, 30 MB on
  this shape would keep the lean-footprint claim honest as the engine grows.
- **Measure**: already have the harness (`bench/speed.py --worker`); wire its
  `ru_maxrss` read into a threshold assertion.

---

*Add entries below as L6, L7, ... during development.*
