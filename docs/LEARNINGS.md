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

---

*Add entries below as L4, L5, ... during development.*
