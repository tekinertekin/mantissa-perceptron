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

> **STATUS: LANDED UPSTREAM.** mantissa v0.1.11 shipped `tk_train_epoch_f32`
> (whole epoch in one crossing; 141x measured in isolation) and v0.1.12 added
> `tk_linear_forward_batch` + `out=`. Adopted here: delta fit 661 -> 4.39 ms
> (151x) on banknote, accuracy bit-identical. The loop closed exactly as this
> entry predicted.

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

## L6. Measured: delta-fit per-epoch breakdown — two FFI crossings are the floor

- **Observation (perf, per-epoch profile, banknote 1030×4 train, delta,
  lr=0.01, 100 epochs, Apple M4, perf_counter medians of 500–3000 runs)**.
  fit() = 4.21 ms total; per-epoch cost decomposes as:

  | component                              | ms/epoch | ×100  | share |
  |----------------------------------------|----------|-------|-------|
  | (c) `train_epoch` FFI call             | 0.0157   | 1.57  | 37.2% |
  | (d) post-epoch mistake count           | 0.0102   | 1.02  | 24.2% |
  |    ↳ decision_function forward (FFI)    | 0.0088   | 0.88  |       |
  |    ↳ numpy `* t <= 0` compare           | 0.0014   | 0.14  |       |
  | (b) shuffle gather copies (X + t)      | 0.0062   | 0.62  | 14.7% |
  | (a) `rng.permutation(n)`               | 0.0058   | 0.58  | 13.7% |
  | (e) Python glue (attr/method/loop)     | ~0.0033  | 0.33  | ~8%   |

  **The two FFI crossings per epoch (c + the forward inside d = 0.0245 ms)
  are 58% of the time and cannot be touched from Python.** Everything else
  is host-side bookkeeping.
- **`ascontiguousarray(X[order])` is NOT a double copy**: fancy-indexing axis
  0 of a C-contiguous array already yields a contiguous array, so
  `ascontiguousarray` returns the *same object*. The cost in (b) is the
  gather itself, not a redundant copy.

## L7. Applied (Python-only): 4.31 → 3.97 ms delta fit, bit-identical

- **Change (perf, `perceptron.py::fit`, semantics-preserving)**: (1) gather the
  shuffled batch with `np.take(..., out=Xe)` into a persistent buffer instead
  of allocating a fresh array each epoch (0.0052 → 0.0030 ms/epoch); (2) inline
  the post-epoch forward with a hoisted engine handle + a persistent `z=`
  buffer, skipping `decision_function`'s per-epoch `_check_X`/`engine()`/
  `load_mantissa()` re-lookups (0.0094 → 0.0085 ms/epoch); (3) hoist all
  per-fit invariants (w/b/lr/rule/shuffle) out of the loop; (4) drop the wasted
  `np.arange(n)` on the delta `shuffle=False` path.
- **Result (bench/speed.py protocol, R=15 interleaved, banknote)**: delta
  4.31 → 3.97 ms (**7.8%**, stable across runs); weights, bias, and `errors_`
  trajectory **bit-identical**; accuracy.json byte-identical; 17 tests green.
- **Tried and DROPPED — batch-precomputed permutations**: generating all 100
  epoch orders in one `rng.permuted` call is only ~0.0006 ms/epoch faster AND
  **consumes the RNG in a different sequence** → different shuffle orders →
  moves the weight trajectory and accuracy.json. Not worth an accuracy break
  for 0.06 ms. Per-epoch `rng.permutation` kept verbatim.

## L8. The ceiling: only C can cross the ~3 ms floor — spec for a one-crossing fit

- **Measured floor**: with every Python cost removed, delta fit cannot beat
  `2 × 0.0088-ish` … concretely the irreducible per-epoch work is the
  `train_epoch` crossing (1.57 ms/100) + the post-epoch forward (0.88 ms/100)
  + `rng.permutation` (0.58 ms/100) ≈ **3.0 ms**. scikit-learn does the whole
  100-epoch fit in **1.4 ms** because it makes *one* Cython entry per `fit()`,
  not 200 FFI crossings. **We cannot match it while the epoch loop lives in
  Python.** This is the crossing-count story from L4, one level up.
- **Highest-leverage C primitives** (integration already guarded in
  `perceptron.py` via `hasattr(engine, ...)`, inert until they land):
  1. **`train_epoch_ord`** — order-index epoch that gathers rows inside the
     one crossing. Removes cost (b) entirely (0.62 ms). Expected delta fit
     3.97 → ~3.4 ms. Contract: `train_epoch_ord(W, X, targets, order, n,
     out_dim, in_dim, act, lr, bias=)`, row `order[k]` visited at step k,
     bit-identical to shuffling first.
  2. **`tk_perceptron_epoch_f32`** — Rosenblatt epoch in C. The perceptron
     rule is 552 ms because it makes **n·100 ≈ 103k** `linear_forward`
     crossings (one per sample). A C epoch collapses that to 100. This in-epoch
     mistake count *is* the perceptron `errors_` semantic (mistakes counted
     as encountered, mid-epoch), so it is accuracy-safe — **unlike delta**,
     whose `errors_` is POST-epoch and must not be replaced by an in-epoch
     count (would move `errors_`, hence `epochs_run`/`converged`, hence
     accuracy.json). Must accept the exact `order` array to preserve the
     sequential-update trajectory.
  3. **`tk_delta_fit` / whole-loop-in-C** (the real ceiling-breaker): move the
     entire multi-epoch loop into one FFI crossing per `fit()` — permutation
     (from a seed), all epochs, POST-epoch mistake counting + zero-mistake
     early stop, returning the `errors_` array. Collapses delta's 200
     crossings/fit to 1; expected to land at or under scikit-learn's 1.4 ms.
     This, not incremental primitives, is what closes the gap at this scale.

## L9. After the crossings were few, the crossings themselves were fat — pre-bound pointers (mantissa v0.1.14 `Trainer`)

- **Measured** (banknote 1030×4, M4): one `tk.perceptron_epoch` wrapper call
  is ~9.8 µs — but the raw C entry with cached pointers is **~3.0 µs**. The
  other ~7 µs is `_as_c_float`×4 + `_as_c_int32` re-deriving ctypes pointers
  (~1.3 µs each; numpy's `.ctypes` property alone is ~0.6 µs) for buffers
  that never change across epochs. L4/L8 counted crossings; this is the cost
  *per crossing* once the count is minimal.
- **Fix, upstream** (the L4 pattern again): mantissa v0.1.14 adds
  `tk.trainer(W, X, t, n, 1, d, bias=b)` — pointers bound once per fit,
  per-epoch calls pass only `lr`/`order`. Bit-identical trajectories
  (pinned by comparison over shuffled epochs, both rules), so accuracy.json
  does not move. Interleaved medians: `perceptron_epoch` 9.8 → 4.8 µs,
  ordered `train_epoch` 19.5 → 14.0 µs, post-epoch margins GEMV
  (`Trainer.margins`) 11.8 → 5.8 µs. `fit()` uses it when present; the
  `inspect.signature` feature probe (measured 14 µs — a fifth of an iris
  fit!) now runs only on the fallback path.
- **Measured and rejected**: batching all 100 epochs' orders with
  `rng.permuted` (11.3 µs/epoch) or an in-place int32 shuffle (11.9 µs) —
  both lose to per-epoch `rng.permutation(n).astype(int32)` (6.4 µs at
  n=1030). (L7 already rejected `rng.permuted` for RNG-stream reasons; it
  also just loses the race.)
- **Result**: Rosenblatt fit now *beats* scikit-learn on every dataset
  (banknote 1.81 → 1.30 ms vs sklearn's 1.44, interleaved). L8's ceiling is
  revised: delta's remaining per-epoch cost is dominated by real C work
  (~12 µs raw ordered epoch — LMS updates every sample, unlike the
  mistake-only rules) + 6.4 µs permutation; the whole-fit-in-C projection
  predated `Trainer` and would now remove ~5 µs/epoch of overhead, not ~15.
- **Benchmark-harness bug found while adding the TensorFlow contender**:
  enumerating contenders imported tensorflow into every peak-RSS worker,
  flattening the whole RSS column to ~460 MB. Availability is now probed
  with `importlib.util.find_spec` (no import); only the TF worker pays TF's
  ~450 MB.

---

*Add entries below as L10, L11, ... during development.*
