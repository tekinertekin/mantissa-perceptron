# mantissa-perceptron

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)
[![Engine](https://img.shields.io/badge/engine-mantissa-00599C.svg)](https://github.com/tekinertekin/mantissa)

Perceptron classifier in Python, powered by the mantissa C engine. Rosenblatt
+ ADALINE rules, classic UCI datasets, honest benchmarks vs scikit-learn —
3.5× leaner RAM, 4.6× faster batch predict.

**Rosenblatt's perceptron, with a C engine.**

A minimal binary classifier (`fit` / `predict` / `score`) whose compute runs
in [mantissa](https://github.com/tekinertekin/mantissa) — a fast, memory-lean
neural-network core in C. The Python layer is thin on purpose: the forward
pass, and under the delta rule the entire training step, happen in C on
zero-copy float32 buffers.

Two classic rules, both honest about what they are:

- `rule="perceptron"` — Rosenblatt (1958), mistake-driven; converges in
  finitely many mistakes on linearly separable data (Novikoff, 1962).
- `rule="delta"` — Widrow-Hoff LMS / ADALINE (1960); each step is a single
  `tk_train_step_f32` call into the engine. The one to use when the data is
  not separable.

Minimal on purpose: binary only, dense float32, no multiclass, no sparse.
sklearn-like interface, not sklearn-compatible.

## Install

```sh
pip install mantissa-perceptron   # after PyPI publication
```

This pulls in the engine (`mantissa-nn`) automatically.

From a checkout (works today, no PyPI needed): clone this repo next to
[mantissa](https://github.com/tekinertekin/mantissa), build the engine
(`make dist` there), then `pip install -e .` here — the package finds the
sibling checkout automatically.

## Quickstart

```python
from mantissa_perceptron import Perceptron, datasets

X, y = datasets.load("banknote")            # prints the curl command if missing
Xtr, Xte, ytr, yte = datasets.split(X, y)   # seeded, stratified, standardized
clf = Perceptron().fit(Xtr, ytr)
print(clf.score(Xte, yte))
```

## Datasets

Five small classics (UCI + the standard Pima mirror). **Nothing downloads
implicitly** — `data/` is gitignored and library code never touches the
network. Fetch explicitly:

```sh
python -m mantissa_perceptron fetch all     # or a single name
python -m mantissa_perceptron list
```

| name | n | d | task |
|------|---|---|------|
| iris | 100 | 4 | setosa vs versicolor (linearly separable) |
| banknote | 1372 | 4 | genuine vs forged banknotes |
| breast_cancer | 569 | 30 | WDBC malignant vs benign |
| sonar | 208 | 60 | mine vs rock |
| pima | 768 | 8 | diabetes onset |

## Results

### Accuracy

<!-- BEGIN:ACCURACY (owned by Dev A — bench/accuracy.py output; do not edit outside these markers) -->
Test accuracy on the held-out 25% (stratified split, seed 42, features
standardized on train statistics only), 100 epochs. The `perceptron` rule
uses `lr=1.0` (the boundary is lr-invariant at zero-init); the `delta`
(ADALINE) rule's `lr` is tuned per dataset on the **train set only** over
{0.001, 0.003, 0.01, 0.03}. Numbers as measured by `python -m bench.accuracy`.

| dataset | n | d | rule | lr | train acc | test acc | epochs run | converged |
|---|---|---|---|---|---|---|---|---|
| iris | 100 | 4 | perceptron | 1 | 1.000 | 1.000 | 3 | yes |
| iris | 100 | 4 | delta | 0.001 | 1.000 | 1.000 | 1 | yes |
| banknote | 1372 | 4 | perceptron | 1 | 0.991 | 0.980 | 100 | no |
| banknote | 1372 | 4 | delta | 0.001 | 0.980 | 0.968 | 100 | no |
| breast_cancer | 569 | 30 | perceptron | 1 | 0.995 | 0.944 | 100 | no |
| breast_cancer | 569 | 30 | delta | 0.001 | 0.970 | 0.937 | 100 | no |
| sonar | 208 | 60 | perceptron | 1 | 0.929 | 0.788 | 100 | no |
| sonar | 208 | 60 | delta | 0.001 | 0.910 | 0.731 | 100 | no |
| pima | 768 | 8 | perceptron | 1 | 0.674 | 0.661 | 100 | no |
| pima | 768 | 8 | delta | 0.001 | 0.780 | 0.771 | 100 | no |

iris is linearly separable, so the perceptron converges to a perfect
boundary (Novikoff). banknote and breast_cancer are nearly separable. On
the non-separable sets the honest picture shows: sonar (60 features, 208
samples) overfits — high train, weak test; pima is genuinely hard for a
linear model, and there the delta rule's smoother LMS objective beats the
oscillating mistake-driven rule on both train and test.
<!-- END:ACCURACY -->

### Speed and memory vs famous implementations

<!-- BEGIN:BENCH (owned by Dev B — bench/speed.py + bench/plots.py; do not edit outside these markers) -->
On the largest dataset (**banknote**, 1372×4, 100-epoch cap, 15 interleaved
repeats, medians):

| contender | fit (ms) ↓ | predict (ms) ↓ | peak RSS (MB) ↓ |
|-----------|-----------:|---------------:|----------------:|
| ours (perceptron) | 1.81 | 0.011 | **27.0** |
| ours (delta)      | 3.68 | 0.010 | **27.1** |
| scikit-learn      | **1.31** | 0.045 | 93.6 |
| numpy (hand-rolled) | 52.46 | **0.006** | 26.6 |
| pure Python       | 38.76 | 0.092 | 26.6 |

*torch omitted — not importable in this environment; the harness includes it
automatically when it is.*

**The story of this table is the two engine releases it forced.** The first
run measured our fits at **661 ms (delta)** and **535 ms (Rosenblatt)** — every
training sample paid a Python→C crossing. Both findings went upstream
([`docs/LEARNINGS.md`](docs/LEARNINGS.md)): mantissa v0.1.11 moved the delta
epoch into one C call (661 → 4.4 ms), and the dataset-epoch primitives that
followed moved the Rosenblatt rule and the shuffle order in as well:

- **Rosenblatt: 535 → 1.81 ms (295×)** — one `tk_perceptron_epoch_f32`
  call per epoch, mistake count included. Now within ~1.4× of scikit-learn's
  Cython SGD, *with* honest early stopping (we stop at zero training mistakes;
  sklearn with `tol=None` never does — on separable data our fit is a few
  epochs, not 100).
- **Delta: 661 → 3.68 ms** — ordered epochs (the shuffle permutation
  crosses as an int32 index array; no per-epoch row copies) plus one post-epoch
  convergence pass, kept deliberately: LMS updates on every sample, so only a
  post-epoch count certifies the final weights (an in-epoch count was measured
  faster but rejected on exactly that semantic).
- **Memory: 27.0 MB vs scikit-learn's 93.6 — 3.5× leaner**, unchanged.
- **Batch predict: 0.010 ms — ~4.7× faster than scikit-learn** (one
  threaded C call into a caller buffer).
- **Accuracy: at parity with scikit-learn.** One honest shift: the Rosenblatt
  rows moved a few tenths of a point (banknote 0.985 → 0.980, WDBC 0.951 →
  0.944) because the old per-sample path trained *through* the engine's
  bfloat16 storage quantization, while `tk_perceptron_epoch_f32` is pure
  float32 end-to-end — the documented semantic for the `_f32` training family.
  Delta rows are bit-identical to before (the ordered epoch is pinned
  bit-identical in mantissa's test suite).

![test accuracy per dataset: ours vs scikit-learn](assets/accuracy.png)
![median fit time per dataset per contender, log scale](assets/fit_time.png)
![peak RSS per contender, import plus fit](assets/peak_rss.png)

**Fairness caveats.**
- scikit-learn's `Perceptron` is Cython SGD doing strictly more bookkeeping per
  epoch; the remaining ~1.4× gap is ~1 ctypes crossing per epoch plus the
  Python-side `rng.permutation` — the C epoch itself is ~4 µs.
- We set `tol=None` on scikit-learn to disable its early stopping and equalize
  the 100-epoch budget; our early stop fires only at zero training mistakes.
- `numpy`/`pure Python` implement the same mistake-driven rule in-process and
  are the honest baseline for what the old per-sample FFI loop was costing.

**Environment.** Apple M4 · Python 3.9.6 · numpy 2.0.2 · scikit-learn 1.6.1 ·
mantissa 0.1.13+ dtype bfloat16 · threads default(10) · 2026-07-13. Full raw
samples and versions in `bench/results/speed.json` (regenerable, gitignored).
<!-- END:BENCH -->

### Methodology

Fixed protocol: stratified 75/25 split (seed 42), features standardized on
train statistics only, epochs capped identically for every contender.
Timings are medians over interleaved repeats on one machine (library
versions and CPU recorded in `bench/results/speed.json`); peak RSS is
measured per contender in a fresh subprocess, import cost included, because
that is what a user pays. *Measure, don't assume.*

## License

MIT — © Tekin Ertekin. Engine:
[mantissa](https://github.com/tekinertekin/mantissa), same author, MIT.
