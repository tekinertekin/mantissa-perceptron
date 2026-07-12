# mantissa-perceptron

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-3776AB.svg)

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
pip install mantissa-perceptron   # TODO(packaging): confirm final names
```

Development layout: clone this repo next to
[mantissa](https://github.com/tekinertekin/mantissa) and build the engine
(`make dist` there); the package finds the sibling checkout automatically.

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
*Pending — produced by `python -m bench.accuracy`.*
<!-- END:ACCURACY -->

### Speed and memory vs famous implementations

<!-- BEGIN:BENCH (owned by Dev B — bench/speed.py + bench/plots.py; do not edit outside these markers) -->
*Pending — produced by `python -m bench.speed` and `python -m bench.plots`.*

![test accuracy](assets/accuracy.png)
![fit time](assets/fit_time.png)
![peak RSS](assets/peak_rss.png)
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
