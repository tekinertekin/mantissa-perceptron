"""Speed + peak-RSS benchmark against famous perceptrons — Dev B package
(docs/PLAN.md §Dev B). This file is owned by Dev B.

Contenders (same data, same split, same epochs cap, seeded):
  ours_perceptron   mantissa_perceptron.Perceptron(rule="perceptron")
  ours_delta        mantissa_perceptron.Perceptron(rule="delta")
  sklearn           sklearn.linear_model.Perceptron(max_iter=EPOCHS, tol=None,
                    shuffle=True, random_state=0) — the canonical library
                    implementation (Cython SGD underneath)
  numpy             hand-rolled vectorized-per-sample numpy perceptron
                    (the honest "what you'd write yourself" baseline)
  purepython        list-of-floats perceptron, no numpy — the floor
  torch (optional)  torch.nn.Linear(d, 1) + manual perceptron rule; include
                    only if torch imports, never a dependency

Methodology — this house measures, never assumes:
  - TIME: fit() wall time only (data already loaded and split; exclude
    imports). R=15 repeats, INTERLEAVED round-robin (A,B,C,... x15) so
    thermal/background drift hits every contender equally. Report the
    median; keep raw samples in the JSON. time.perf_counter().
  - Also time batch predict() over the test set (median of 100 calls).
  - PEAK RSS: one (contender, dataset) per fresh subprocess; the child
    imports its own library, runs fit once, then reports
    resource.getrusage(RUSAGE_SELF).ru_maxrss on stdout. Import cost is
    deliberately included: it is what a user pays. Normalize units —
    ru_maxrss is BYTES on macOS, KiB on Linux.
  - Fairness: identical epochs cap and shuffle seeds where the API allows;
    record library versions, CPU, MANTISSA_THREADS. Note in the README
    that sklearn's Perceptron does strictly more work per epoch than the
    naive rule; we disable its early stopping (tol=None) to equalize.

Outputs: bench/results/speed.json (schema below; plots.py consumes it).

speed.json schema:
{
  "env": {"cpu": "...", "python": "3.x", "numpy": "...", "sklearn": "...",
          "mantissa_dtype": "bfloat16", "threads": "...", "date": "..."},
  "protocol": {"epochs": 100, "repeats": 15, "test_size": 0.25, "split_seed": 42},
  "fit_ms":     {"<dataset>": {"<contender>": {"median": 1.23, "samples": [...]}}},
  "predict_ms": {"<dataset>": {"<contender>": {"median": 0.01, "samples": [...]}}},
  "peak_rss_mb": {"<dataset>": {"<contender>": 34.5}}
}

Run from the repo root:  python -m bench.speed
(the RSS worker re-invokes:  python -m bench.speed --worker <contender> <dataset>)
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

# numpy 2.x on Apple Accelerate emits spurious FPE RuntimeWarnings from the
# BLAS matmul kernel even on finite inputs (verified: contender weights stay
# bounded). They fire from both our numpy baseline and sklearn's internals.
warnings.filterwarnings("ignore", message=".*encountered in matmul",
                        category=RuntimeWarning)

# --- protocol (fixed) ------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = REPO_ROOT / "bench" / "results" / "speed.json"

EPOCHS = 100
REPEATS = 15
PREDICT_CALLS = 100
TEST_SIZE = 0.25
SPLIT_SEED = 42
LR = 0.01
DATASETS = ["iris", "banknote", "breast_cancer", "sonar", "pima"]


# --- baseline contenders (no third-party training code) --------------------

class NumpyPerceptron:
    """Mistake-driven Rosenblatt rule, hand-rolled in numpy — the honest
    'what you'd write yourself' baseline. Same rule as ours_perceptron:
    per-sample forward, update only on a mistake, early-stop at zero."""

    def __init__(self, epochs: int = EPOCHS, lr: float = LR, seed: int = 0):
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, X, y):
        import numpy as np
        n, d = X.shape
        self.classes_ = np.unique(y)
        t = np.where(y == self.classes_[1], 1.0, -1.0)
        w = np.zeros(d, dtype=np.float64)
        b = 0.0
        rng = np.random.default_rng(self.seed)
        # numpy 2.x on Apple Accelerate raises spurious FPE flags from the
        # SIMD matmul kernel even on finite inputs; weights stay bounded.
        with np.errstate(all="ignore"):
            for _ in range(self.epochs):
                mistakes = 0
                for i in rng.permutation(n):
                    if t[i] * (X[i] @ w + b) <= 0.0:
                        w += self.lr * t[i] * X[i]
                        b += self.lr * t[i]
                        mistakes += 1
                if mistakes == 0:
                    break
        self.w_, self.b_ = w, b
        return self

    def predict(self, X):
        import numpy as np
        with np.errstate(all="ignore"):
            z = X @ self.w_ + self.b_
        return np.where(z > 0.0, self.classes_[1], self.classes_[0])


class PurePyPerceptron:
    """Same rule, list-of-floats, no numpy — the interpreted floor."""

    def __init__(self, epochs: int = EPOCHS, lr: float = LR, seed: int = 0):
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, X, y):
        import random
        rng = random.Random(self.seed)
        n, d = len(X), len(X[0])
        self.classes_ = sorted(set(y))
        pos = self.classes_[1]
        t = [1.0 if yi == pos else -1.0 for yi in y]
        w = [0.0] * d
        b = 0.0
        order = list(range(n))
        for _ in range(self.epochs):
            rng.shuffle(order)
            mistakes = 0
            for i in order:
                xi = X[i]
                z = b
                for k in range(d):
                    z += w[k] * xi[k]
                if t[i] * z <= 0.0:
                    step = self.lr * t[i]
                    for k in range(d):
                        w[k] += step * xi[k]
                    b += step
                    mistakes += 1
            if mistakes == 0:
                break
        self.w_, self.b_ = w, b
        return self

    def predict(self, X):
        w, b, c = self.w_, self.b_, self.classes_
        out = []
        for row in X:
            z = b
            for k in range(len(w)):
                z += w[k] * row[k]
            out.append(c[1] if z > 0.0 else c[0])
        return out


class TorchPerceptron:
    """torch.nn.Linear(d, 1) with the manual mistake-driven rule. Only used
    when torch imports; never a dependency."""

    def __init__(self, epochs: int = EPOCHS, lr: float = LR, seed: int = 0):
        self.epochs, self.lr, self.seed = epochs, lr, seed

    def fit(self, X, y):
        import torch
        torch.manual_seed(self.seed)
        import numpy as np
        classes = np.unique(y)
        self.classes_ = classes
        Xt = torch.as_tensor(X, dtype=torch.float32)
        t = torch.as_tensor(np.where(y == classes[1], 1.0, -1.0), dtype=torch.float32)
        n, d = Xt.shape
        w = torch.zeros(d, dtype=torch.float32)
        b = torch.zeros(1, dtype=torch.float32)
        g = torch.Generator().manual_seed(self.seed)
        with torch.no_grad():
            for _ in range(self.epochs):
                mistakes = 0
                for i in torch.randperm(n, generator=g):
                    if (t[i] * (Xt[i] @ w + b)).item() <= 0.0:
                        w += self.lr * t[i] * Xt[i]
                        b += self.lr * t[i]
                        mistakes += 1
                if mistakes == 0:
                    break
        self.w_, self.b_ = w, b
        return self

    def predict(self, X):
        import torch
        import numpy as np
        Xt = torch.as_tensor(X, dtype=torch.float32)
        z = (Xt @ self.w_ + self.b_).numpy()
        return np.where(z > 0.0, self.classes_[1], self.classes_[0])


# --- contender registry ----------------------------------------------------
# Each entry: build() -> fresh estimator; prep_X/prep_y map the (numpy) data
# into the contender's native form ONCE, outside the timed region, so fit()
# measures training only. Heavy imports live inside build()/prep so an RSS
# worker only pays for the library it actually uses.

def _to_f32(X):
    import numpy as np
    return np.ascontiguousarray(X, dtype=np.float32)


def _to_f64(X):
    import numpy as np
    return np.ascontiguousarray(X, dtype=np.float64)


def _to_lists_X(X):
    return [[float(v) for v in row] for row in X]


def _to_list_y(y):
    return [int(v) for v in y]


def _identity(v):
    return v


def _build_ours(rule):
    def build():
        from mantissa_perceptron import Perceptron
        return Perceptron(rule=rule, epochs=EPOCHS, lr=LR, seed=0, shuffle=True)
    return build


def _build_sklearn():
    from sklearn.linear_model import Perceptron as SkPerceptron
    return SkPerceptron(max_iter=EPOCHS, tol=None, shuffle=True, random_state=0)


def _contenders():
    """Ordered list of (name, build, prep_X, prep_y). Torch appended only if
    importable."""
    reg = [
        ("ours_perceptron", _build_ours("perceptron"), _to_f32, _identity),
        ("ours_delta", _build_ours("delta"), _to_f32, _identity),
        ("sklearn", _build_sklearn, _to_f32, _identity),
        ("numpy", lambda: NumpyPerceptron(), _to_f64, _identity),
        ("purepython", lambda: PurePyPerceptron(), _to_lists_X, _to_list_y),
    ]
    try:
        import torch  # noqa: F401
        reg.append(("torch", lambda: TorchPerceptron(), _to_f32, _identity))
    except ImportError:
        pass
    return reg


# --- data ------------------------------------------------------------------

def _load_split(dataset):
    from mantissa_perceptron import datasets
    X, y = datasets.load(dataset)
    return datasets.split(X, y, test_size=TEST_SIZE, seed=SPLIT_SEED,
                          standardize=True)


# --- RSS worker ------------------------------------------------------------

def _rss_mb() -> float:
    import resource
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss: bytes on macOS, KiB on Linux.
    if sys.platform == "darwin":
        return maxrss / (1024.0 * 1024.0)
    return maxrss / 1024.0


def _run_worker(contender: str, dataset: str) -> int:
    """Fresh subprocess: import the contender's library, fit once, print peak
    RSS in MB. Import cost is included on purpose — it is what a user pays."""
    spec = {name: (build, px, py) for name, build, px, py in _contenders()}.get(contender)
    if spec is None:
        print(f"unknown contender {contender!r}", file=sys.stderr)
        return 2
    build, prep_X, prep_y = spec
    Xtr, _Xte, ytr, _yte = _load_split(dataset)
    est = build()
    est.fit(prep_X(Xtr), prep_y(ytr))
    print(f"{_rss_mb():.4f}")
    return 0


def _measure_rss(contender: str, dataset: str) -> float:
    proc = subprocess.run(
        [sys.executable, "-m", "bench.speed", "--worker", contender, dataset],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"RSS worker failed for {contender}/{dataset}:\n{proc.stderr}")
    return float(proc.stdout.strip().splitlines()[-1])


# --- timing ----------------------------------------------------------------

def _time_dataset(dataset, contenders):
    """Interleaved round-robin timing for one dataset. Returns
    (fit_ms, predict_ms) dicts keyed by contender."""
    Xtr, Xte, ytr, yte = _load_split(dataset)
    # Prepare native data once per contender (not timed).
    data = {}
    for name, build, prep_X, prep_y in contenders:
        data[name] = (prep_X(Xtr), prep_X(Xte), prep_y(ytr))

    fit_samples = {name: [] for name, *_ in contenders}
    # FIT: outer loop repeats, inner loop contenders -> true round-robin.
    for _ in range(REPEATS):
        for name, build, prep_X, prep_y in contenders:
            Xtr_n, _Xte_n, ytr_n = data[name]
            est = build()
            t0 = time.perf_counter()
            est.fit(Xtr_n, ytr_n)
            fit_samples[name].append((time.perf_counter() - t0) * 1000.0)

    # PREDICT: fit one model per contender, then round-robin batch predict.
    fitted = {}
    for name, build, prep_X, prep_y in contenders:
        Xtr_n, _Xte_n, ytr_n = data[name]
        fitted[name] = build().fit(Xtr_n, ytr_n)
    predict_samples = {name: [] for name, *_ in contenders}
    for _ in range(PREDICT_CALLS):
        for name, build, prep_X, prep_y in contenders:
            _Xtr_n, Xte_n, _ytr_n = data[name]
            model = fitted[name]
            t0 = time.perf_counter()
            model.predict(Xte_n)
            predict_samples[name].append((time.perf_counter() - t0) * 1000.0)

    fit_ms = {n: {"median": median(s), "samples": s} for n, s in fit_samples.items()}
    predict_ms = {n: {"median": median(s), "samples": s}
                  for n, s in predict_samples.items()}
    return fit_ms, predict_ms


# --- environment -----------------------------------------------------------

def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or "unknown"


def _env_block(contenders) -> dict:
    import numpy as np
    env = {
        "cpu": _cpu_name(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "threads": os.environ.get("MANTISSA_THREADS", f"default({os.cpu_count()})"),
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    }
    try:
        import sklearn
        env["sklearn"] = sklearn.__version__
    except ImportError:
        env["sklearn"] = None
    try:
        from mantissa_perceptron import engine
        env["mantissa_dtype"] = engine().dtype
    except Exception:
        env["mantissa_dtype"] = None
    if any(n == "torch" for n, *_ in contenders):
        import torch
        env["torch"] = torch.__version__
    return env


# --- entrypoint ------------------------------------------------------------

def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--worker":
        return _run_worker(argv[1], argv[2])

    contenders = _contenders()
    names = [n for n, *_ in contenders]
    print(f"contenders: {', '.join(names)}")

    fit_ms, predict_ms, peak_rss_mb = {}, {}, {}
    for dataset in DATASETS:
        print(f"\n[{dataset}] timing (R={REPEATS}, interleaved) ...")
        f, p = _time_dataset(dataset, contenders)
        fit_ms[dataset] = f
        predict_ms[dataset] = p
        for name in names:
            print(f"  {name:16s} fit {f[name]['median']:9.3f} ms   "
                  f"predict {p[name]['median']:8.4f} ms")
        print(f"[{dataset}] peak RSS (fresh subprocess each) ...")
        peak_rss_mb[dataset] = {}
        for name in names:
            mb = _measure_rss(name, dataset)
            peak_rss_mb[dataset][name] = round(mb, 4)
            print(f"  {name:16s} {mb:7.1f} MB")

    out = {
        "env": _env_block(contenders),
        "protocol": {"epochs": EPOCHS, "repeats": REPEATS,
                     "test_size": TEST_SIZE, "split_seed": SPLIT_SEED},
        "fit_ms": fit_ms,
        "predict_ms": predict_ms,
        "peak_rss_mb": peak_rss_mb,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {RESULTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
