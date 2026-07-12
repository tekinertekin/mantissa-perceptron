"""Five small classic binary-classification datasets.

Nothing downloads implicitly. ``load(name)`` reads the file from the data
directory; if it is missing it prints (and raises with) the exact curl
command. The only code that touches the network is the explicit helper::

    python -m mantissa_perceptron fetch <name|all>

Data directory: ``./data`` relative to the current working directory, or
the ``MANTISSA_PERCEPTRON_DATA`` environment variable. The directory is
gitignored — datasets are never committed.

Labels are always mapped to {0, 1}; features come back float32 (n, d).

| name          | n    | d  | positive class (1)      | source |
|---------------|------|----|-------------------------|--------|
| iris          |  100 |  4 | Iris-versicolor         | UCI (setosa vs versicolor — linearly separable, the classic perceptron demo) |
| banknote      | 1372 |  4 | forged (label 1)        | UCI 00267 |
| breast_cancer |  569 | 30 | malignant (M)           | UCI WDBC |
| sonar         |  208 | 60 | mine (M)                | UCI connectionist-bench |
| pima          |  768 |  8 | diabetic (label 1)      | UCI original withdrawn; the standard mirror (jbrownlee/Datasets) |

All URLs verified fetchable 2026-07.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, NamedTuple, Tuple

import numpy as np

__all__ = ["DATASETS", "data_dir", "data_path", "fetch_command", "load", "split"]

_DATA_ENV = "MANTISSA_PERCEPTRON_DATA"

_Rows = List[List[str]]
_XY = Tuple[np.ndarray, np.ndarray]


def _xy(feats, labels) -> _XY:
    return (np.asarray(feats, dtype=np.float32),
            np.asarray(labels, dtype=np.int64))


def _parse_iris(rows: _Rows) -> _XY:
    keep = [r for r in rows if r[-1] in ("Iris-setosa", "Iris-versicolor")]
    return _xy([r[:4] for r in keep],
               [1 if r[-1] == "Iris-versicolor" else 0 for r in keep])


def _parse_last_col_int(rows: _Rows) -> _XY:
    return _xy([r[:-1] for r in rows], [int(float(r[-1])) for r in rows])


def _parse_wdbc(rows: _Rows) -> _XY:      # id, diagnosis(M/B), 30 features
    return _xy([r[2:] for r in rows], [1 if r[1] == "M" else 0 for r in rows])


def _parse_sonar(rows: _Rows) -> _XY:     # 60 features, R(rock)/M(mine)
    return _xy([r[:-1] for r in rows], [1 if r[-1] == "M" else 0 for r in rows])


class _Spec(NamedTuple):
    filename: str
    url: str
    parse: Callable[[_Rows], _XY]
    note: str


DATASETS = {
    "iris": _Spec(
        "iris.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        _parse_iris,
        "setosa vs versicolor: linearly separable — the perceptron must converge"),
    "banknote": _Spec(
        "data_banknote_authentication.txt",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        _parse_last_col_int,
        "wavelet features of genuine vs forged banknotes"),
    "breast_cancer": _Spec(
        "wdbc.data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        _parse_wdbc,
        "Wisconsin diagnostic: malignant vs benign, 30 features"),
    "sonar": _Spec(
        "sonar.all-data",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data",
        _parse_sonar,
        "sonar returns, mine vs rock — small, wide, not separable"),
    "pima": _Spec(
        "pima-indians-diabetes.csv",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        _parse_last_col_int,
        "Pima Indians diabetes; UCI withdrew the original, this is the standard mirror"),
}


def data_dir() -> Path:
    return Path(os.environ.get(_DATA_ENV, "data"))


def data_path(name: str) -> Path:
    return data_dir() / DATASETS[name].filename


def fetch_command(name: str) -> str:
    """The exact shell command that downloads dataset `name`."""
    spec = DATASETS[name]
    return f"curl -L --create-dirs -o {data_dir() / spec.filename} {spec.url}"


def load(name: str) -> _XY:
    """Load dataset `name` from the data directory as (X float32, y {0,1}).

    Never downloads: if the file is missing, prints the exact download
    command and raises FileNotFoundError carrying the same message.
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    path = data_path(name)
    if not path.is_file():
        msg = (f"dataset {name!r} not found at {path} — download it with:\n"
               f"  {fetch_command(name)}\n"
               f"or: python -m mantissa_perceptron fetch {name}")
        print(msg, file=sys.stderr)
        raise FileNotFoundError(msg)
    with open(path) as f:
        rows = [line.strip().split(",") for line in f if line.strip()]
    return DATASETS[name].parse(rows)


def split(X, y, test_size: float = 0.25, seed: int = 42, standardize: bool = True):
    """Seeded stratified holdout -> (X_train, X_test, y_train, y_test).

    Stratified per class so small sets (sonar) keep both classes on both
    sides. With ``standardize=True`` features are centred/scaled using
    train-set statistics only — no test leakage.
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    test_idx = []
    for c in np.unique(y):
        idx = rng.permutation(np.flatnonzero(y == c))
        n_test = max(1, int(round(test_size * len(idx))))
        test_idx.append(idx[:n_test])
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[np.concatenate(test_idx)] = True

    X_train, X_test = X[~test_mask], X[test_mask]
    y_train, y_test = y[~test_mask], y[test_mask]
    if standardize:
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd[sd == 0.0] = 1.0
        X_train = (X_train - mu) / sd
        X_test = (X_test - mu) / sd
    return (np.ascontiguousarray(X_train, dtype=np.float32),
            np.ascontiguousarray(X_test, dtype=np.float32),
            y_train, y_test)
