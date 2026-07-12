"""Dataset parsing/split tests. Each dataset skips individually when its
file is absent, so CI without a populated data/ still passes. Fetch with:
``python -m mantissa_perceptron fetch all``. These tests need no engine."""
import numpy as np
import pytest

from mantissa_perceptron import datasets

# (name, n, d) — the header table in datasets.py; the parser must reproduce it.
EXPECTED = [
    ("iris", 100, 4),
    ("banknote", 1372, 4),
    ("breast_cancer", 569, 30),
    ("sonar", 208, 60),
    ("pima", 768, 8),
]


def _load_or_skip(name):
    if not datasets.data_path(name).is_file():
        pytest.skip(f"{name} not fetched ({datasets.fetch_command(name)})")
    return datasets.load(name)


def test_registry_matches_expected_names():
    assert set(datasets.DATASETS) == {name for name, _, _ in EXPECTED}


@pytest.mark.parametrize("name, n, d", EXPECTED)
def test_shapes_dtypes_and_labels(name, n, d):
    X, y = _load_or_skip(name)
    assert X.shape == (n, d)
    assert X.dtype == np.float32
    assert set(np.unique(y).tolist()) == {0, 1}   # always mapped to {0, 1}
    assert len(y) == n
    assert np.isfinite(X).all()
    assert (y == 1).sum() > 0 and (y == 0).sum() > 0   # both classes present


@pytest.mark.parametrize("name, n, d", EXPECTED)
def test_split_is_stratified_seeded_and_leak_free(name, n, d):
    X, y = _load_or_skip(name)
    Xtr, Xte, ytr, yte = datasets.split(X, y, test_size=0.25, seed=42)

    # partition: no lost/duplicated rows, correct feature width
    assert len(ytr) + len(yte) == n
    assert Xtr.shape[1] == d and Xte.shape[1] == d
    assert Xtr.dtype == np.float32 and Xte.dtype == np.float32

    # stratified: both classes land on both sides
    for part in (ytr, yte):
        assert set(np.unique(part).tolist()) == {0, 1}

    # standardized on TRAIN statistics only (no test leakage): train mean ~0
    assert np.allclose(Xtr.mean(axis=0), 0.0, atol=1e-4)

    # seeded: identical split on a repeat call
    Xtr2, Xte2, ytr2, yte2 = datasets.split(X, y, test_size=0.25, seed=42)
    assert np.array_equal(Xtr, Xtr2) and np.array_equal(yte, yte2)


def test_unknown_dataset_raises():
    with pytest.raises(KeyError):
        datasets.load("not_a_dataset")
