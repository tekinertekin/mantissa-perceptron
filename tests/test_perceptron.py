"""Smoke tests for the Perceptron. Skipped entirely if the mantissa engine
is not available (pip install or sibling ../mantissa checkout with a built
dist/libmantissa.*)."""
import numpy as np
import pytest

from mantissa_perceptron import Perceptron

try:
    from mantissa_perceptron import engine
    engine()
except Exception as e:                       # ImportError or missing dylib
    pytest.skip(f"mantissa engine unavailable: {e}", allow_module_level=True)


def separable_blobs(n=200, d=5, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d)).astype(np.float32)
    w = rng.standard_normal(d).astype(np.float32)
    y = (X @ w + 0.5 > 0).astype(np.int64)
    X += 0.3 * np.sign(X @ w + 0.5)[:, None] * w / np.linalg.norm(w)  # widen margin
    return X, y


@pytest.mark.parametrize("rule", ["perceptron", "delta"])
def test_converges_on_separable_data(rule):
    X, y = separable_blobs()
    clf = Perceptron(rule=rule, epochs=200).fit(X, y)
    assert clf.score(X, y) == 1.0
    if rule == "perceptron":
        assert clf.converged_                     # Novikoff guarantee


def test_perceptron_rule_is_lr_invariant():
    X, y = separable_blobs()
    p1 = Perceptron(rule="perceptron", lr=0.01, shuffle=False).fit(X, y)
    p2 = Perceptron(rule="perceptron", lr=10.0, shuffle=False).fit(X, y)
    assert np.array_equal(p1.predict(X), p2.predict(X))


def test_predict_shapes_and_labels():
    X, y = separable_blobs()
    y_lab = np.where(y == 1, 7, -3)              # arbitrary label values survive
    clf = Perceptron().fit(X, y_lab)
    pred = clf.predict(X[:10])
    assert pred.shape == (10,)
    assert set(np.unique(pred)) <= {-3, 7}
    assert clf.decision_function(X).shape == (len(X),)


def test_rejects_bad_input():
    X, y = separable_blobs()
    with pytest.raises(ValueError):
        Perceptron(rule="nonsense")
    with pytest.raises(ValueError):
        Perceptron().fit(X, np.zeros(len(X)))    # one class
    with pytest.raises(ValueError):
        Perceptron().fit(X, y).predict(X[:, :2])  # wrong feature count
