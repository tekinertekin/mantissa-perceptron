"""Accuracy benchmark — Dev A work package (docs/PLAN.md §Dev A).

Reports test accuracy of our Perceptron on every dataset, both rules,
under the fixed protocol. This file is owned by Dev A.

Protocol (fixed — do not vary per dataset except where marked):
  - datasets.split(X, y, test_size=0.25, seed=42, standardize=True)
  - rule="perceptron": lr=1.0 (boundary is lr-invariant), epochs=100, seed=0
  - rule="delta": epochs=100, seed=0, lr tuned per dataset over
    {0.001, 0.003, 0.01, 0.03} on the TRAIN set only (report the chosen lr)
  - accuracy = Perceptron.score on the held-out test set

Outputs:
  1. bench/results/accuracy.json  (schema below — plots.py consumes it)
  2. a GitHub-markdown table on stdout, pasted by Dev A into README.md
     between the ACCURACY markers only.

accuracy.json schema:
{
  "protocol": {"test_size": 0.25, "split_seed": 42, "epochs": 100},
  "rows": [
    {"dataset": "iris", "n": 100, "d": 4,
     "rule": "perceptron", "lr": 1.0,
     "train_acc": 1.0, "test_acc": 1.0,
     "epochs_run": 3, "converged": true},
    ...
  ]
}

Run from the repo root:  python -m bench.accuracy
"""
from __future__ import annotations

import json
from pathlib import Path

from mantissa_perceptron import Perceptron, datasets

# --- fixed protocol constants ------------------------------------------------
TEST_SIZE = 0.25
SPLIT_SEED = 42
EPOCHS = 100
MODEL_SEED = 0
PERCEPTRON_LR = 1.0                       # boundary is lr-invariant at zero-init
DELTA_LR_GRID = (0.001, 0.003, 0.01, 0.03)

_RESULTS = Path(__file__).resolve().parent / "results" / "accuracy.json"


def _fit(rule: str, lr: float, Xtr, ytr) -> Perceptron:
    return Perceptron(rule=rule, lr=lr, epochs=EPOCHS, seed=MODEL_SEED).fit(Xtr, ytr)


def _tune_delta_lr(Xtr, ytr) -> float:
    """Pick the delta-rule lr with the best TRAIN accuracy (no test leakage).

    Ties keep the smaller lr — LMS is more stable there, and the choice is
    deterministic. Grid walked ascending with strict improvement.
    """
    best_lr, best_train = DELTA_LR_GRID[0], -1.0
    for lr in DELTA_LR_GRID:
        train_acc = _fit("delta", lr, Xtr, ytr).score(Xtr, ytr)
        if train_acc > best_train:
            best_lr, best_train = lr, train_acc
    return best_lr


def _row(name: str, X, rule: str, lr: float, Xtr, Xte, ytr, yte) -> dict:
    clf = _fit(rule, lr, Xtr, ytr)
    return {
        "dataset": name,
        "n": int(X.shape[0]),
        "d": int(X.shape[1]),
        "rule": rule,
        "lr": float(lr),
        "train_acc": clf.score(Xtr, ytr),
        "test_acc": clf.score(Xte, yte),
        "epochs_run": int(clf.n_epochs_),
        "converged": bool(clf.converged_),
    }


def evaluate() -> dict:
    rows = []
    for name in datasets.DATASETS:
        X, y = datasets.load(name)
        Xtr, Xte, ytr, yte = datasets.split(
            X, y, test_size=TEST_SIZE, seed=SPLIT_SEED, standardize=True)
        rows.append(_row(name, X, "perceptron", PERCEPTRON_LR, Xtr, Xte, ytr, yte))
        delta_lr = _tune_delta_lr(Xtr, ytr)
        rows.append(_row(name, X, "delta", delta_lr, Xtr, Xte, ytr, yte))
    return {
        "protocol": {"test_size": TEST_SIZE, "split_seed": SPLIT_SEED, "epochs": EPOCHS},
        "rows": rows,
    }


def _markdown(result: dict) -> str:
    head = ("| dataset | n | d | rule | lr | train acc | test acc "
            "| epochs run | converged |")
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [head, sep]
    for r in result["rows"]:
        lines.append(
            f"| {r['dataset']} | {r['n']} | {r['d']} | {r['rule']} "
            f"| {r['lr']:g} | {r['train_acc']:.3f} | {r['test_acc']:.3f} "
            f"| {r['epochs_run']} | {'yes' if r['converged'] else 'no'} |")
    return "\n".join(lines)


def main() -> int:
    result = evaluate()
    _RESULTS.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(_markdown(result))
    print(f"\nwrote {_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
