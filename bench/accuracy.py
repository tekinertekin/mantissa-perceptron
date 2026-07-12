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

# TODO(Dev A): implement per docs/PLAN.md §Dev A. Keep the schema above exact.


def main() -> int:
    raise NotImplementedError("Dev A: implement per docs/PLAN.md §Dev A")


if __name__ == "__main__":
    raise SystemExit(main())
