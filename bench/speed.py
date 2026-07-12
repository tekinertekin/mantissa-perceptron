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

# TODO(Dev B): implement per docs/PLAN.md §Dev B. Keep the schema above exact.


def main() -> int:
    raise NotImplementedError("Dev B: implement per docs/PLAN.md §Dev B")


if __name__ == "__main__":
    raise SystemExit(main())
