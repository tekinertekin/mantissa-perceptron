"""Benchmark plots — Dev B work package (docs/PLAN.md §Dev B).

Reads bench/results/accuracy.json and bench/results/speed.json, writes
committed PNGs into assets/ (matplotlib, no seaborn, no network):

  assets/accuracy.png   grouped bars: test accuracy per dataset, ours
                        (both rules) vs sklearn
  assets/fit_time.png   grouped bars, log-scale ms: median fit() time per
                        dataset per contender
  assets/peak_rss.png   grouped bars: peak RSS (MB) per contender
                        (import + fit; whole-process peak)

Conventions: one consistent color per contender across all plots; value
labels on bars; title carries the machine + date from speed.json "env";
150 dpi; readable in both GitHub light and dark themes.

Run from the repo root:  python -m bench.plots
"""
from __future__ import annotations

# TODO(Dev B): implement per docs/PLAN.md §Dev B.


def main() -> int:
    raise NotImplementedError("Dev B: implement per docs/PLAN.md §Dev B")


if __name__ == "__main__":
    raise SystemExit(main())
