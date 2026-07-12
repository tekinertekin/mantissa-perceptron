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

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "bench" / "results"
ASSETS = REPO_ROOT / "assets"

DATASETS = ["iris", "banknote", "breast_cancer", "sonar", "pima"]

# One stable color per contender across every plot (dataviz categorical
# slots, validated CVD-safe; every bar carries a direct value label, which
# is the relief for the two lower-contrast hues).
COLORS = {
    "ours_perceptron": "#2a78d6",   # blue
    "ours_delta": "#1baf7a",        # aqua
    "sklearn": "#eda100",           # yellow
    "numpy": "#008300",             # green
    "purepython": "#4a3aa7",        # violet
    "torch": "#e34948",             # red
}
LABELS = {
    "ours_perceptron": "ours (perceptron)",
    "ours_delta": "ours (delta)",
    "sklearn": "sklearn",
    "numpy": "numpy",
    "purepython": "pure Python",
    "torch": "torch",
}

# Opaque light surface so the PNG reads on GitHub light AND dark themes.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": AXIS,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
    })


def _load(name):
    return json.loads((RESULTS / name).read_text())


def _short_env(env) -> str:
    bits = [env.get("cpu", "?"), f"Python {env.get('python', '?')}"]
    if env.get("mantissa_dtype"):
        bits.append(f"mantissa {env['mantissa_dtype']}")
    if env.get("date"):
        bits.append(env["date"])
    return "  ·  ".join(bits)


def _grouped_bars(ax, contenders, values, log=False, fmt="{:.2f}", unit=""):
    """values[contender][dataset] -> float. One group per dataset."""
    import numpy as np
    n_series = len(contenders)
    x = np.arange(len(DATASETS))
    width = 0.8 / n_series
    all_h = []
    for si, c in enumerate(contenders):
        heights = [values[c].get(d, 0.0) for d in DATASETS]
        offset = (si - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, heights, width, label=LABELS[c],
                      color=COLORS[c], edgecolor=SURFACE, linewidth=0.8, zorder=3)
        for rect, h in zip(bars, heights):
            if h <= 0:
                continue
            all_h.append(h)
            ax.annotate(fmt.format(h), (rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6.5, rotation=90,
                        color=INK2)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    if log:
        ax.set_yscale("log")
        # headroom for the rotated value labels + legend row above tall bars
        ax.set_ylim(min(all_h) / 3.0, max(all_h) * 12.0)
    else:
        ax.margins(y=0.18)


def plot_accuracy(acc, env):
    """Test accuracy per dataset: ours (both rules, from accuracy.json) vs
    sklearn (computed live under the identical fixed protocol)."""
    # ours, from Dev A's accuracy.json
    values = {"ours_perceptron": {}, "ours_delta": {}, "sklearn": {}}
    for row in acc["rows"]:
        key = "ours_perceptron" if row["rule"] == "perceptron" else "ours_delta"
        values[key][row["dataset"]] = row["test_acc"]
    # sklearn, same split/protocol (deterministic; needs the datasets present)
    values["sklearn"] = _sklearn_accuracy(acc["protocol"])
    contenders = [c for c in ("ours_perceptron", "ours_delta", "sklearn")
                  if values[c]]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    _grouped_bars(ax, contenders, values, fmt="{:.3f}")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Test accuracy — ours vs scikit-learn", color=INK,
                 fontsize=13, fontweight="bold", pad=28, loc="left")
    ax.text(0, 1.03, _short_env(env), transform=ax.transAxes, fontsize=8,
            color=INK2, va="bottom")
    ax.legend(loc="lower right", framealpha=0.9, facecolor=SURFACE,
              edgecolor=GRID, fontsize=8, ncol=3)
    _save(fig, "accuracy.png")


def _sklearn_accuracy(protocol) -> dict:
    """Fit sklearn's Perceptron under the fixed protocol on each dataset and
    return {dataset: test_acc}. Returns {} for datasets whose files are
    absent, so the plot degrades to ours-only rather than crashing."""
    try:
        import numpy as np
        from sklearn.linear_model import Perceptron as SkPerceptron
        from mantissa_perceptron import datasets
    except ImportError:
        return {}
    out = {}
    epochs = protocol.get("epochs", 100)
    seed = protocol.get("split_seed", 42)
    test_size = protocol.get("test_size", 0.25)
    for name in DATASETS:
        try:
            X, y = datasets.load(name)
        except FileNotFoundError:
            continue
        Xtr, Xte, ytr, yte = datasets.split(X, y, test_size=test_size,
                                             seed=seed, standardize=True)
        # numpy 2.x + Apple Accelerate raises spurious FPE flags from the
        # matmul kernel on finite inputs; results are unaffected.
        with np.errstate(all="ignore"):
            clf = SkPerceptron(max_iter=epochs, tol=None, shuffle=True,
                               random_state=0).fit(Xtr, ytr)
            out[name] = float(clf.score(Xte, yte))
    return out


def plot_fit_time(speed, env):
    contenders = _contenders(speed["fit_ms"])
    values = {c: {d: speed["fit_ms"][d][c]["median"] for d in DATASETS
                  if c in speed["fit_ms"][d]} for c in contenders}
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    _grouped_bars(ax, contenders, values, log=True, fmt="{:.2f}")
    ax.set_ylabel("median fit() time — ms (log scale)")
    ax.set_title("Training time — median of 15 interleaved fits", color=INK,
                 fontsize=13, fontweight="bold", pad=28, loc="left")
    ax.text(0, 1.03, _short_env(env), transform=ax.transAxes, fontsize=8,
            color=INK2, va="bottom")
    ax.legend(loc="upper left", framealpha=0.9, facecolor=SURFACE,
              edgecolor=GRID, fontsize=8, ncol=len(contenders))
    _save(fig, "fit_time.png")


def plot_peak_rss(speed, env):
    """Peak RSS per contender (import + one fit, whole-process peak). RSS is
    essentially flat across datasets, so we show one bar per contender on
    the largest dataset (banknote)."""
    ds = "banknote" if "banknote" in speed["peak_rss_mb"] else DATASETS[0]
    rss = speed["peak_rss_mb"][ds]
    contenders = _contenders(speed["fit_ms"])
    contenders = [c for c in contenders if c in rss]
    heights = [rss[c] for c in contenders]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=150)
    xs = range(len(contenders))
    bars = ax.bar(xs, heights, width=0.62,
                  color=[COLORS[c] for c in contenders],
                  edgecolor=SURFACE, linewidth=0.8, zorder=3)
    for rect, h in zip(bars, heights):
        ax.annotate(f"{h:.0f} MB", (rect.get_x() + rect.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([LABELS[c] for c in contenders], rotation=15, ha="right")
    ax.set_ylabel("peak RSS — MB")
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.margins(y=0.16)
    ax.set_title(f"Peak memory — import + fit ({ds}, fresh process)",
                 color=INK, fontsize=13, fontweight="bold", pad=28, loc="left")
    ax.text(0, 1.03, _short_env(env), transform=ax.transAxes, fontsize=8,
            color=INK2, va="bottom")
    _save(fig, "peak_rss.png")


def _contenders(fit_ms):
    """Contender order from the data, ours first."""
    seen = []
    for d in DATASETS:
        for c in fit_ms.get(d, {}):
            if c not in seen:
                seen.append(c)
    order = ["ours_perceptron", "ours_delta", "sklearn", "numpy",
             "purepython", "torch"]
    return [c for c in order if c in seen] + [c for c in seen if c not in order]


def _save(fig, name):
    ASSETS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    # metadata=Software:None -> byte-stable across runs (no version/timestamp).
    fig.savefig(ASSETS / name, dpi=150, metadata={"Software": None})
    plt.close(fig)
    print(f"wrote assets/{name}")


def main() -> int:
    _style()
    speed = _load("speed.json")
    try:
        acc = _load("accuracy.json")
    except FileNotFoundError:
        acc = None
    env = speed.get("env", {})

    if acc is not None:
        plot_accuracy(acc, env)
    else:
        print("accuracy.json absent — skipping accuracy.png")
    plot_fit_time(speed, env)
    plot_peak_rss(speed, env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
