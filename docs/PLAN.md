# Build plan — mantissa-perceptron

Architect scaffold is complete and working (package, datasets, model,
tests). Two developer packages remain. File ownership is exact; the two
packages share **no files** except README.md, where each dev edits only
between their own HTML markers (`BEGIN:ACCURACY` / `BEGIN:BENCH`).

## Architecture decisions (fixed — do not relitigate)

1. **Engine loading** (`mantissa_perceptron/_engine.py`): try
   `import mantissa` (pip), else the sibling checkout
   `../mantissa/python`. Pip name is the `MANTISSA_PIP_NAME` placeholder —
   only the packaging agent changes it (there and in pyproject.toml).
2. **Two rules, honestly named.** mantissa defines the STEP/SIGN gradient
   as 0 (`include/activations.h`), so a step unit cannot train through
   `tk_train_step_f32`. Therefore: `rule="perceptron"` = mistake-driven
   Rosenblatt update (mantissa forward + one axpy), `rule="delta"` =
   ADALINE, full step in C. Labels {0,1} externally, targets ±1
   internally, threshold at 0.
3. **Batch inference trick**: `decision_function` passes X (n×d) as the
   layer's weight matrix and w as the input, so the whole batch is one
   threaded GEMV call. Do not replace with a per-row loop.
4. **No implicit downloads.** `datasets.load` raises with the exact curl
   command; only `python -m mantissa_perceptron fetch` hits the network.
   `data/` and `bench/results/` are gitignored; `assets/*.png` are
   committed.
5. **Measurement house rules**: fixed seeds, interleaved repeats, medians,
   subprocess-isolated RSS, versions recorded. Never assume.
6. **Learnings flow upstream**: anything that would make mantissa itself
   faster/smaller goes to `docs/LEARNINGS.md` as a measurable idea.

## Dev A — datasets, model validation, accuracy table

**Files owned**: `bench/accuracy.py`, `tests/` (may extend),
`mantissa_perceptron/datasets.py` + `perceptron.py` (bug fixes and lr
tuning only — no API changes without architect sign-off), README.md
*inside the ACCURACY markers only*.

Tasks:
1. `python -m mantissa_perceptron fetch all`; verify each loader's shapes
   and class balance against the header table in `datasets.py` (iris 100×4,
   banknote 1372×4, wdbc 569×30, sonar 208×60, pima 768×8). Add a
   `tests/test_datasets.py` that validates parsing (skipped when files are
   absent, so CI without data still passes).
2. Run the full test suite against the built sibling engine; fix what
   fails.
3. Implement `bench/accuracy.py` exactly to the protocol and JSON schema
   in its docstring (fixed split seed 42, epochs 100; delta-rule lr tuned
   on train only over {0.001, 0.003, 0.01, 0.03}).
4. Paste the emitted markdown table into README between the ACCURACY
   markers, including epochs-run and converged columns.

Acceptance criteria:
- `pytest` green with the engine present; graceful skip without it.
- `python -m bench.accuracy` is deterministic (two runs, identical JSON).
- iris test accuracy = 1.0 with `rule="perceptron"` and `converged_ =
  True` (it is linearly separable — anything less is a bug, not a result).
- banknote ≥ 0.97, breast_cancer ≥ 0.93 test accuracy (best of the two
  rules; architect's untuned sanity run already hit 0.982 / 0.951);
  sonar/pima reported as measured, no cherry-picking.
- `bench/results/accuracy.json` matches the documented schema.

## Dev B — competitor benchmark, plots, README results

**Files owned**: `bench/speed.py`, `bench/plots.py`, `assets/*.png`,
README.md *inside the BENCH markers only*. Consumes Dev A's
`bench/results/accuracy.json`; does not modify package code. If a package
bug blocks benchmarking, report it — do not fix in place.

Tasks:
1. Implement `bench/speed.py` to its docstring spec. Contenders:
   ours (both rules), `sklearn.linear_model.Perceptron` (max_iter=100,
   tol=None, random_state=0), hand-rolled numpy perceptron, pure-Python
   perceptron; torch optional-if-importable. Interleaved repeats (R=15),
   medians, `perf_counter`; peak RSS per contender in a fresh subprocess
   via `getrusage` (ru_maxrss: bytes on macOS, KiB on Linux — normalize).
2. Implement `bench/plots.py`: accuracy.png, fit_time.png (log ms),
   peak_rss.png into `assets/`, 150 dpi, one stable color per contender.
3. Fill the README BENCH section: a results table (fit ms + RSS MB per
   contender on the largest dataset, banknote), the three plots, and the
   environment line (CPU, Python, versions, mantissa dtype, threads).
   State fairness caveats explicitly (sklearn epoch does more work;
   ctypes per-call overhead — see LEARNINGS).
4. Record any engine-level findings in `docs/LEARNINGS.md` with numbers.

Acceptance criteria:
- `python -m bench.speed && python -m bench.plots` runs clean from repo
  root on a machine with the `[bench]` extra installed.
- speed.json matches the documented schema; env block complete.
- Interleaving verified in code review (round-robin, not blocked runs).
- Plots regenerate byte-stable modulo timestamps; committed under assets/.
- No dataset files, no bench/results/*.json committed.

## Sequencing

Dev A and Dev B can start in parallel (Dev B can develop against
synthetic data + a hand-written accuracy.json matching the schema);
Dev B's final README/plots run happens after Dev A's accuracy.json lands.
