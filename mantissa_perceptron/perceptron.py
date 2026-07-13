"""Binary perceptron on the mantissa C engine.

Two classic training rules, selected by ``rule=``:

- ``"perceptron"`` — Rosenblatt's mistake-driven rule (Rosenblatt, 1958):
  update only on a misclassified sample, ``w += lr * t * x`` with targets
  t in {-1, +1}. Converges in finitely many mistakes on linearly separable
  data (Novikoff, 1962) and is invariant to ``lr`` when weights start at
  zero. The per-sample forward pass runs in mantissa (``tk_linear_forward``);
  the update itself is a single axpy.
- ``"delta"`` — the Widrow-Hoff least-mean-squares rule (Widrow & Hoff,
  1960), a.k.a. ADALINE: SGD on the squared error of the *linear* output.
  The entire step (forward + gradient + weight update) is one call into
  mantissa's ``tk_train_step_f32``. Better behaved on non-separable data,
  where the pure perceptron rule oscillates forever.

mantissa's backward pass defines the STEP/SIGN derivative as 0 (see
mantissa ``include/activations.h``), so a step activation cannot be trained
through ``train_step`` — which is exactly why the perceptron rule is
implemented as the mistake-driven update rather than pretend-gradients.

Deliberately minimal: binary classification only, dense float32 inputs,
no sample weights, no sparse matrices, no multiclass one-vs-rest.
Interface follows scikit-learn conventions (fit / predict / score,
trailing-underscore fitted attributes) without claiming compatibility.
"""
from __future__ import annotations

import numpy as np

from ._engine import engine, load_mantissa

__all__ = ["Perceptron"]


class Perceptron:
    """Single-neuron binary classifier: predict class 1 iff w·x + b > 0.

    Parameters
    ----------
    lr : float
        Learning rate. Irrelevant to the decision boundary under
        ``rule="perceptron"`` (zero-init makes updates scale-invariant);
        matters under ``rule="delta"`` — keep it small on wide inputs
        (LMS stability bound shrinks with feature count).
    epochs : int
        Maximum passes over the data. Training stops early at the end of
        any epoch with zero misclassified training samples.
    rule : {"perceptron", "delta"}
        Training rule (see module docstring).
    shuffle : bool
        Reshuffle sample order each epoch (seeded, reproducible).
    seed : int
        Seed for the shuffle RNG.

    Fitted attributes
    -----------------
    w_ : float32 ndarray, shape (n_features,)
    b_ : float32 ndarray, shape (1,)
    classes_ : ndarray, shape (2,) — sorted original labels; classes_[1]
        is the positive class.
    errors_ : list of int — misclassified training samples per epoch.
        Rosenblatt counts in-epoch as samples are visited (it updates only
        on mistakes, so a zero certifies separation); delta counts with a
        post-epoch pass over the final weights (LMS updates on every
        sample, so only the post-epoch count certifies convergence).
    n_epochs_ : int — epochs actually run.
    converged_ : bool — last epoch had zero training mistakes.
    """

    def __init__(self, lr: float = 0.01, epochs: int = 100,
                 rule: str = "perceptron", shuffle: bool = True, seed: int = 0):
        if rule not in ("perceptron", "delta"):
            raise ValueError(f"rule must be 'perceptron' or 'delta', got {rule!r}")
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.rule = rule
        self.shuffle = bool(shuffle)
        self.seed = int(seed)

    # -- training -----------------------------------------------------------

    def fit(self, X, y):
        X = self._check_X(X)
        y = np.asarray(y).ravel()
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"binary classifier: need exactly 2 classes, got {len(self.classes_)}")
        t = np.where(y == self.classes_[1], 1.0, -1.0).astype(np.float32)

        n, d = X.shape
        self.w_ = np.zeros(d, dtype=np.float32)
        self.b_ = np.zeros(1, dtype=np.float32)
        self.errors_ = []

        tk = engine()
        identity = load_mantissa().IDENTITY
        rng = np.random.default_rng(self.seed)

        # Hoist per-fit invariants out of the epoch loop: attribute lookups,
        # the engine handle, and the activation id are constant across epochs.
        w, b, lr = self.w_, self.b_, self.lr        # w/b mutated in place by C
        shuffle, is_perceptron = self.shuffle, self.rule == "perceptron"
        errors = self.errors_

        # Preferred path: a pre-bound training session (mantissa v0.1.14+).
        # Measured on banknote (1030x4, M4): the per-epoch wrapper calls spent
        # more time re-deriving ctypes pointers for the SAME five buffers than
        # in the C epoch itself (perceptron_epoch ~9.8 us of which ~3 us C).
        # tk.trainer() binds W/X/targets/bias once per fit; per-epoch calls
        # then pass only lr and the visit order. Same C entry points,
        # bit-identical weight trajectories.
        trainer = (tk.trainer(w, X, t, n, 1, d, bias=b)
                   if hasattr(tk, "trainer") else None)

        # Fallback feature-detection for older engines only — inspect.signature
        # costs ~14 us per fit (measured), a fifth of an iris fit, so it must
        # not run on the trainer path.
        if trainer is None:
            import inspect
            have_ord = shuffle and "order" in inspect.signature(tk.train_epoch).parameters
            have_pe = hasattr(tk, "perceptron_epoch")

        # Persistent scratch: the post-epoch forward output (delta mistake
        # count) and the host-side shuffle buffer — allocated once, reused
        # every epoch instead of a fresh allocation per epoch.
        z = np.empty(n, dtype=np.float32) if not is_perceptron else None  # post-epoch count buffer
        Xe = (np.empty((n, d), dtype=np.float32)
              if (trainer is None and not is_perceptron and shuffle) else None)

        if trainer is not None:
            for epoch in range(self.epochs):
                # rng.permutation(n).astype(int32) measured FASTER per epoch
                # than batching all epochs' orders with rng.permuted or an
                # in-place int32 shuffle (6.4 vs 11.3 / 11.9 us at n=1030) —
                # keep the obvious form.
                order = (rng.permutation(n).astype(np.int32)
                         if shuffle else None)
                if is_perceptron:
                    mistakes = trainer.perceptron_epoch(lr, order=order)
                else:
                    trainer.train_epoch(identity, lr, order=order)
                    # errors_ is a POST-epoch count on purpose: LMS updates on
                    # every sample, so only the final weights certify
                    # convergence (Rosenblatt's in-epoch zero is exact).
                    trainer.margins(z)
                    mistakes = int(np.count_nonzero((z + b[0]) * t <= 0.0))
                errors.append(mistakes)
                if mistakes == 0:
                    break
            self.n_epochs_ = len(self.errors_)
            self.converged_ = self.errors_[-1] == 0
            return self

        for epoch in range(self.epochs):
            if is_perceptron:
                order = (rng.permutation(n).astype(np.int32)
                         if shuffle else None)
                if have_pe:
                    # One C call per epoch (tk_perceptron_epoch_f32,
                    # mantissa v0.1.14): same rule, same in-epoch mistake
                    # count, ~314x fewer FFI crossings. Pure float32 — on a
                    # narrow-dtype engine build the old per-sample path
                    # quantized W/x through the storage type each forward,
                    # so decisions can differ at the margin; f32 end-to-end
                    # is the documented semantic now.
                    mistakes = tk.perceptron_epoch(w, X, t, n, 1, d, lr,
                                                   bias=b, order=order)
                else:
                    idx = order if order is not None else range(n)
                    mistakes = 0
                    for i in idx:
                        zi = tk.linear_forward(w, X[i], b, 1, d, identity)[0]
                        if t[i] * zi <= 0.0:                # mistake (0-margin counts)
                            w += lr * t[i] * X[i]
                            b += lr * t[i]
                            mistakes += 1
            else:
                # delta: the WHOLE epoch is one C call (tk_train_epoch_f32) —
                # sequential SGD identical to per-sample train_step calls, but
                # one FFI crossing per epoch instead of one per sample
                # (mantissa v0.1.11; measured ~140x on this exact pattern).
                if have_ord:
                    # order= applies the shuffle inside the single crossing —
                    # no permuted copies of X/targets. errors_ stays a
                    # POST-epoch count on purpose: LMS updates on every
                    # sample (correct ones included), so an in-epoch zero
                    # would not certify that the FINAL weights separate the
                    # data — the convergence early-stop needs the post-epoch
                    # pass. (Rosenblatt differs: it updates only on
                    # mistakes, so its in-epoch zero is exact.)
                    order = (rng.permutation(n).astype(np.int32)
                             if shuffle else None)
                    tk.train_epoch(w, X, t, n, 1, d, identity,
                                   lr, bias=b, order=order)
                    tk.linear_forward(X, w, None, out_dim=n, in_dim=d,
                                      act=identity, out=z)
                    mistakes = int(np.count_nonzero((z + b[0]) * t <= 0.0))
                elif shuffle:
                    order = rng.permutation(n)
                    np.take(X, order, axis=0, out=Xe)       # reuse buffer
                    te = t[order]                           # already contiguous
                    tk.train_epoch(w, Xe, te, n, 1, d, identity, lr, bias=b)
                    tk.linear_forward(X, w, None, out_dim=n, in_dim=d,
                                      act=identity, out=z)
                    mistakes = int(np.count_nonzero((z + b[0]) * t <= 0.0))
                else:
                    tk.train_epoch(w, X, t, n, 1, d, identity, lr, bias=b)
                    tk.linear_forward(X, w, None, out_dim=n, in_dim=d,
                                      act=identity, out=z)
                    mistakes = int(np.count_nonzero((z + b[0]) * t <= 0.0))
            errors.append(mistakes)
            if mistakes == 0:
                break

        self.n_epochs_ = len(self.errors_)
        self.converged_ = self.errors_[-1] == 0
        return self

    # -- inference ----------------------------------------------------------

    def decision_function(self, X):
        """Signed margins w·x + b, shape (n_samples,).

        One threaded C call for the whole batch: X (n×d, row-major) is passed
        as the layer's weight matrix and w as its input, so the GEMV computes
        X @ w — mantissa's row-parallel kernel does the batch for free.
        """
        X = self._check_X(X, n_features=self.w_.shape[0])
        z = np.empty(X.shape[0], dtype=np.float32)
        engine().linear_forward(X, self.w_, None,
                                out_dim=X.shape[0], in_dim=X.shape[1],
                                act=load_mantissa().IDENTITY, out=z)
        return z + self.b_[0]

    def predict(self, X):
        return np.where(self.decision_function(X) > 0.0,
                        self.classes_[1], self.classes_[0])

    def score(self, X, y) -> float:
        """Mean accuracy on (X, y)."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))

    # -- internals ----------------------------------------------------------

    @staticmethod
    def _check_X(X, n_features=None):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_samples, n_features), got ndim={X.ndim}")
        if n_features is not None and X.shape[1] != n_features:
            raise ValueError(f"X has {X.shape[1]} features, model was fit with {n_features}")
        return X
