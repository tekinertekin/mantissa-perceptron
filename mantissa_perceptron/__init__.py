"""mantissa-perceptron: a minimal binary perceptron on the mantissa C engine.

>>> from mantissa_perceptron import Perceptron, datasets
>>> X, y = datasets.load("banknote")
>>> Xtr, Xte, ytr, yte = datasets.split(X, y)
>>> print(Perceptron().fit(Xtr, ytr).score(Xte, yte))
"""
import sys as _sys

from ._engine import MANTISSA_PIP_NAME, engine, load_mantissa
from .perceptron import Perceptron
from . import datasets

__version__ = "0.1.0"
__all__ = ["Perceptron", "datasets", "engine", "load_mantissa", "MANTISSA_PIP_NAME"]

# Surface a missing engine at import time, but stay importable: dataset
# listing/fetching must work without mantissa. Training/inference raise the
# same ImportError on first use.
try:
    load_mantissa()
except ImportError as _e:
    print(_e, file=_sys.stderr)
