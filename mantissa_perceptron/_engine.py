"""Locate and load the mantissa engine.

Resolution order:
1. an installed ``mantissa`` package (pip),
2. the sibling development checkout ``../mantissa/python`` (the repo layout
   used before the PyPI release).
"""
from __future__ import annotations

import sys
from pathlib import Path

# PLACEHOLDER: the PyPI distribution name for mantissa is being finalized by
# the packaging agent. Update this one constant (and pyproject.toml) when it
# lands; nothing else references the pip name.
MANTISSA_PIP_NAME = "mantissa-nn"

# Sibling checkout when this repo lives next to mantissa/:
#   <parent>/perceptron/mantissa_perceptron/_engine.py  ->  <parent>/mantissa/python
_DEV_PYTHON_DIR = Path(__file__).resolve().parents[2] / "mantissa" / "python"

_tk = None  # process-wide engine singleton (one dylib load)


def load_mantissa():
    """Import and return the ``mantissa`` module.

    Raises ImportError with the exact install command if it cannot be found.
    """
    try:
        import mantissa
        return mantissa
    except ImportError:
        pass
    # The checkout ships either a module (mantissa.py) or a package
    # (mantissa/__init__.py) depending on the packaging work — accept both.
    if (_DEV_PYTHON_DIR / "mantissa.py").is_file() \
            or (_DEV_PYTHON_DIR / "mantissa" / "__init__.py").is_file():
        p = str(_DEV_PYTHON_DIR)
        if p not in sys.path:
            sys.path.insert(0, p)
        import mantissa
        return mantissa
    raise ImportError(
        f"mantissa is not installed — run: pip install {MANTISSA_PIP_NAME}\n"
        f"(dev fallback also checked: {_DEV_PYTHON_DIR})"
    )


def engine():
    """Return the shared ``mantissa.Mantissa`` instance, loading it on first use."""
    global _tk
    if _tk is None:
        _tk = load_mantissa().Mantissa()
    return _tk
