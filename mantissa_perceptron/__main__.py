"""Explicit dataset downloader — the only code here that touches the network.

Usage:
    python -m mantissa_perceptron fetch <name|all>
    python -m mantissa_perceptron list
"""
from __future__ import annotations

import sys
import urllib.request

from .datasets import DATASETS, data_dir, data_path


def _fetch(name: str) -> None:
    path = data_path(name)
    if path.is_file():
        print(f"{name}: already present at {path}")
        return
    url = DATASETS[name].url
    print(f"{name}: {url}\n  -> {path}")
    data_dir().mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as r:
        body = r.read()
    path.write_bytes(body)
    print(f"  done ({len(body):,} bytes)")


def main(argv) -> int:
    if len(argv) == 1 and argv[0] == "list":
        for name, spec in DATASETS.items():
            state = "present" if data_path(name).is_file() else "missing"
            print(f"{name:14} {state:8} {spec.note}")
        return 0
    if len(argv) == 2 and argv[0] == "fetch":
        names = list(DATASETS) if argv[1] == "all" else [argv[1]]
        for name in names:
            if name not in DATASETS:
                print(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}",
                      file=sys.stderr)
                return 2
            _fetch(name)
        return 0
    print(__doc__.strip(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
