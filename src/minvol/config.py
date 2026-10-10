"""Project paths and the parameter file.

Everything downstream imports ``PARAMS`` and the ``Paths`` helper from here so
that no module has to guess where the repository root is or hard-code a number
that belongs in ``config/params.yaml``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


def _find_root(start: Path) -> Path:
    """Walk up from ``start`` until a directory holding config/params.yaml appears.

    Resolving the root by searching rather than by counting ``parent`` calls
    means the package still imports correctly when it is installed, when it is
    run from a notebook two directories down, and when tests are invoked from
    the repository root.
    """
    for candidate in [start, *start.parents]:
        if (candidate / "config" / "params.yaml").is_file():
            return candidate
    raise RuntimeError(
        f"No config/params.yaml found at or above {start}. "
        "Run from inside the repository, or set MINVOL_ROOT."
    )


ROOT = _find_root(Path(__file__).resolve())


@dataclass(frozen=True)
class Paths:
    """Canonical locations for inputs and outputs."""

    root: Path = ROOT
    config: Path = ROOT / "config"
    raw: Path = ROOT / "data" / "raw"
    interim: Path = ROOT / "data" / "interim"
    processed: Path = ROOT / "data" / "processed"
    reports: Path = ROOT / "reports"
    figures: Path = ROOT / "reports" / "figures"
    tables: Path = ROOT / "reports" / "tables"
    references: Path = ROOT / "references"

    def ensure(self) -> "Paths":
        for p in (self.raw, self.interim, self.processed,
                  self.reports, self.figures, self.tables):
            p.mkdir(parents=True, exist_ok=True)
        return self


PATHS = Paths()


@functools.lru_cache(maxsize=1)
def load_params(path: Path | None = None) -> dict[str, Any]:
    """Read config/params.yaml once and cache it for the process."""
    target = path or (PATHS.config / "params.yaml")
    with open(target, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


PARAMS = load_params()
