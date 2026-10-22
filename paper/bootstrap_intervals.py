"""Block-bootstrap intervals for the two differences the paper's argument rests on.

The manuscript compares a 2.10 percentage point tracking-error difference between
two readings of the style exemption against the much smaller spread produced by
changing the covariance estimator, and concludes the first dominates. That
conclusion is a comparison of two point estimates, and a reader is entitled to
ask whether either is distinguishable from zero, let alone from the other.

Daily active returns are serially correlated and the rebalance calendar is
semi-annual, so an iid resample would understate the uncertainty badly. We use a
circular block bootstrap with a 21-day block, roughly a trading month, resampling
whole rows so that every specification sees the same resampled dates and the
difference between two specifications is a paired quantity.

This quantifies existing estimates. It does not re-run the pipeline, change a
portfolio, or produce a new empirical finding.

    python3 paper/bootstrap_intervals.py
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
I = ROOT / "data" / "interim"
OUT = ROOT / "paper" / "tables"

BLOCK = 21
DRAWS = 2000
SEED = 20260301          # the project's configured seed, reused for traceability
TRADING_DAYS = 252

ESTIMATORS = ["usmv_structural", "usmv_sample", "usmv_ledoit_wolf", "usmv_pca"]
VARIANT = "usmv_structural_betafree"


def _active_matrix() -> pd.DataFrame:
    """Daily active returns of every USMV specification against the fund."""
    rets = pd.read_parquet(P / "portfolio_returns_net.parquet")
    etf = pd.read_parquet(I / "etf_prices.parquet")
    fund = etf["USMV"].dropna().pct_change().dropna()

    cols = ESTIMATORS + [VARIANT]
    joined = rets[cols].join(fund.rename("fund"), how="inner").dropna()
    return joined[cols].sub(joined["fund"], axis=0)


def _te(active: np.ndarray) -> np.ndarray:
    """Annualised tracking error down each column."""
    return active.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)


def _circular_blocks(n: int, rng: np.random.Generator) -> np.ndarray:
    """Row indices for one circular block-bootstrap replicate."""
    starts = rng.integers(0, n, size=int(np.ceil(n / BLOCK)))
    idx = (starts[:, None] + np.arange(BLOCK)[None, :]).ravel() % n
    return idx[:n]


def main() -> None:
    active = _active_matrix()
    names = list(active.columns)
    A = active.to_numpy()
    n = len(A)
    rng = np.random.default_rng(SEED)

    point = _te(A)
    base_i = names.index("usmv_structural")
    var_i = names.index(VARIANT)
    est_i = [names.index(e) for e in ESTIMATORS]

    exemption = np.empty(DRAWS)
    spread = np.empty(DRAWS)
    for b in range(DRAWS):
        te = _te(A[_circular_blocks(n, rng)])
        exemption[b] = te[base_i] - te[var_i]
        spread[b] = te[est_i].max() - te[est_i].min()

    def ci(x):
        return np.percentile(x, 2.5), np.percentile(x, 97.5)

    rows = []
    e_lo, e_hi = ci(exemption)
    s_lo, s_hi = ci(spread)
    rows.append({
        "quantity": "exemption effect on tracking error",
        "point": float(point[base_i] - point[var_i]),
        "ci_low": float(e_lo), "ci_high": float(e_hi),
        "note": "structural minus beta-exempt, both net, paired resample",
    })
    rows.append({
        "quantity": "estimator spread in tracking error",
        "point": float(point[est_i].max() - point[est_i].min()),
        "ci_low": float(s_lo), "ci_high": float(s_hi),
        "note": "max minus min across the four estimators, paired resample",
    })
    # Does the exemption effect exceed the estimator spread on the same draw?
    frac = float((exemption > spread).mean())
    rows.append({
        "quantity": "fraction of draws where exemption effect exceeds estimator spread",
        "point": frac, "ci_low": np.nan, "ci_high": np.nan,
        "note": f"{DRAWS} draws, block {BLOCK}, seed {SEED}",
    })

    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT / "bootstrap_intervals.parquet", index=False)

    pd.set_option("display.width", 200)
    print(out.to_string(index=False))
    print(f"\nobservations {n}, block {BLOCK}, draws {DRAWS}, seed {SEED}")


if __name__ == "__main__":
    main()
