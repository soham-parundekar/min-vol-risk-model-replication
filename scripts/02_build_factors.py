"""Build weekly exposures (Module A) and cross-sectional factor returns (Module B).

Also writes the survivorship-gap report, which compares the rebuilt
cap-weighted parent index against SPY. That comparison is the study's only
direct measurement of what the missing delisted names cost, so it is produced
here rather than left to the evaluation stage.

    python scripts/02_build_factors.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minvol.config import PARAMS, PATHS  # noqa: E402
from minvol.data import universe as universe_mod  # noqa: E402
from minvol.factors import exposures as exposures_mod  # noqa: E402
from minvol.factors import returns as returns_mod  # noqa: E402


def survivorship_report(parent: pd.Series, spy: pd.Series,
                        start: str) -> pd.DataFrame:
    """Annual and full-sample gap between the rebuilt parent index and SPY.

    A positive gap means the rebuilt index outperformed the real one, which is
    the direction survivorship bias pushes: companies that failed are missing
    from the free price history, so what is left is the set that made it.
    Part of the gap is SPY's own expense ratio, which works the other way and
    is noted alongside the number rather than netted off silently.
    """
    joined = pd.concat([parent.rename("parent"), spy.rename("spy")],
                       axis=1, sort=True).dropna()
    joined = joined.loc[start:]
    diff = joined["parent"] - joined["spy"]
    ppy = PARAMS["evaluation"]["trading_days_per_year"]

    rows = [{
        "period": "full sample",
        "n_days": len(joined),
        "parent_cagr": (1 + joined["parent"]).prod() ** (ppy / len(joined)) - 1,
        "spy_cagr": (1 + joined["spy"]).prod() ** (ppy / len(joined)) - 1,
        "gap_ann": diff.mean() * ppy,
        "tracking_error_ann": diff.std(ddof=1) * np.sqrt(ppy),
        "correlation": joined["parent"].corr(joined["spy"]),
        "parent_vol_ann": joined["parent"].std(ddof=1) * np.sqrt(ppy),
        "spy_vol_ann": joined["spy"].std(ddof=1) * np.sqrt(ppy),
    }]
    for year, grp in joined.groupby(joined.index.year):
        d = grp["parent"] - grp["spy"]
        rows.append({
            "period": str(year),
            "n_days": len(grp),
            "parent_cagr": (1 + grp["parent"]).prod() - 1,
            "spy_cagr": (1 + grp["spy"]).prod() - 1,
            "gap_ann": d.mean() * ppy,
            "tracking_error_ann": d.std(ddof=1) * np.sqrt(ppy),
            "correlation": grp["parent"].corr(grp["spy"]),
            "parent_vol_ann": grp["parent"].std(ddof=1) * np.sqrt(ppy),
            "spy_vol_ann": grp["spy"].std(ddof=1) * np.sqrt(ppy),
        })
    return pd.DataFrame(rows).set_index("period")


def main() -> None:
    started = time.time()
    PATHS.ensure()
    sample = PARAMS["sample"]

    uni, _ = universe_mod.build()
    px = {
        name: pd.read_parquet(PATHS.interim / f"prices_{name}.parquet")
        for name in ("close", "adj_close", "volume")
    }
    market_cap = pd.read_parquet(PATHS.processed / "market_cap.parquet")
    book_to_price = pd.read_parquet(PATHS.processed / "book_to_price.parquet")

    print("Module A: weekly exposures")
    built = exposures_mod.build(uni, px, market_cap, book_to_price)
    dates = built["dates"]
    print(f"   {len(dates)} weekly cross-sections, "
          f"{dates[0].date()} to {dates[-1].date()}")
    coverage = built["coverage"]
    print(coverage.groupby("factor")["coverage_pct"].mean().round(1).to_string())

    print("\nSurvivorship gap: rebuilt parent index against SPY")
    etf = pd.read_parquet(PATHS.interim / "etf_prices.parquet")
    spy = etf[PARAMS["evaluation"]["benchmark_ticker"]].pct_change(fill_method=None)
    report = survivorship_report(built["parent_returns"], spy, sample["exposure_start"])
    report.to_parquet(PATHS.tables / "survivorship_gap.parquet")
    print(report.loc[["full sample"]].round(4).to_string())

    print("\nModule B: cross-sectional factor returns")
    out = returns_mod.build(
        built["exposures"], px["adj_close"], built["market_cap_weekly"],
        pd.read_parquet(PATHS.processed / "industries.parquet"), built["tradable"],
    )
    factors = out["factor_returns"]
    diag = out["diagnostics"]
    print(f"   {factors.shape[0]} weeks, {factors.shape[1]} factors")
    print(f"   cross-sectional R2: mean {diag['r2'].mean():.3f}, "
          f"median {diag['r2'].median():.3f}")
    print(f"   names per regression: median {diag['n'].median():.0f}")

    summary = pd.DataFrame({
        "mean_ann": factors.mean() * 52,
        "vol_ann": factors.std(ddof=1) * np.sqrt(52),
    })
    summary["t_stat"] = (
        factors.mean() / factors.std(ddof=1) * np.sqrt(len(factors))
    )
    summary.sort_values("vol_ann", ascending=False).to_parquet(
        PATHS.tables / "factor_return_summary.parquet"
    )
    print(summary.sort_values("vol_ann", ascending=False).round(4).to_string())

    print(f"\nFinished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
