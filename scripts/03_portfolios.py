"""Fit the risk models and build both replica portfolios.

Runs the USMV rule under each of the four covariance estimators so that Module
G can compare them on realised risk rather than on how good the covariance
looked in sample, then runs the SPLV rule, which needs no risk model at all.

    python scripts/03_portfolios.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minvol import pipeline as pl  # noqa: E402
from minvol.config import PARAMS, PATHS  # noqa: E402
from minvol.data import universe as universe_mod  # noqa: E402
from minvol.optimize import schedule, splv_rules, usmv_rules  # noqa: E402

ESTIMATORS = ["structural", "sample", "ledoit_wolf", "pca"]


def main(only: str | None = None, skip_splv: bool = False) -> None:
    started = time.time()
    PATHS.ensure()
    panels = pl.load_panels()
    sample = PARAMS["sample"]

    usmv_dates = schedule.rebalance_dates(
        panels.adj_close.index, PARAMS["usmv"]["rebalance_months"],
        PARAMS["usmv"]["rebalance_rule"], sample["exposure_start"], sample["price_end"],
    )
    print(f"USMV rebalances: {len(usmv_dates)} "
          f"({usmv_dates[0].date()} to {usmv_dates[-1].date()})")

    print("\nFitting covariance estimators")
    bundles = pl.covariance_bundles(panels, usmv_dates, verbose=True)

    # Record what each estimator chose, which is part of the story: the number
    # of PCA factors and the shrinkage intensity both move with the data.
    choices = pd.DataFrame([
        {"date": d, "asof": b.get("asof"), "n_names": len(b.get("names", [])),
         "n_obs": b.get("n_obs"), "pca_k": b.get("pca_k"),
         "ledoit_wolf_intensity": b.get("ledoit_wolf_intensity")}
        for d, b in bundles.items()
    ]).set_index("date")
    choices.to_parquet(PATHS.processed / "estimator_choices.parquet")
    print(choices[["n_names", "n_obs", "pca_k", "ledoit_wolf_intensity"]]
          .describe().round(3).to_string())

    parent = panels.parent_weight_panel()

    # The optimiser normalises parent weights over the covariance's name set,
    # which is narrower than the full tradable universe. Saving the vector it
    # actually used lets the constraint tests check the same quantity the
    # solver constrained, rather than a near-miss reconstruction that reports
    # phantom breaches of a few tenths of a basis point.
    used = {}
    for date, bundle in bundles.items():
        asof = bundle.get("asof")
        names = [n for n in bundle.get("names", []) if n in parent.columns]
        if asof is None or not names:
            continue
        row = parent.loc[asof, names].astype("float64").fillna(0.0)
        total = row.sum()
        if total > 0:
            used[date] = row / total
    pd.DataFrame(used).T.sort_index().to_parquet(
        PATHS.processed / "parent_weights_used.parquet"
    )

    def run_variant(estimator: str, label: str, free: list[str] | None) -> None:
        weights, diagnostics = usmv_rules.run(
            usmv_dates, estimator, bundles, parent,
            panels.industry_labels, panels.exposures, verbose=True,
            free=free, daily_returns=panels.daily_returns,
        )
        weights.to_parquet(PATHS.processed / f"weights_usmv_{label}.parquet")
        if not diagnostics.empty:
            # The relaxation record is a dict per row; store it as text so the
            # parquet stays flat and still readable.
            flat = diagnostics.copy()
            flat["relaxations"] = flat["relaxations"].astype(str)
            flat.to_parquet(PATHS.processed / f"diag_usmv_{label}.parquet")
            print(f"  {len(weights)} rebalances, median holdings "
                  f"{diagnostics['n_holdings'].median():.0f}, "
                  f"mean turnover {diagnostics['turnover'].mean():.3f}, "
                  f"relaxed on {(diagnostics['ladder_rung'] > 0).sum()} dates")

    wanted = [only] if only and only in ESTIMATORS else (
        [] if only else ESTIMATORS)
    for estimator in wanted:
        print(f"\nUSMV rule under {estimator}")
        run_variant(estimator, estimator, None)

    # The 2012 methodology exempts one volatility factor; the 2025 document
    # names two, BETA and RESVOL. Running both readings is the only way to find
    # out which one the live index actually behaves like.
    variant = PARAMS["exposures"].get("unconstrained_styles_variant")
    if variant and only in (None, "betafree"):
        print(f"\nUSMV rule under structural, styles {variant} left free")
        run_variant("structural", "structural_betafree", variant)

    if skip_splv:
        print(f"\nFinished in {time.time() - started:.0f}s")
        return

    print("\nSPLV rule")
    splv_dates = schedule.rebalance_dates(
        panels.adj_close.index, PARAMS["splv"]["rebalance_months"],
        PARAMS["splv"]["rebalance_rule"],
        PARAMS["evaluation"]["splv_inception"], sample["price_end"],
    )
    uni, _ = universe_mod.build()
    tradable_names = set(panels.tradable.columns[panels.tradable.any(axis=0)])
    splv_weights, splv_diag = splv_rules.run(
        splv_dates, panels.daily_returns, uni, tradable_names, verbose=False,
    )
    splv_weights.to_parquet(PATHS.processed / "weights_splv.parquet")
    splv_diag.to_parquet(PATHS.processed / "diag_splv.parquet")
    print(f"  {len(splv_weights)} rebalances "
          f"({splv_weights.index.min().date()} to {splv_weights.index.max().date()})")
    print(splv_diag[["n_holdings", "turnover", "median_selected_vol_ann",
                     "max_weight"]].describe().round(4).to_string())

    print(f"\nFinished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None,
                    help="one estimator name, or 'betafree' for the variant")
    ap.add_argument("--skip-splv", action="store_true")
    main(**vars(ap.parse_args()))
