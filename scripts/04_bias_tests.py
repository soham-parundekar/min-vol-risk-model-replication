"""Module D: are the risk forecasts calibrated?

Refits all four covariance estimators every week and standardises the following
week's realised portfolio return by the forecast made beforehand. The standard
deviation of those standardised returns is the bias statistic, and it should sit
near one.

Covariances are computed on demand and discarded. Holding 870 weeks of four
450-by-450 matrices would run to several gigabytes for no benefit, since each is
read once.

    python scripts/04_bias_tests.py [--every N]
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
from minvol.risk import bias_tests, covariance  # noqa: E402

ESTIMATORS = ["structural", "sample", "ledoit_wolf", "pca"]


def main(every: int = 1) -> None:
    started = time.time()
    PATHS.ensure()
    panels = pl.load_panels()

    weekly = panels.weekly_dates
    first = PARAMS["sample"]["min_weeks_for_covariance"]
    weekly = weekly[first:]
    if every > 1:
        weekly = weekly[::every]
    print(f"Bias tests over {len(weekly)} weekly dates "
          f"({weekly[0].date()} to {weekly[-1].date()})")

    cache: dict[pd.Timestamp, dict] = {}

    def covariance_fn(date: pd.Timestamp) -> dict:
        """Fit the estimators as at ``date``, keeping only the latest in memory."""
        if date in cache:
            return cache[date]
        cache.clear()
        names = panels.live_names(date)
        if len(names) < 60:
            cache[date] = {}
            return cache[date]
        x = panels.exposure_matrix(date, names)
        bundle = covariance.estimate_all(
            date, names, panels.weekly_returns, x,
            panels.factor_returns, panels.specific_returns,
            panels.market_cap_weekly.loc[date],
        )
        cache[date] = bundle
        return bundle

    def exposures_by_date(date, names):
        return panels.exposure_matrix(date, [n for n in names
                                             if n in panels.market_cap_weekly.columns])

    holdings = {}
    for est in ESTIMATORS:
        path = PATHS.processed / f"weights_usmv_{est}.parquet"
        if path.is_file():
            holdings[est] = pd.read_parquet(path)

    result = bias_tests.run(
        weekly, panels.weekly_returns, exposures_by_date,
        panels.market_cap_weekly, covariance_fn, ESTIMATORS,
        holdings=holdings, verbose=True,
    )

    summary = result["summary"]
    summary.to_parquet(PATHS.tables / "bias_statistics.parquet", index=False)
    print("\nBias statistics")
    print(summary.round(4).to_string(index=False))

    # Rolling series, which is where regime-specific miscalibration shows up.
    window = PARAMS["bias_tests"]["rolling_window_weeks"]
    rolling = {}
    for key, frame in result["standardised"].items():
        rolling[key] = bias_tests.rolling_bias_pooled(frame, window)
        # Keep the standardised panel so the rolling view can be recomputed
        # without refitting every covariance again.
        safe = key.replace("::", "__")
        frame.to_parquet(PATHS.processed / f"bias_standardised_{safe}.parquet")
    rolling_frame = pd.DataFrame(rolling).sort_index()
    rolling_frame.to_parquet(PATHS.processed / "bias_rolling.parquet")

    print("\nRolling bias, worst 52-week readings per series")
    for col in rolling_frame.columns:
        s = rolling_frame[col].dropna()
        if s.empty:
            continue
        print(f"  {col:34s} max {s.max():.2f} on {s.idxmax().date()} | "
              f"min {s.min():.2f} on {s.idxmin().date()}")

    covid = rolling_frame.loc["2020-01-01":"2020-12-31"]
    if not covid.empty:
        print("\nRolling bias through 2020")
        print(covid.resample("QE").last().round(3).to_string())

    print(f"\nFinished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--every", type=int, default=1,
                    help="test every Nth week (1 = all weeks)")
    main(**vars(ap.parse_args()))
