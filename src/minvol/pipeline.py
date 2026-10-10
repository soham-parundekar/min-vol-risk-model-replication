"""Shared loading and assembly for the analysis stages.

The data stages write a set of processed panels to disk. Everything after that
(covariance estimation, both index rules, the bias tests and the evaluation)
needs the same handful of them, plus the same exposure matrix built the same
way. Keeping that in one place means the design matrix the risk model is fitted
on is provably the same object the optimiser constrains and the attribution
decomposes, rather than three similar-looking reconstructions that drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import PARAMS, PATHS


@dataclass
class Panels:
    """Everything the analysis stages read, loaded once."""

    exposures: dict[str, pd.DataFrame]
    factor_returns: pd.DataFrame
    specific_returns: pd.DataFrame
    weekly_returns: pd.DataFrame
    daily_returns: pd.DataFrame
    market_cap_weekly: pd.DataFrame
    tradable: pd.DataFrame
    industry_labels: pd.Series
    adj_close: pd.DataFrame
    parent_returns: pd.Series

    @property
    def weekly_dates(self) -> pd.DatetimeIndex:
        return self.exposures[PARAMS["exposures"]["style_factors"][0]].index

    def asof(self, date) -> pd.Timestamp | None:
        """Most recent weekly cross-section on or before ``date``."""
        idx = self.weekly_dates
        pos = idx.searchsorted(pd.Timestamp(date), side="right") - 1
        return idx[pos] if pos >= 0 else None

    def asof_for_rebalance(self, date, lag_months: int = 0) -> pd.Timestamp | None:
        """Cross-section a rebalance is allowed to see.

        MSCI dates the covariance matrix to the end of the month before the
        review: the May rebalance uses end-of-April data and the November one
        end-of-October. With ``lag_months`` at zero this is just the latest
        cross-section on or before the rebalance.
        """
        date = pd.Timestamp(date)
        if lag_months > 0:
            cutoff = (date.replace(day=1) - pd.offsets.MonthBegin(lag_months - 1)
                      ) - pd.Timedelta(days=1)
        else:
            cutoff = date
        return self.asof(cutoff)

    def live_names(self, date) -> list[str]:
        """Names that were index members, priced and capitalised on ``date``."""
        caps = self.market_cap_weekly.loc[date]
        live = self.tradable.loc[date]
        return [n for n in self.market_cap_weekly.columns
                if bool(live.get(n, False)) and np.isfinite(caps.get(n, np.nan))
                and caps.get(n, 0.0) > 0]

    def parent_weights(self, date, names: list[str]) -> pd.Series:
        """Cap weights of the parent universe, normalised over ``names``."""
        caps = self.market_cap_weekly.loc[date, names].astype("float64").fillna(0.0)
        total = caps.sum()
        if total <= 0:
            return pd.Series(0.0, index=names)
        return caps / total

    def parent_weight_panel(self) -> pd.DataFrame:
        """Cap weights of the parent universe for every weekly cross-section.

        This is the ``b`` the USMV constraints are written against: the sector
        band and the style band are both stated relative to it, and the
        per-name cap is partly a multiple of it.
        """
        caps = self.market_cap_weekly.where(self.tradable)
        caps = caps.where(caps > 0)
        return caps.div(caps.sum(axis=1), axis=0)

    def exposure_matrix(self, date, names: list[str]) -> pd.DataFrame:
        """Design matrix: market, industry dummies, then the style block.

        Column order matters downstream, because the risk decomposition reads
        groups off the column names and the optimiser slices the style block out
        by position.
        """
        styles = PARAMS["exposures"]["style_factors"]
        out = pd.DataFrame(index=pd.Index(names, name="ticker"))
        out["market"] = 1.0
        labels = self.industry_labels.reindex(names).fillna("Other")
        for industry in sorted(labels.unique()):
            out[f"ind_{industry}"] = (labels == industry).astype("float64").to_numpy()
        for style in styles:
            out[style] = self.exposures[style].loc[date, names].fillna(0.0).to_numpy()
        return out


def load_panels() -> Panels:
    """Read every processed panel the analysis stages need."""
    styles = PARAMS["exposures"]["style_factors"]
    exposures = {
        s: pd.read_parquet(PATHS.processed / f"exposure_{s}.parquet") for s in styles
    }
    return Panels(
        exposures=exposures,
        factor_returns=pd.read_parquet(PATHS.processed / "factor_returns.parquet"),
        specific_returns=pd.read_parquet(PATHS.processed / "specific_returns.parquet"),
        weekly_returns=pd.read_parquet(PATHS.processed / "weekly_returns.parquet"),
        daily_returns=pd.read_parquet(PATHS.processed / "daily_returns.parquet"),
        market_cap_weekly=pd.read_parquet(PATHS.processed / "market_cap_weekly.parquet"),
        tradable=pd.read_parquet(PATHS.processed / "tradable_mask.parquet"),
        industry_labels=pd.read_parquet(
            PATHS.processed / "industry_labels.parquet")["industry"],
        adj_close=pd.read_parquet(PATHS.interim / "prices_adj_close.parquet"),
        parent_returns=pd.read_parquet(
            PATHS.processed / "parent_index_returns.parquet")["parent"],
    )


def covariance_bundles(panels: Panels, dates: pd.DatetimeIndex,
                       verbose: bool = True) -> dict[pd.Timestamp, dict]:
    """Fit all four estimators on each rebalance date.

    Keyed by the rebalance date, but every estimate is fitted on the weekly
    cross-section at or before it, so nothing from after the trade enters.
    """
    from .risk import covariance as cov

    lag = PARAMS["usmv"].get("covariance_lag_months", 0)
    out: dict[pd.Timestamp, dict] = {}
    for date in dates:
        asof = panels.asof_for_rebalance(date, lag)
        if asof is None:
            continue
        names = panels.live_names(asof)
        if len(names) < 60:
            continue
        x = panels.exposure_matrix(asof, names)
        bundle = cov.estimate_all(
            asof, names, panels.weekly_returns, x,
            panels.factor_returns, panels.specific_returns,
            panels.market_cap_weekly.loc[asof],
        )
        bundle["asof"] = asof
        out[date] = bundle
        if verbose:
            got = [k for k in ("structural", "sample", "ledoit_wolf", "pca")
                   if k in bundle]
            print(f"  cov {date.date()} (asof {asof.date()}): "
                  f"{len(bundle.get('names', []))} names, {got}", flush=True)
    return out
