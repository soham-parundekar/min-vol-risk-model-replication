"""Module F: the SPLV rule.

Take the 100 S&P 500 constituents with the lowest trailing 252-day realised
volatility and weight each one by the inverse of that volatility, so the
quietest names get the largest positions. Rebalance quarterly. There is no
optimiser, no covariance matrix, no sector constraint and no turnover cap.

It fits comfortably in one screen, which is the point. The entire study is a
comparison between this and the machinery in Module E, and the interesting
question is not whether the optimiser is more sophisticated. It plainly is.
The question is whether that sophistication shows up in realised risk, and
whether what it buys is worth the turnover and the model risk it brings with
it.

Worth noticing what the rule gives up by construction. Ranking on *total*
volatility with no sector constraint means the selection concentrates wherever
low-volatility stocks happen to live, which in practice is utilities, staples
and real estate. A sector bet is not a side effect of the method; it is the
method. MSCI's sector band exists precisely to stop that from happening, and
the risk decomposition in Module G is where the difference becomes visible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PARAMS


def trailing_volatility(returns: pd.DataFrame, asof: pd.Timestamp, window: int,
                        min_obs: int | None = None) -> pd.Series:
    """Realised standard deviation of daily returns over the trailing window.

    Total volatility, not residual. That is what the published rule uses, and
    swapping in a market-adjusted figure would be a different index.
    """
    min_obs = min_obs or int(window * 0.8)
    hist = returns.loc[returns.index <= asof].tail(window)
    if len(hist) < min_obs:
        return pd.Series(dtype="float64")
    counts = hist.notna().sum()
    vol = hist.std(ddof=1)
    return vol.where(counts >= min_obs).dropna()


def select_and_weight(vol: pd.Series, n_holdings: int) -> pd.Series:
    """Lowest-volatility names, weighted by inverse volatility and normalised."""
    usable = vol[vol > 0].nsmallest(n_holdings)
    if usable.empty:
        return pd.Series(dtype="float64")
    inverse = 1.0 / usable
    return inverse / inverse.sum()


def run(dates: pd.DatetimeIndex, daily_returns: pd.DataFrame, universe,
        tradable_names: set[str] | None = None, verbose: bool = True
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the rule at each rebalance date and return weights plus diagnostics."""
    cfg = PARAMS["splv"]
    rows: dict[pd.Timestamp, pd.Series] = {}
    diagnostics: list[dict] = []
    prev: pd.Series | None = None

    for date in dates:
        members = universe.members_on(date)
        names = [n for n in daily_returns.columns if n in members]
        if tradable_names is not None:
            names = [n for n in names if n in tradable_names]
        if len(names) < cfg["n_holdings"]:
            continue

        vol = trailing_volatility(daily_returns.loc[:, names], date,
                                  cfg["vol_window_days"])
        if len(vol) < cfg["n_holdings"]:
            continue
        weights = select_and_weight(vol, cfg["n_holdings"])
        if weights.empty:
            continue

        turnover = np.nan
        if prev is not None:
            union = weights.index.union(prev.index)
            turnover = 0.5 * float(
                (weights.reindex(union).fillna(0.0)
                 - prev.reindex(union).fillna(0.0)).abs().sum()
            )

        rows[date] = weights
        diagnostics.append({
            "date": date,
            "n_holdings": int(len(weights)),
            "n_candidates": int(len(vol)),
            "turnover": turnover,
            "median_selected_vol_ann": float(
                vol.reindex(weights.index).median()
                * np.sqrt(PARAMS["evaluation"]["trading_days_per_year"])
            ),
            "max_weight": float(weights.max()),
        })
        prev = weights
        if verbose:
            print(f"  {date.date()} SPLV: {len(weights)} names, "
                  f"turnover={turnover if np.isfinite(turnover) else 0:.3f}, "
                  f"max_w={weights.max()*100:.2f}%", flush=True)

    panel = pd.DataFrame(rows).T.sort_index()
    panel.index.name = "date"
    diag = pd.DataFrame(diagnostics).set_index("date") if diagnostics else pd.DataFrame()
    return panel, diag
