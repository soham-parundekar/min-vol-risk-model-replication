"""Module G, part one: turn weight panels into returns, then compare with the real ETFs.

Weights are set on a rebalance date and then left alone. Between rebalances the
portfolio drifts: a holding that rises becomes a larger share of the book
without anyone trading it. Re-normalising to the target weights every day would
quietly describe a daily-rebalanced portfolio, which is a different and much
more expensive strategy than either index runs, and it would understate
turnover to roughly zero.

So the arithmetic is buy-and-hold between rebalance dates,

    r_p,t = sum_i w_i,t-1 r_i,t,      w_i,t = w_i,t-1 (1 + r_i,t) / (1 + r_p,t)

with the weights reset only when the rule says to rebalance.

Costs are charged on the notional actually traded at each reset, which is the
sum of absolute weight changes. That is twice the one-way turnover figure the
constraint is written in, and stating it that way avoids the factor-of-two
ambiguity that makes published cost numbers hard to compare. Everything is
reported gross and net.

**What tracking error against the real ETF does and does not mean here.** The
replica is not trying to be USMV. It is trying to answer whether a factor risk
model built from free data reproduces what MSCI's commercial model and
optimiser produce. Several differences are baked in before any modelling
choice: the parent universe is the S&P 500 rather than the MSCI USA IMI, the
industry scheme is SIC-derived rather than GICS, the real fund's holdings
differ from the index it tracks, and the ETF return is net of a fee the replica
does not pay. The tracking error is therefore a measure of total divergence,
and the report attributes it rather than presenting it as a score.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PARAMS


def portfolio_returns(weights: pd.DataFrame, daily_returns: pd.DataFrame,
                      cost_bps: float | None = None
                      ) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Daily gross and net returns from a panel of rebalance-date weights.

    Returns the gross series, the net series, and a per-rebalance record of the
    notional traded and the cost charged.
    """
    cost_bps = PARAMS["evaluation"]["transaction_cost_bps"] if cost_bps is None else cost_bps
    cost_rate = cost_bps / 10_000.0

    rebal_dates = list(weights.index)
    if not rebal_dates:
        return pd.Series(dtype="float64"), pd.Series(dtype="float64"), pd.DataFrame()

    calendar = daily_returns.index
    start = calendar.searchsorted(rebal_dates[0], side="left")
    calendar = calendar[start:]

    held = pd.Series(dtype="float64")
    gross: dict[pd.Timestamp, float] = {}
    net: dict[pd.Timestamp, float] = {}
    trades: list[dict] = []
    next_rebal = 0

    for day in calendar:
        charge = 0.0
        if next_rebal < len(rebal_dates) and day >= rebal_dates[next_rebal]:
            target = weights.loc[rebal_dates[next_rebal]].dropna()
            target = target[target > 0]
            union = target.index.union(held.index)
            traded = float((target.reindex(union).fillna(0.0)
                            - held.reindex(union).fillna(0.0)).abs().sum())
            charge = traded * cost_rate
            trades.append({
                "date": rebal_dates[next_rebal],
                "notional_traded": traded,
                "one_way_turnover": traded / 2.0,
                "cost": charge,
                "n_holdings": int(len(target)),
            })
            held = target.copy()
            next_rebal += 1

        if held.empty:
            continue
        r = daily_returns.loc[day].reindex(held.index).fillna(0.0)
        day_return = float((held * r).sum())
        gross[day] = day_return
        net[day] = day_return - charge

        grown = held * (1.0 + r)
        total = grown.sum()
        held = grown / total if total > 0 else grown

    return (pd.Series(gross).sort_index(), pd.Series(net).sort_index(),
            pd.DataFrame(trades).set_index("date") if trades else pd.DataFrame())


def performance_stats(returns: pd.Series, rf: pd.Series | None = None,
                      periods_per_year: int | None = None) -> dict[str, float]:
    """Standard summary statistics for a daily return series."""
    ppy = periods_per_year or PARAMS["evaluation"]["trading_days_per_year"]
    r = returns.dropna()
    if len(r) < 2:
        return {}

    excess = r - rf.reindex(r.index).fillna(0.0) if rf is not None else r
    growth = float((1.0 + r).prod())
    years = len(r) / ppy
    vol = float(r.std(ddof=1) * np.sqrt(ppy))
    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=1) * np.sqrt(ppy)) if len(downside) > 1 else np.nan

    curve = (1.0 + r).cumprod()
    drawdown = curve / curve.cummax() - 1.0

    return {
        "n_days": int(len(r)),
        "start": str(r.index.min().date()),
        "end": str(r.index.max().date()),
        "total_return": growth - 1.0,
        "cagr": growth ** (1.0 / years) - 1.0 if years > 0 else np.nan,
        "volatility": vol,
        "sharpe": float(excess.mean() / r.std(ddof=1) * np.sqrt(ppy)) if r.std(ddof=1) > 0 else np.nan,
        "sortino": float(excess.mean() * ppy / downside_vol) if downside_vol and downside_vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()),
        "skew": float(r.skew()),
        "kurtosis": float(r.kurtosis()),
    }


def tracking_error(replica: pd.Series, target: pd.Series,
                   periods_per_year: int | None = None) -> dict[str, float]:
    """Annualised standard deviation of the return difference, plus related measures.

    Both series are restricted to days on which each actually traded, so a
    holiday in one and not the other cannot masquerade as tracking error.
    """
    ppy = periods_per_year or PARAMS["evaluation"]["trading_days_per_year"]
    joined = pd.concat([replica.rename("replica"), target.rename("target")],
                       axis=1).dropna()
    if len(joined) < 20:
        return {}
    diff = joined["replica"] - joined["target"]
    return {
        "n_days": int(len(joined)),
        "start": str(joined.index.min().date()),
        "end": str(joined.index.max().date()),
        "tracking_error_ann": float(diff.std(ddof=1) * np.sqrt(ppy)),
        "mean_active_return_ann": float(diff.mean() * ppy),
        "correlation": float(joined["replica"].corr(joined["target"])),
        "beta_to_target": float(
            joined["replica"].cov(joined["target"]) / joined["target"].var()
        ),
        "replica_vol_ann": float(joined["replica"].std(ddof=1) * np.sqrt(ppy)),
        "target_vol_ann": float(joined["target"].std(ddof=1) * np.sqrt(ppy)),
    }


def etf_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily total returns from adjusted ETF closes."""
    return prices.sort_index().pct_change(fill_method=None)


def realised_vol_by_period(returns: pd.Series, freq: str = "YE",
                           periods_per_year: int | None = None) -> pd.Series:
    """Annualised realised volatility within each calendar period."""
    ppy = periods_per_year or PARAMS["evaluation"]["trading_days_per_year"]
    return returns.resample(freq).std(ddof=1) * np.sqrt(ppy)
