"""Module A: weekly style exposures on the live cross-section.

Six descriptors, computed every week for whichever stocks were actually in the
index that week:

size            log market cap
beta            252-day regression beta against the rebuilt parent index,
                shrunk toward one
resid_vol       volatility of the residual from that same regression
momentum        twelve-month return skipping the most recent month
book_to_price   point-in-time book equity over market cap
liquidity       log average daily dollar volume over the trailing quarter

Two choices here matter more than they look.

**Residual volatility rather than total volatility.** A risk model's volatility
factor is meant to capture volatility that is *not* already explained by market
exposure. Using raw total volatility double-counts beta, and the two are
correlated enough in the cross-section that the style block starts fighting the
market factor for the same variation. Taking the residual from the beta
regression makes the descriptor orthogonal to beta by construction.

**Standardising on the live cross-section, every week.** The tempting shortcut
is to pool the whole history and standardise once. That leaks: the mean and
standard deviation of 2013's size distribution would then depend on market caps
from 2019. Every week's standardisation uses only the names in the index that
week and only data available that week.

The target is a cap-weighted mean of zero and an equal-weighted standard
deviation of one. That pairing is the Barra convention and it is not arbitrary.
Centring on the cap-weighted mean means the parent index has zero exposure to
every style by construction, so a constraint written as "stay within 0.25
standard deviations of the parent" becomes simply "stay within 0.25", and the
USMV style bands in Module E are expressed in those units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PARAMS, PATHS

STYLE_ORDER = PARAMS["exposures"]["style_factors"]


def weekly_dates(trading_days: pd.DatetimeIndex, day: str = "FRI",
                 start: str | None = None, end: str | None = None) -> pd.DatetimeIndex:
    """One observation date per week: the last trading day on or before each target day.

    Anchoring to the last available session rather than to the calendar Friday
    keeps holiday weeks from silently dropping out of the sample.
    """
    days = pd.DatetimeIndex(trading_days).sort_values()
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    if len(days) == 0:
        return pd.DatetimeIndex([])
    frame = pd.Series(days, index=days)
    weekly = frame.resample(f"W-{day}").last().dropna()
    return pd.DatetimeIndex(weekly.to_numpy()).unique()


def membership_mask(universe, dates: pd.DatetimeIndex,
                    tickers: list[str]) -> pd.DataFrame:
    """Boolean panel: was this ticker an index member on this date?"""
    pos = {t: i for i, t in enumerate(tickers)}
    mask = np.zeros((len(dates), len(tickers)), dtype=bool)
    for r, d in enumerate(dates):
        for t in universe.members_on(d):
            j = pos.get(t)
            if j is not None:
                mask[r, j] = True
    return pd.DataFrame(mask, index=dates, columns=tickers)


def parent_index_returns(returns: pd.DataFrame, market_cap: pd.DataFrame,
                         mask: pd.DataFrame) -> pd.Series:
    """Daily cap-weighted return of the rebuilt parent universe.

    Weights come from the previous session's market cap, which is what an
    investor holding the index would actually have had on that morning. Using
    same-day caps would quietly use the day's own price move to set the weight
    it is applied to.
    """
    cap = market_cap.where(mask).shift(1)
    cap = cap.where(returns.notna())
    weights = cap.div(cap.sum(axis=1), axis=0)
    return (weights * returns).sum(axis=1, min_count=1).rename("parent")


def rolling_beta_resid_vol(returns: pd.DataFrame, market: pd.Series,
                           dates: pd.DatetimeIndex, window: int,
                           min_obs: int | None = None
                           ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Beta and residual volatility from a trailing OLS fit, evaluated on ``dates``.

    Evaluating only on the weekly observation dates rather than every session
    turns this from a 4,700-step rolling regression into roughly 900 matrix
    operations. For each date the whole cross-section is fitted at once:
    slope is cov(r_i, r_m) / var(r_m) across columns, and the residual standard
    deviation follows from the regression identity rather than from rebuilding
    residuals stock by stock.
    """
    min_obs = min_obs or int(window * 0.6)
    all_days = returns.index
    betas = pd.DataFrame(np.nan, index=dates, columns=returns.columns)
    rvols = pd.DataFrame(np.nan, index=dates, columns=returns.columns)

    r_mkt_full = market.reindex(all_days)
    for d in dates:
        end = all_days.searchsorted(d, side="right")
        start = max(0, end - window)
        if end - start < min_obs:
            continue
        y = returns.iloc[start:end].to_numpy(dtype="float64")
        x = r_mkt_full.iloc[start:end].to_numpy(dtype="float64")

        valid_x = np.isfinite(x)
        if valid_x.sum() < min_obs:
            continue
        y = y[valid_x]
        x = x[valid_x]
        ok = np.isfinite(y)
        n = ok.sum(axis=0)

        ysafe = np.where(ok, y, 0.0)
        xcol = x[:, None]
        sum_x = (xcol * ok).sum(axis=0)
        sum_y = ysafe.sum(axis=0)
        sum_xx = ((xcol ** 2) * ok).sum(axis=0)
        sum_xy = (xcol * ysafe).sum(axis=0)
        sum_yy = (ysafe ** 2).sum(axis=0)

        with np.errstate(invalid="ignore", divide="ignore"):
            denom = n * sum_xx - sum_x ** 2
            slope = (n * sum_xy - sum_x * sum_y) / denom
            intercept = (sum_y - slope * sum_x) / n
            # Residual sum of squares written out from the sufficient
            # statistics, which avoids materialising an N x T residual array.
            rss = (sum_yy - 2 * slope * sum_xy - 2 * intercept * sum_y
                   + 2 * slope * intercept * sum_x + (slope ** 2) * sum_xx
                   + n * intercept ** 2)
            resid_sd = np.sqrt(np.maximum(rss, 0.0) / np.maximum(n - 2, 1))

        bad = (n < min_obs) | ~np.isfinite(slope)
        slope = np.where(bad, np.nan, slope)
        resid_sd = np.where(bad, np.nan, resid_sd)
        betas.loc[d] = slope
        rvols.loc[d] = resid_sd
    return betas, rvols


def shrink_beta(raw: pd.DataFrame) -> pd.DataFrame:
    """Blume-style shrinkage toward one.

    A 252-day regression beta is a noisy estimate, and the noise is not
    symmetric in its consequences: a minimum-variance optimiser hunts for the
    lowest-beta names and so loads on whichever betas are most understated by
    estimation error. Pulling every estimate toward one takes the edge off
    that. Slope and intercept sum to one so the market portfolio keeps a beta
    of exactly one.
    """
    cfg = PARAMS["exposures"]
    return cfg["beta_shrink_slope"] * raw + cfg["beta_shrink_intercept"]


def momentum(adj_close: pd.DataFrame, dates: pd.DatetimeIndex,
             lookback: int, skip: int) -> pd.DataFrame:
    """Cumulative return from t-lookback to t-skip.

    Skipping the most recent month is the standard construction. Without it the
    descriptor picks up short-horizon reversal, which runs the opposite way to
    momentum and muddies the factor.
    """
    px = adj_close
    out = pd.DataFrame(np.nan, index=dates, columns=px.columns)
    all_days = px.index
    for d in dates:
        end = all_days.searchsorted(d, side="right") - 1
        i_recent = end - skip
        i_start = end - lookback
        if i_start < 0 or i_recent <= i_start:
            continue
        p0 = px.iloc[i_start]
        p1 = px.iloc[i_recent]
        out.loc[d] = (p1 / p0 - 1.0).to_numpy()
    return out


def liquidity(close: pd.DataFrame, volume: pd.DataFrame, dates: pd.DatetimeIndex,
              window: int) -> pd.DataFrame:
    """Log of average daily dollar volume over the trailing window."""
    dollar = (close * volume).astype("float64")
    rolled = dollar.rolling(window, min_periods=max(5, window // 3)).mean()
    sub = rolled.reindex(dates, method="ffill")
    return np.log(sub.where(sub > 0))


def winsorize_cross_section(row: pd.Series, lower: float, upper: float) -> pd.Series:
    """Clip a single week's cross-section at the given percentiles."""
    vals = row.dropna()
    if len(vals) < 10:
        return row
    lo, hi = np.nanpercentile(vals, [lower, upper])
    return row.clip(lower=lo, upper=hi)


def standardize_cross_section(row: pd.Series, caps: pd.Series) -> pd.Series:
    """Cap-weighted mean of zero, equal-weighted standard deviation of one.

    The scale is the ordinary equal-weighted standard deviation. Because a
    standard deviation does not depend on where the series is centred, it can
    be taken before or after subtracting the cap-weighted mean and gives the
    same number either way.
    """
    vals = row.dropna()
    if len(vals) < 10:
        return pd.Series(np.nan, index=row.index)
    w = caps.reindex(vals.index).astype("float64")
    w = w.where(w > 0)
    if w.notna().sum() < 10 or not np.isfinite(w.sum()) or w.sum() <= 0:
        centre = vals.mean()
    else:
        w = w.fillna(0.0)
        centre = float((w * vals).sum() / w.sum())
    scale = float(vals.std(ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        return pd.Series(np.nan, index=row.index)
    return (row - centre) / scale


def build(universe, prices: dict[str, pd.DataFrame], market_cap: pd.DataFrame,
          book_to_price: pd.DataFrame, cache: bool = True
          ) -> dict[str, object]:
    """Compute the standardised weekly exposure panel and the parent index.

    Returns the standardised exposures keyed by style name, the raw descriptors
    for auditing, the weekly membership mask, the daily parent index return
    series and a coverage table.
    """
    cfg = PARAMS["exposures"]
    s = PARAMS["sample"]
    PATHS.ensure()

    adj, close, vol = prices["adj_close"], prices["close"], prices["volume"]
    returns = adj.sort_index().pct_change(fill_method=None)
    tickers = list(adj.columns)

    dates = weekly_dates(adj.index, s["weekly_day"], s["exposure_start"], s["price_end"])
    daily_mask = membership_mask(universe, adj.index, tickers)
    parent = parent_index_returns(returns, market_cap, daily_mask)

    raw_beta, raw_rvol = rolling_beta_resid_vol(
        returns, parent, dates, cfg["beta_window_days"]
    )
    raw = {
        "size": np.log(market_cap.where(market_cap > 0)).reindex(dates, method="ffill"),
        "beta": shrink_beta(raw_beta),
        "resid_vol": raw_rvol,
        "momentum": momentum(adj, dates, cfg["momentum_lookback_days"],
                             cfg["momentum_skip_days"]),
        "book_to_price": book_to_price.reindex(dates, method="ffill"),
        "liquidity": liquidity(close, vol, dates, cfg["liquidity_window_days"]),
    }

    weekly_mask = daily_mask.reindex(dates)
    caps_weekly = market_cap.reindex(dates, method="ffill").where(weekly_mask)
    # A name needs a live price that week as well as index membership; a halted
    # or not-yet-listed stock contributes nothing to the cross-section.
    tradable = weekly_mask & adj.reindex(dates, method="ffill").notna()

    standardized: dict[str, pd.DataFrame] = {}
    coverage_rows = []
    for name in STYLE_ORDER:
        panel = raw[name].reindex(index=dates, columns=tickers).where(tradable)
        out = pd.DataFrame(np.nan, index=dates, columns=tickers)
        for d in dates:
            row = winsorize_cross_section(
                panel.loc[d], cfg["winsor_lower_pct"], cfg["winsor_upper_pct"]
            )
            out.loc[d] = standardize_cross_section(row, caps_weekly.loc[d])
        standardized[name] = out
        coverage_rows.append(
            pd.DataFrame({
                "date": dates,
                "factor": name,
                "n_live": tradable.sum(axis=1).to_numpy(),
                "n_with_value": panel.notna().sum(axis=1).to_numpy(),
            })
        )

    coverage = pd.concat(coverage_rows, ignore_index=True)
    coverage["coverage_pct"] = (
        100.0 * coverage["n_with_value"] / coverage["n_live"].replace(0, np.nan)
    )

    result = {
        "dates": dates,
        "exposures": standardized,
        "raw": raw,
        "tradable": tradable,
        "market_cap_weekly": caps_weekly,
        "parent_returns": parent,
        "daily_returns": returns,
        "coverage": coverage,
    }
    if cache:
        for name, df in standardized.items():
            df.to_parquet(PATHS.processed / f"exposure_{name}.parquet")
        tradable.to_parquet(PATHS.processed / "tradable_mask.parquet")
        caps_weekly.to_parquet(PATHS.processed / "market_cap_weekly.parquet")
        parent.to_frame().to_parquet(PATHS.processed / "parent_index_returns.parquet")
        returns.to_parquet(PATHS.processed / "daily_returns.parquet")
        coverage.to_parquet(PATHS.processed / "exposure_coverage.parquet", index=False)
    return result
