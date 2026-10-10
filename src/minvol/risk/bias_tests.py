"""Module D: does the risk model's forecast match what actually happened?

For a portfolio held over week t, take the return it actually earned and divide
it by the standard deviation the model predicted *before* the week began:

    z_t = r_t / sigma_hat_{t-1}

and let the bias statistic be the standard deviation of z across the sample. A
well-calibrated model gives b near 1. Below 1 means the model overstated risk;
above 1, and more worryingly, means it understated it.

The timing is the whole test. Predicting week t's volatility using week t's
data would produce a beautiful number and measure nothing: the forecast would
have seen the outcome. Every sigma here comes from a model fitted on data
ending at t-1, applied to weights also fixed at t-1.

Under the null that the model is right, b has standard error roughly
1/sqrt(2T), so a two-sided 95% band is about 1 +/- sqrt(2/T). With a few hundred
weekly observations that band is tight enough to be informative and wide enough
that small deviations should not be over-read.

A full-sample b near 1 is necessary but nowhere near sufficient. A model can
overstate risk in calm years and understate it in a crisis and still average
out to something respectable. The rolling series is where that shows up, and
early 2020 is the obvious place to look: a model whose correlations are
estimated over several years is still describing a pre-pandemic world in
February 2020. Reporting only the full-sample number would hide exactly the
failure a risk model exists to avoid.

Three portfolio families are tested, because bias is not a single property of a
covariance matrix but depends on which direction you look in:

* **factor-mimicking** portfolios probe the factor block directly;
* **random** long-only portfolios probe generic directions, where specific risk
  dominates and errors tend to diversify away;
* the **minimum-variance** portfolios the model itself produced are the hardest
  test. An optimiser actively seeks the directions a model believes are
  quietest, so any underestimate of risk is precisely what it loads on. If a
  model is going to fail anywhere, it fails here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PARAMS


def confidence_band(n_obs: int) -> tuple[float, float]:
    """Approximate two-sided 95% band for the bias statistic under the null."""
    if n_obs < 2:
        return (np.nan, np.nan)
    half = np.sqrt(2.0 / n_obs)
    return (1.0 - half, 1.0 + half)


def factor_mimicking_weights(exposures: pd.DataFrame, cap_weights: pd.Series
                             ) -> pd.DataFrame:
    """Weights of the portfolios implied by the cross-sectional regression.

    The Module B fit is f = (X'WX)^-1 X'W r, so row k of that leading matrix is
    the portfolio whose return *is* factor k's estimated return. These are the
    portfolios the factor covariance is a statement about, which makes them the
    natural place to start testing it.
    """
    x = exposures.to_numpy(dtype="float64")
    w = np.sqrt(cap_weights.reindex(exposures.index).fillna(0.0).to_numpy())
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() <= 0:
        return pd.DataFrame()
    w = w / w.mean()

    xtwx = x.T @ (x * w[:, None])
    try:
        inv = np.linalg.pinv(xtwx)
    except np.linalg.LinAlgError:
        return pd.DataFrame()
    weights = inv @ (x * w[:, None]).T
    return pd.DataFrame(weights.T, index=exposures.index, columns=exposures.columns)


def random_long_only(names: list[str], n_portfolios: int, rng: np.random.Generator,
                     max_names: int = 60) -> pd.DataFrame:
    """A set of random long-only portfolios over a subset of the live names."""
    if len(names) < 10:
        return pd.DataFrame()
    cols = {}
    for i in range(n_portfolios):
        k = min(max_names, len(names))
        picked = rng.choice(len(names), size=k, replace=False)
        raw = rng.dirichlet(np.ones(k))
        s = pd.Series(0.0, index=names)
        s.iloc[picked] = raw
        cols[f"rand_{i:03d}"] = s
    return pd.DataFrame(cols)


def predicted_vol(weights: pd.DataFrame, covariance: pd.DataFrame) -> pd.Series:
    """Model standard deviation for each column of ``weights``.

    Computed as the row-wise quadratic form rather than by materialising
    W' Sigma W, which would be a needless n_portfolios x n_portfolios matrix
    when only its diagonal is ever read.
    """
    names = [n for n in covariance.index if n in weights.index]
    w = weights.reindex(names).fillna(0.0).to_numpy(dtype="float64")
    sigma = covariance.loc[names, names].to_numpy(dtype="float64")
    var = np.einsum("ij,jk,ik->i", w.T, sigma, w.T)
    return pd.Series(np.sqrt(np.clip(var, 0.0, None)), index=weights.columns)


def realised_return(weights: pd.DataFrame, returns: pd.Series) -> pd.Series:
    """Return each portfolio actually earned, given weights fixed beforehand."""
    names = [n for n in weights.index if n in returns.index]
    r = returns.reindex(names).fillna(0.0).to_numpy(dtype="float64")
    w = weights.reindex(names).fillna(0.0).to_numpy(dtype="float64")
    return pd.Series(w.T @ r, index=weights.columns)


def bias_statistic(standardised: pd.DataFrame) -> pd.Series:
    """Standard deviation of the standardised returns, per portfolio."""
    return standardised.std(ddof=1)


def summarise(standardised: pd.DataFrame, family: str, estimator: str) -> dict:
    """Collapse a family of portfolios into one reportable row."""
    per_portfolio = bias_statistic(standardised)
    n_obs = int(standardised.notna().all(axis=1).sum())
    lo, hi = confidence_band(n_obs)
    inside = per_portfolio.between(lo, hi).mean() if np.isfinite(lo) else np.nan
    return {
        "estimator": estimator,
        "family": family,
        "n_portfolios": int(per_portfolio.notna().sum()),
        "n_obs": n_obs,
        "bias_mean": float(per_portfolio.mean()),
        "bias_median": float(per_portfolio.median()),
        "ci_low": lo,
        "ci_high": hi,
        "pct_inside_band": float(inside) if np.isfinite(inside) else np.nan,
    }


def rolling_bias(standardised: pd.Series, window: int) -> pd.Series:
    """Bias statistic through time for a single portfolio."""
    return standardised.rolling(window, min_periods=max(10, window // 2)).std(ddof=1)


def rolling_bias_pooled(standardised: pd.DataFrame, window: int) -> pd.Series:
    """Bias statistic through time, pooled across a family of portfolios.

    The tempting shortcut is to average the standardised returns across
    portfolios and take the rolling standard deviation of that average. It is
    wrong, and wrong in a flattering direction: averaging nineteen
    factor-mimicking portfolios diversifies away most of their variance, so the
    mean series has a standard deviation far below one and the model looks like
    it is wildly overstating risk. On this sample that shortcut reported a
    rolling bias near 0.13 for a family whose full-sample statistic is 0.94.

    The right pooled quantity is the second moment across both dimensions.
    Under the null each z has unit variance and zero mean, so the bias statistic
    is the square root of the mean of z squared, averaged over portfolios first
    and then over the window.
    """
    squared = (standardised ** 2).mean(axis=1)
    rolled = squared.rolling(window, min_periods=max(10, window // 2)).mean()
    return np.sqrt(rolled)


def run(weekly_dates: pd.DatetimeIndex, weekly_returns: pd.DataFrame,
        exposures_by_date, cap_weights: pd.DataFrame,
        covariance_fn, estimators: list[str],
        holdings: dict[str, pd.DataFrame] | None = None,
        verbose: bool = True) -> dict[str, object]:
    """Compute bias statistics across estimators, families and dates.

    ``covariance_fn(date)`` must return a mapping from estimator name to a
    covariance of weekly returns fitted on data through ``date``.
    ``holdings`` optionally supplies the realised minimum-variance book per
    estimator, as a weight panel indexed by rebalance date, which is forward
    filled between rebalances.
    """
    cfg = PARAMS["bias_tests"]
    rng = np.random.default_rng(cfg["random_seed"])
    records: dict[tuple[str, str], dict[pd.Timestamp, pd.Series]] = {}

    for i in range(1, len(weekly_dates)):
        prev, now = weekly_dates[i - 1], weekly_dates[i]
        bundle = covariance_fn(prev)
        if not bundle:
            continue
        available = [e for e in estimators if e in bundle]
        if not available:
            continue

        names = list(bundle[available[0]].index)
        rets = weekly_returns.loc[now].reindex(names)
        live = rets.notna()
        names = [n for n, ok in zip(names, live) if ok]
        if len(names) < 50:
            continue
        rets = rets.reindex(names).fillna(0.0)

        families: dict[str, pd.DataFrame] = {}
        if "factor_mimicking" in cfg["portfolio_types"]:
            x = exposures_by_date(prev, names)
            if x is not None and not x.empty:
                fmp = factor_mimicking_weights(x, cap_weights.loc[prev])
                if not fmp.empty:
                    families["factor_mimicking"] = fmp
        if "random" in cfg["portfolio_types"]:
            families["random"] = random_long_only(
                names, cfg["n_random_portfolios"], rng
            )
        if holdings and "min_variance" in cfg["portfolio_types"]:
            for est, panel in holdings.items():
                held = panel.loc[panel.index <= prev]
                if held.empty:
                    continue
                w = held.iloc[-1].dropna()
                w = w[w.index.isin(names)]
                if w.sum() <= 0:
                    continue
                families[f"min_variance::{est}"] = (w / w.sum()).to_frame("minvar")

        for est in available:
            cov = bundle[est]
            for family, weights in families.items():
                if family.startswith("min_variance::") and not family.endswith(est):
                    continue
                cols = [n for n in weights.index if n in cov.index]
                if len(cols) < 20:
                    continue
                sub = weights.reindex(cols).fillna(0.0)
                sigma = predicted_vol(sub, cov.loc[cols, cols])
                r = realised_return(sub, rets.reindex(cols))
                z = r / sigma.replace(0.0, np.nan)
                key = (est, "min_variance" if family.startswith("min_variance")
                       else family)
                records.setdefault(key, {})[now] = z

        if verbose and i % 100 == 0:
            print(f"  bias {now.date()} ({i}/{len(weekly_dates)})", flush=True)

    summary_rows, panels = [], {}
    for (est, family), by_date in records.items():
        frame = pd.DataFrame(by_date).T.sort_index()
        panels[f"{est}::{family}"] = frame
        summary_rows.append(summarise(frame, family, est))

    return {"summary": pd.DataFrame(summary_rows), "standardised": panels}
