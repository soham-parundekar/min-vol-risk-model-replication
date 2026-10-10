"""Module G, part three: does low-volatility alpha survive betting-against-beta?

Regress a portfolio's excess return on the Fama-French five factors plus
momentum and a positive intercept usually appears. Taken at face value that
says a low-volatility portfolio earns a return the standard factors cannot
explain, which would be an anomaly in need of a story.

Frazzini and Pedersen (2014) supply one. Investors who want more return but
cannot or will not borrow buy high-beta stocks instead of levering low-beta
ones. That bids high-beta prices up and leaves low-beta stocks cheap, so a
portfolio that is long low beta and short high beta, funded to be beta neutral,
earns a premium. Their BAB factor is that portfolio. If the low-volatility
intercept is compensation for the same leverage constraint, adding BAB to the
right-hand side should absorb it.

So the test is a comparison of two regressions, not one:

    r_p - r_f = a + b1 MKT + b2 SMB + b3 HML + b4 RMW + b5 CMA + b6 MOM + e
    r_p - r_f = a + ... + b6 MOM + b7 BAB + e

and the quantity of interest is what happens to ``a`` between them. A large
drop with a significant BAB loading supports the leverage-constraint reading.
An intercept that survives does not.

Standard errors are Newey-West. Daily portfolio returns are autocorrelated and
heteroskedastic, and ordinary OLS standard errors would understate the
uncertainty on exactly the coefficient the whole question turns on. The lag
length follows the usual rule of thumb, roughly four times (T/100)^(2/9).

A caveat that belongs with the result rather than under it: a surviving
intercept is evidence that these particular factors do not explain the return.
It is not evidence of mispricing. The honest reading is that this is a horse
race between named models, and the most a regression can say is which of them
accounts for the return.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BASE_FACTORS = ["MKT_RF", "SMB", "HML", "RMW", "CMA", "MOM"]


def newey_west_lags(n_obs: int) -> int:
    """Automatic bandwidth, the standard 4*(T/100)^(2/9) rule."""
    if n_obs < 10:
        return 0
    return int(np.floor(4.0 * (n_obs / 100.0) ** (2.0 / 9.0)))


def run_regression(portfolio_returns: pd.Series, factors: pd.DataFrame,
                   factor_names: list[str], rf_col: str = "RF",
                   lags: int | None = None) -> dict[str, object]:
    """Regress excess portfolio return on the named factors, Newey-West errors."""
    import statsmodels.api as sm

    cols = [c for c in factor_names if c in factors.columns]
    data = pd.concat([portfolio_returns.rename("r"), factors[[*cols, rf_col]]],
                     axis=1).dropna()
    if len(data) < 60:
        return {}

    y = data["r"] - data[rf_col]
    x = sm.add_constant(data[cols])
    lags = newey_west_lags(len(data)) if lags is None else lags
    fit = sm.OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": lags})

    ppy = 252
    out = {
        "n_obs": int(len(data)),
        "start": str(data.index.min().date()),
        "end": str(data.index.max().date()),
        "nw_lags": lags,
        "alpha_daily": float(fit.params["const"]),
        "alpha_ann": float(fit.params["const"] * ppy),
        "alpha_t": float(fit.tvalues["const"]),
        "alpha_p": float(fit.pvalues["const"]),
        "r_squared": float(fit.rsquared),
        "adj_r_squared": float(fit.rsquared_adj),
    }
    for c in cols:
        out[f"beta_{c}"] = float(fit.params[c])
        out[f"t_{c}"] = float(fit.tvalues[c])
    return out


def alpha_with_and_without_bab(portfolio_returns: pd.Series, factors: pd.DataFrame,
                               label: str = "") -> pd.DataFrame:
    """Both regressions side by side, so the intercept can be read across them."""
    rows = []
    base = run_regression(portfolio_returns, factors, BASE_FACTORS)
    if base:
        base["model"] = "FF5 + MOM"
        base["portfolio"] = label
        rows.append(base)
    if "BAB" in factors.columns:
        with_bab = run_regression(portfolio_returns, factors, [*BASE_FACTORS, "BAB"])
        if with_bab:
            with_bab["model"] = "FF5 + MOM + BAB"
            with_bab["portfolio"] = label
            rows.append(with_bab)
    if not rows:
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    lead = ["portfolio", "model", "n_obs", "start", "end", "alpha_ann",
            "alpha_t", "alpha_p", "r_squared"]
    ordered = [c for c in lead if c in frame.columns]
    return frame.loc[:, ordered + [c for c in frame.columns if c not in ordered]]


#: Below this annualised intercept, a percentage change in alpha is not worth
#: quoting: dividing by something indistinguishable from zero produces a large
#: number that means nothing.
ALPHA_MATERIALITY = 0.005


def alpha_change(table: pd.DataFrame) -> dict[str, dict[str, float]]:
    """What happens to the intercept when BAB enters the regression.

    Reported as a change in percentage points rather than as a percentage of
    the original. Most of these portfolios have a base intercept that is
    already statistically indistinguishable from zero, and "BAB absorbed 400%
    of the alpha" is an artefact of dividing by a number near zero, not a
    finding. The share absorbed is filled in only where the base intercept is
    large enough for the ratio to carry meaning.
    """
    out: dict[str, dict[str, float]] = {}
    for label, grp in table.groupby("portfolio"):
        base = grp[grp["model"] == "FF5 + MOM"]
        bab = grp[grp["model"] == "FF5 + MOM + BAB"]
        if base.empty or bab.empty:
            continue
        a0 = float(base["alpha_ann"].iloc[0])
        a1 = float(bab["alpha_ann"].iloc[0])
        out[label] = {
            "alpha_ann_base": a0,
            "alpha_ann_with_bab": a1,
            "change_pp": (a1 - a0) * 100.0,
            "absorbed_pct": ((a0 - a1) / a0 * 100.0
                             if abs(a0) >= ALPHA_MATERIALITY else np.nan),
            "t_base": float(base["alpha_t"].iloc[0]),
            "t_with_bab": float(bab["alpha_t"].iloc[0]),
            "beta_bab": float(bab.get("beta_BAB", pd.Series([np.nan])).iloc[0]),
            "t_bab": float(bab.get("t_BAB", pd.Series([np.nan])).iloc[0]),
            "base_alpha_material": abs(a0) >= ALPHA_MATERIALITY,
        }
    return out
