"""Module B: weekly factor returns from a constrained cross-sectional regression.

Each week the cross-section of stock returns is regressed on the exposures
known at the start of that week:

    r_i = f_mkt + sum_j X_ij^ind f_j^ind + sum_k X_ik^sty f_k^sty + u_i

fitted by weighted least squares with weights proportional to the square root
of market cap. Square root rather than cap itself: a cap-weighted fit is
effectively a regression on about thirty mega caps, and an equal-weighted fit
hands the estimate to the noisiest small names. The square root is the usual
compromise and it is what Barra uses.

**The timing.** Exposures come from the end of week t-1 and returns span t-1 to
t. Fitting this week's return on this week's exposures would use information
from inside the period being explained, and since several descriptors are built
from trailing returns, the regression would partly explain the return with
itself.

**The rank deficiency.** Industry dummies are 0/1 and every stock sits in
exactly one industry, so the industry columns sum to the intercept for every
row. The design matrix is rank deficient by one and the market factor is not
separately identified without a side condition.

The usual fix in textbooks is to drop one industry column. That works
numerically and quietly changes the meaning of everything: the intercept stops
being the market return and becomes the return of the omitted industry, and
each remaining industry coefficient becomes a difference against it. Comparing
such a "market factor" with anything Barra-shaped is then apples to oranges.

What this module does instead is impose the economically meaningful condition
that cap-weighted industry returns sum to zero,

    sum_j c_j f_j^ind = 0,   c_j = industry j's share of total market cap,

by reparametrising. Write f^ind = R g with R chosen so that c'R = 0, fit the
unconstrained regression in g, then map back. The intercept keeps its meaning
as the cap-weighted market return and each industry return reads as a deviation
from the market. ``tests/test_industry_return_identification.py`` checks the
constraint holds in the fitted output and that dropping a column instead gives
a different market factor.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PARAMS, PATHS

OTHER = "Other"


def weekly_returns(adj_close: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return over each weekly interval, indexed by the interval's end date."""
    px = adj_close.reindex(dates, method="ffill")
    return px.pct_change(fill_method=None)


def industry_panel(industries: pd.DataFrame, tickers: list[str]) -> pd.Series:
    """Ticker to industry label, with anything unmapped sent to the catch-all."""
    lookup = industries.set_index("ticker")["industry"]
    return pd.Series(
        [lookup.get(t, OTHER) for t in tickers], index=tickers, name="industry"
    )


def collapse_small_industries(labels: pd.Series, caps: pd.Series,
                              min_names: int) -> pd.Series:
    """Fold thinly populated industries into the catch-all for this week.

    An industry return estimated from three stocks is an average of three
    stocks, not an industry effect, and it makes the cross-sectional fit
    unstable in exactly the weeks where those names move together.
    """
    live = labels[caps.notna() & (caps > 0)]
    counts = live.value_counts()
    small = set(counts[counts < min_names].index)
    if not small:
        return labels
    return labels.where(~labels.isin(small), OTHER)


def build_constraint_basis(cap_by_industry: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Matrix R mapping K-1 free parameters to K industry returns with c'R = 0.

    The last industry absorbs the constraint: given the first K-1 returns, the
    final one is pinned at -(1/c_K) * sum_{j<K} c_j f_j. The industry with the
    largest cap weight is placed last so the division is by the biggest
    denominator available, which keeps the reparametrisation numerically calm.
    """
    order = cap_by_industry.sort_values().index.tolist()
    c = cap_by_industry.reindex(order).to_numpy(dtype="float64")
    k = len(order)
    if k < 2:
        return np.ones((1, 1)), order
    R = np.zeros((k, k - 1), dtype="float64")
    R[: k - 1, :] = np.eye(k - 1)
    R[k - 1, :] = -c[: k - 1] / c[k - 1]
    return R, order


def fit_week(returns: pd.Series, styles: pd.DataFrame, labels: pd.Series,
             caps: pd.Series, min_names: int
             ) -> tuple[pd.Series, pd.Series, float, int] | None:
    """Fit one weekly cross-section. Returns (factor returns, specific returns, R2, n)."""
    ok = returns.notna() & caps.notna() & (caps > 0) & styles.notna().all(axis=1)
    if ok.sum() < 50:
        return None

    r = returns[ok].astype("float64")
    X_sty = styles.loc[ok].astype("float64")
    cap = caps[ok].astype("float64")
    lab = collapse_small_industries(labels.loc[ok], cap, min_names)

    cap_by_ind = cap.groupby(lab).sum()
    cap_by_ind = cap_by_ind / cap_by_ind.sum()
    if len(cap_by_ind) < 2:
        return None
    R, ind_order = build_constraint_basis(cap_by_ind)

    D = pd.get_dummies(lab).reindex(columns=ind_order, fill_value=0).to_numpy(dtype="float64")
    design = np.column_stack([np.ones(len(r)), D @ R, X_sty.to_numpy()])

    w = np.sqrt(cap.to_numpy())
    w = w / w.mean()
    sw = np.sqrt(w)
    A = design * sw[:, None]
    b = r.to_numpy() * sw

    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    fitted = design @ coef
    resid = r.to_numpy() - fitted

    # Weighted R-squared against a weighted-mean benchmark, which is the right
    # null for a WLS fit.
    rbar = float((w * r.to_numpy()).sum() / w.sum())
    ss_res = float((w * resid ** 2).sum())
    ss_tot = float((w * (r.to_numpy() - rbar) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    n_ind = R.shape[1]
    f_mkt = coef[0]
    f_ind = R @ coef[1 : 1 + n_ind]
    f_sty = coef[1 + n_ind :]

    factors = pd.Series(
        [f_mkt, *f_ind, *f_sty],
        index=["market", *[f"ind_{i}" for i in ind_order], *X_sty.columns],
        dtype="float64",
    )
    specific = pd.Series(resid, index=r.index, dtype="float64")
    return factors, specific, r2, int(ok.sum())


def build(exposures: dict[str, pd.DataFrame], adj_close: pd.DataFrame,
          market_cap_weekly: pd.DataFrame, industries: pd.DataFrame,
          tradable: pd.DataFrame, cache: bool = True) -> dict[str, object]:
    """Run the weekly regression across the sample.

    Exposures are lagged one week relative to returns, so the factor return
    stamped at date t is earned over the week ending at t using exposures known
    at t-1.
    """
    PATHS.ensure()
    min_names = PARAMS["industries"]["min_names_per_industry"]
    style_names = PARAMS["exposures"]["style_factors"]

    dates = exposures[style_names[0]].index
    tickers = list(exposures[style_names[0]].columns)
    rets = weekly_returns(adj_close, dates)
    labels = industry_panel(industries, tickers)

    factor_rows: dict[pd.Timestamp, pd.Series] = {}
    specific_rows: dict[pd.Timestamp, pd.Series] = {}
    diag = []

    for i in range(1, len(dates)):
        t_prev, t = dates[i - 1], dates[i]
        styles = pd.DataFrame(
            {name: exposures[name].loc[t_prev] for name in style_names}
        )
        live = tradable.loc[t_prev] & tradable.loc[t]
        styles = styles.where(live)
        caps = market_cap_weekly.loc[t_prev].where(live)

        fit = fit_week(rets.loc[t], styles, labels, caps, min_names)
        if fit is None:
            diag.append({"date": t, "n": 0, "r2": np.nan, "n_industries": 0})
            continue
        factors, specific, r2, n = fit
        factor_rows[t] = factors
        specific_rows[t] = specific
        diag.append({
            "date": t, "n": n, "r2": r2,
            "n_industries": int(sum(k.startswith("ind_") for k in factors.index)),
        })

    factor_returns = pd.DataFrame(factor_rows).T.sort_index()
    factor_returns.index.name = "date"
    specific_returns = pd.DataFrame(specific_rows).T.sort_index()
    specific_returns.index.name = "date"
    diagnostics = pd.DataFrame(diag).set_index("date")

    out = {
        "factor_returns": factor_returns,
        "specific_returns": specific_returns,
        "diagnostics": diagnostics,
        "industry_labels": labels,
        "weekly_returns": rets,
    }
    if cache:
        factor_returns.to_parquet(PATHS.processed / "factor_returns.parquet")
        specific_returns.to_parquet(PATHS.processed / "specific_returns.parquet")
        diagnostics.to_parquet(PATHS.processed / "regression_diagnostics.parquet")
        labels.to_frame().to_parquet(PATHS.processed / "industry_labels.parquet")
        rets.to_parquet(PATHS.processed / "weekly_returns.parquet")
    return out
