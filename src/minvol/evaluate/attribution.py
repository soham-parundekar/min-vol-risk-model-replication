"""Module G, part two: where each portfolio's risk actually comes from.

Marginal contribution to risk is the derivative of portfolio volatility with
respect to each holding,

    MCR = Sigma w / sqrt(w' Sigma w),

and because volatility is homogeneous of degree one in the weights, the
position-level contributions ``w * MCR`` add up exactly to portfolio
volatility. That exact additivity is what makes the decomposition worth doing:
the numbers are shares of a whole, not an approximation that nearly adds up.

With a structural model the same total splits a second way, into the part
driven by common factors and the part that is name-specific:

    w' Sigma w = (X'w)' F (X'w) + w' Delta w

and the first term splits again across factors, so a single portfolio can be
read as a set of factor and industry bets plus a residual.

This is the section that makes the headline comparison concrete. A minimum
variance optimiser run without sector constraints does not spread itself
thoughtfully across the market; it finds whichever industries happened to be
quiet and sits on them, historically utilities, staples and real estate. That
shows up here as a large positive industry contribution rather than as anything
visible in a weight list. MSCI's sector band exists to stop it. SPLV has no
such band, so the comparison between the two decompositions is a direct reading
of what the constraint buys.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def marginal_contributions(weights: pd.Series, covariance: pd.DataFrame
                           ) -> pd.DataFrame:
    """Per-name marginal and total contribution to portfolio volatility."""
    names = [n for n in covariance.index if n in weights.index]
    w = weights.reindex(names).fillna(0.0).to_numpy(dtype="float64")
    sigma = covariance.loc[names, names].to_numpy(dtype="float64")

    variance = float(w @ sigma @ w)
    vol = np.sqrt(max(variance, 0.0))
    if vol <= 0:
        return pd.DataFrame(index=names)
    mcr = (sigma @ w) / vol
    return pd.DataFrame({
        "weight": w,
        "mcr": mcr,
        "contribution": w * mcr,
        "pct_of_risk": (w * mcr) / vol,
    }, index=names)


def factor_risk_decomposition(weights: pd.Series, model) -> dict[str, object]:
    """Split portfolio variance into per-factor contributions and specific risk.

    ``model`` is a :class:`minvol.risk.covariance.FactorRiskModel`.
    """
    names = [n for n in model.exposures.index if n in weights.index]
    w = weights.reindex(names).fillna(0.0).to_numpy(dtype="float64")
    x = model.exposures.loc[names].to_numpy(dtype="float64")
    f = model.factor_cov.to_numpy(dtype="float64")
    delta = model.specific_var.reindex(names).fillna(0.0).to_numpy(dtype="float64")

    exposure = x.T @ w
    factor_var = float(exposure @ f @ exposure)
    specific_var = float((w ** 2) @ delta)
    total_var = factor_var + specific_var
    vol = np.sqrt(max(total_var, 0.0))

    per_factor = exposure * (f @ exposure)
    contributions = pd.Series(per_factor, index=model.factor_cov.index)

    groups = {
        "market": contributions.reindex(["market"]).sum(),
        "industry": contributions[[c for c in contributions.index
                                   if str(c).startswith("ind_")]].sum(),
        "style": contributions[[c for c in contributions.index
                                if c != "market" and not str(c).startswith("ind_")]].sum(),
    }

    return {
        "portfolio_vol": vol,
        "total_variance": total_var,
        "factor_variance": factor_var,
        "specific_variance": specific_var,
        "pct_factor": factor_var / total_var if total_var > 0 else np.nan,
        "pct_specific": specific_var / total_var if total_var > 0 else np.nan,
        "exposures": pd.Series(exposure, index=model.factor_cov.index),
        "factor_contributions": contributions,
        "group_contributions": pd.Series(groups),
    }


def sector_weights(weights: pd.Series, labels: pd.Series) -> pd.Series:
    """Portfolio weight by industry group."""
    lab = labels.reindex(weights.index).fillna("Other")
    return weights.groupby(lab).sum().sort_values(ascending=False)


def active_sector_weights(weights: pd.Series, parent: pd.Series,
                          labels: pd.Series) -> pd.DataFrame:
    """Portfolio, parent and active weight per industry group."""
    union = weights.index.union(parent.index)
    w = weights.reindex(union).fillna(0.0)
    b = parent.reindex(union).fillna(0.0)
    lab = labels.reindex(union).fillna("Other")
    out = pd.DataFrame({
        "portfolio": w.groupby(lab).sum(),
        "parent": b.groupby(lab).sum(),
    })
    out["active"] = out["portfolio"] - out["parent"]
    return out.sort_values("active", ascending=False)


def concentration(weights: pd.Series) -> dict[str, float]:
    """How concentrated a weight vector is, by a few common measures."""
    w = weights[weights > 0].sort_values(ascending=False)
    if w.empty:
        return {}
    return {
        "n_holdings": int(len(w)),
        "max_weight": float(w.iloc[0]),
        "top10_weight": float(w.head(10).sum()),
        "herfindahl": float((w ** 2).sum()),
        "effective_n": float(1.0 / (w ** 2).sum()),
    }
