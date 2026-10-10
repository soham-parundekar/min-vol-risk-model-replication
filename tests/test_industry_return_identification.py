"""The industry dummies are rank deficient, and how you fix that changes the answer.

Every stock sits in exactly one industry, so the industry columns sum to the
intercept and the design matrix loses a rank. Something has to pin the system
down, and the two usual choices are not equivalent:

* drop one industry column, which makes the intercept the omitted industry's
  return and every other coefficient a difference against it;
* require cap-weighted industry returns to sum to zero, which leaves the
  intercept as the cap-weighted market return.

Only the second gives a market factor that means what a Barra-style model means
by one. The difference is not a rounding detail: on a single week in the sample
the two parametrisations differ by more than half a percent on the market
factor alone.

These tests check that the constraint is imposed exactly, that the market
factor it produces tracks the actual cap-weighted market, and that the fitted
values are identical either way, which is the point that makes the choice about
interpretation rather than fit.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minvol.config import PARAMS, PATHS  # noqa: E402
from minvol.factors import returns as factor_returns  # noqa: E402

FACTOR_PATH = PATHS.processed / "factor_returns.parquet"
needs_data = pytest.mark.skipif(
    not FACTOR_PATH.is_file(), reason="run scripts/02_build_factors.py first"
)


def _toy_cross_section(seed: int = 0):
    """A small synthetic cross-section with a known industry structure."""
    rng = np.random.default_rng(seed)
    n = 200
    names = [f"S{i:03d}" for i in range(n)]
    industries = pd.Series(
        rng.choice(["Tech", "Fin", "Util", "Health"], size=n), index=names
    )
    caps = pd.Series(rng.lognormal(mean=10, sigma=1.5, size=n), index=names)
    styles = pd.DataFrame(
        rng.normal(size=(n, 2)), index=names, columns=["size", "beta"]
    )
    rets = pd.Series(rng.normal(scale=0.03, size=n), index=names)
    return rets, styles, industries, caps


def test_constraint_basis_annihilates_the_cap_weights():
    """c'R = 0 by construction, so any g maps to industry returns summing to zero."""
    caps = pd.Series({"A": 0.4, "B": 0.35, "C": 0.25})
    basis, order = factor_returns.build_constraint_basis(caps)
    c = caps.reindex(order).to_numpy()
    assert np.allclose(c @ basis, 0.0, atol=1e-12)
    assert basis.shape == (3, 2)


def test_fitted_industry_returns_sum_to_zero_under_cap_weights():
    rets, styles, industries, caps = _toy_cross_section()
    out = factor_returns.fit_week(rets, styles, industries, caps, min_names=5)
    assert out is not None
    factors, _, _, _ = out

    cap_share = caps.groupby(industries).sum()
    cap_share = cap_share / cap_share.sum()
    total = sum(cap_share[k] * factors[f"ind_{k}"] for k in cap_share.index)
    assert abs(total) < 1e-10, f"cap-weighted industry returns sum to {total:.3e}"


def test_dropping_a_column_gives_the_same_fit_but_a_different_market_factor():
    """Same residuals, different intercept. That is why the choice matters."""
    rets, styles, industries, caps = _toy_cross_section(seed=3)
    constrained = factor_returns.fit_week(rets, styles, industries, caps, min_names=5)
    assert constrained is not None
    factors, specific, _, _ = constrained

    dummies = pd.get_dummies(industries).iloc[:, 1:]
    design = np.column_stack([
        np.ones(len(rets)), dummies.to_numpy(dtype="float64"),
        styles.to_numpy(dtype="float64"),
    ])
    w = np.sqrt(caps.to_numpy())
    w = w / w.mean()
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], rets.to_numpy() * sw, rcond=None)
    dropped_resid = rets.to_numpy() - design @ coef

    # The two parametrisations span the same column space, so they fit equally.
    assert np.allclose(np.sort(dropped_resid), np.sort(specific.to_numpy()), atol=1e-8)
    # But the intercept is a different quantity.
    assert abs(coef[0] - factors["market"]) > 1e-6


@needs_data
def test_live_market_factor_tracks_the_rebuilt_parent_index():
    """On real data the market factor should be the cap-weighted market return."""
    factors = pd.read_parquet(FACTOR_PATH)
    parent = pd.read_parquet(
        PATHS.processed / "parent_index_returns.parquet")["parent"]
    weekly_parent = (1.0 + parent).resample("W-FRI").prod() - 1.0

    joined = pd.concat(
        [factors["market"].rename("market"), weekly_parent.rename("parent")],
        axis=1, sort=True,
    ).dropna()
    assert len(joined) > 500
    corr = joined["market"].corr(joined["parent"])
    assert corr > 0.98, f"market factor correlates only {corr:.4f} with the parent index"

    gap = abs(joined["market"].mean() - joined["parent"].mean()) * 52
    assert gap < 0.03, f"annualised market factor differs from parent by {gap:.3%}"


@needs_data
def test_industry_columns_cover_the_configured_scheme():
    factors = pd.read_parquet(FACTOR_PATH)
    industry_cols = [c for c in factors.columns if c.startswith("ind_")]
    styles = PARAMS["exposures"]["style_factors"]
    assert len(industry_cols) >= 8, "too few industries identified"
    assert "market" in factors.columns
    for style in styles:
        assert style in factors.columns, f"missing style factor {style}"


@needs_data
def test_cross_sectional_fit_is_in_a_sane_range():
    """A weekly equity factor model explains a meaningful but far from total share."""
    diag = pd.read_parquet(PATHS.processed / "regression_diagnostics.parquet")
    mean_r2 = diag["r2"].mean()
    assert 0.10 < mean_r2 < 0.75, f"mean cross-sectional R2 of {mean_r2:.3f} is implausible"
    assert diag["n"].median() > 200, "too few names per cross-section"
