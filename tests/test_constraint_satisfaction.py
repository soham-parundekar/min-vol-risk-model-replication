"""Every rebalance must satisfy every published bound, or say which one it relaxed.

The USMV constraint set is the thing being replicated. A portfolio that quietly
breaches the 1.5% cap, or holds a 2bp position the index could never publish, is
not a replication of anything. These tests read the saved weight panels and check
each bound directly rather than trusting the solver's status flag, because
"optimal" only means the solver was happy with the problem it was handed, which
is not the same as the problem being the right one.

One subtlety that cost a round of false failures. The optimiser normalises parent
weights over the covariance's name set, which is narrower than the full tradable
universe. Rebuilding the parent vector from the universe instead produces
slightly different weights and therefore slightly different tilts, which showed
up as breaches of 0.2523 against a 0.25 band. Those were artefacts of the
reconstruction, not of the portfolios. The tests now read
``parent_weights_used.parquet``, the vector the solver actually constrained.

Where a bound is genuinely breached, the diagnostics must record a relaxation
for that date, and the test checks the bound that was in force rather than the
published one. A silent breach fails; a logged one is an observation about how
binding MSCI's published rules turn out to be.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minvol import pipeline as pl  # noqa: E402
from minvol.config import PARAMS, PATHS  # noqa: E402

ESTIMATORS = ["structural", "sample", "ledoit_wolf", "pca"]
TOL = 1e-6
# Solver tolerance. CLARABEL and OSQP both return "optimal" at a residual well
# below this, and a handful of hard rebalances finish as "optimal_inaccurate".
SLACK = 2e-3


def _weights(estimator: str) -> pd.DataFrame | None:
    path = PATHS.processed / f"weights_usmv_{estimator}.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _diagnostics(estimator: str) -> pd.DataFrame | None:
    path = PATHS.processed / f"diag_usmv_{estimator}.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _parent_used() -> pd.DataFrame | None:
    path = PATHS.processed / "parent_weights_used.parquet"
    return pd.read_parquet(path) if path.is_file() else None


def _relaxations(diag: pd.DataFrame | None, date) -> dict:
    if diag is None or date not in diag.index:
        return {}
    raw = diag.loc[date, "relaxations"]
    if isinstance(raw, str):
        try:
            raw = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return {}
    return raw if isinstance(raw, dict) else {}


def _in_force(diag, date, name: str, default: float) -> float:
    """The bound actually applied on this date, relaxed or not."""
    return float(_relaxations(diag, date).get(name, default))


def _not_rebalanced(diag: pd.DataFrame | None, date) -> bool:
    if diag is None or date not in diag.index or "not_rebalanced" not in diag.columns:
        return False
    return bool(diag.loc[date, "not_rebalanced"])


available = [e for e in ESTIMATORS if _weights(e) is not None]
needs_runs = pytest.mark.skipif(
    not available, reason="run scripts/03_portfolios.py first"
)
needs_parent = pytest.mark.skipif(
    _parent_used() is None, reason="parent_weights_used.parquet is missing"
)


@needs_runs
@pytest.mark.parametrize("estimator", available or ["structural"])
def test_weights_are_long_only_and_fully_invested(estimator):
    weights = _weights(estimator)
    for date, row in weights.iterrows():
        w = row.dropna()
        assert (w >= -TOL).all(), f"{estimator} {date.date()}: negative weight"
        assert abs(w.sum() - 1.0) < 1e-6, (
            f"{estimator} {date.date()}: weights sum to {w.sum():.8f}"
        )


@needs_runs
@pytest.mark.parametrize("estimator", available or ["structural"])
def test_no_holding_between_zero_and_the_floor(estimator):
    """The semi-continuous rule: a position is zero or at least the floor in force."""
    base_floor = PARAMS["usmv"]["min_holding"]
    weights = _weights(estimator)
    diag = _diagnostics(estimator)
    for date, row in weights.iterrows():
        held = row.dropna()
        held = held[held > TOL]
        if held.empty or _not_rebalanced(diag, date):
            continue
        floor = _in_force(diag, date, "min_holding", base_floor)
        assert held.min() >= floor - 1e-9, (
            f"{estimator} {date.date()}: smallest holding {held.min():.6f} is "
            f"below the {floor:.4f} floor in force"
        )


@needs_runs
@needs_parent
@pytest.mark.parametrize("estimator", available or ["structural"])
def test_position_cap_respected(estimator):
    """No name above the lower of the cap in force and 20x its parent weight."""
    cfg = PARAMS["usmv"]
    parent_used = _parent_used()
    weights = _weights(estimator)
    diag = _diagnostics(estimator)

    for date, row in weights.iterrows():
        held = row.dropna()
        held = held[held > TOL]
        if held.empty or _not_rebalanced(diag, date) or date not in parent_used.index:
            continue
        b = parent_used.loc[date].reindex(held.index).fillna(0.0)
        max_abs = _in_force(diag, date, "max_weight", cfg["max_weight_abs"])
        cap = np.minimum(max_abs, cfg["max_weight_parent_multiple"] * b)
        breach = held[held > cap + SLACK]
        assert breach.empty, (
            f"{estimator} {date.date()}: {len(breach)} names above cap, "
            f"worst {breach.max():.5f}"
        )


@needs_runs
@pytest.mark.parametrize("estimator", available or ["structural"])
def test_turnover_within_cap(estimator):
    """Turnover against the drifted book, which is the quantity MSCI constrains.

    The reference portfolio is the existing index drifted to the rebalance date,
    not the target set six months earlier, so the recorded figure is the one the
    constraint applies to. Comparing target against target instead measures
    something the optimiser never bounded and reads as a breach on most dates.
    """
    base_cap = PARAMS["usmv"]["turnover_cap"]
    diag = _diagnostics(estimator)
    if diag is None or "turnover" not in diag.columns:
        pytest.skip("no turnover recorded")

    for date, row in diag.iterrows():
        turnover = row["turnover"]
        if not np.isfinite(turnover) or _not_rebalanced(diag, date):
            continue
        cap = _in_force(diag, date, "turnover_cap", base_cap)
        assert turnover <= cap + SLACK, (
            f"{estimator} {date.date()}: turnover {turnover:.4f} exceeds the "
            f"{cap:.2f} cap in force"
        )


@needs_runs
@needs_parent
@pytest.mark.parametrize("estimator", available or ["structural"])
def test_sector_band_respected(estimator):
    base_band = PARAMS["usmv"]["sector_band"]
    panels = pl.load_panels()
    parent_used = _parent_used()
    labels = panels.industry_labels
    weights = _weights(estimator)
    diag = _diagnostics(estimator)

    for date, row in weights.iterrows():
        held = row.dropna()
        held = held[held > TOL]
        if held.empty or _not_rebalanced(diag, date) or date not in parent_used.index:
            continue
        band = _in_force(diag, date, "sector_band", base_band)
        b = parent_used.loc[date].dropna()
        union = held.index.union(b.index)
        w = held.reindex(union).fillna(0.0)
        bb = b.reindex(union).fillna(0.0)
        lab = labels.reindex(union).fillna("Other")
        active = (w.groupby(lab).sum() - bb.groupby(lab).sum()).abs()
        assert active.max() <= band + SLACK, (
            f"{estimator} {date.date()}: sector active weight {active.max():.4f} "
            f"exceeds the {band} band in force"
        )


@needs_runs
@needs_parent
@pytest.mark.parametrize("estimator", available or ["structural"])
def test_style_band_is_never_relaxed(estimator):
    """MSCI's Appendix III loosens turnover, the floor, the cap and sectors. Never style.

    This is the one bound that holds on every rebalance, so it is asserted with
    no escape hatch.
    """
    band = PARAMS["usmv"]["style_band"]
    lag = PARAMS["usmv"].get("covariance_lag_months", 0)
    free = set(PARAMS["exposures"]["unconstrained_styles"])
    constrained = [s for s in PARAMS["exposures"]["style_factors"] if s not in free]

    panels = pl.load_panels()
    parent_used = _parent_used()
    weights = _weights(estimator)
    diag = _diagnostics(estimator)

    for date, row in weights.iterrows():
        held = row.dropna()
        held = held[held > TOL]
        if held.empty or _not_rebalanced(diag, date) or date not in parent_used.index:
            continue
        assert "style_band" not in _relaxations(diag, date), (
            f"{estimator} {date.date()}: the style band was relaxed, which the "
            f"published methodology never does"
        )
        asof = panels.asof_for_rebalance(date, lag)
        b = parent_used.loc[date].dropna()
        union = held.index.union(b.index)
        w = held.reindex(union).fillna(0.0).to_numpy()
        bb = b.reindex(union).fillna(0.0).to_numpy()
        for style in constrained:
            x = panels.exposures[style].loc[asof].reindex(union).fillna(0.0).to_numpy()
            tilt = abs(float(x @ (w - bb)))
            assert tilt <= band + SLACK, (
                f"{estimator} {date.date()}: {style} tilt {tilt:.4f} exceeds the "
                f"{band} band, which is never relaxed"
            )


def _median_tilt(estimator: str, style: str) -> float | None:
    """Median active exposure to one style across a run's rebalances."""
    lag = PARAMS["usmv"].get("covariance_lag_months", 0)
    panels = pl.load_panels()
    parent_used = _parent_used()
    weights = _weights(estimator)
    if weights is None or weights.empty or parent_used is None:
        return None

    tilts = []
    for date, row in weights.iterrows():
        held = row.dropna()
        held = held[held > TOL]
        if held.empty or date not in parent_used.index:
            continue
        asof = panels.asof_for_rebalance(date, lag)
        b = parent_used.loc[date].dropna()
        union = held.index.union(b.index)
        w = held.reindex(union).fillna(0.0).to_numpy()
        bb = b.reindex(union).fillna(0.0).to_numpy()
        x = panels.exposures[style].loc[asof].reindex(union).fillna(0.0).to_numpy()
        tilts.append(float(x @ (w - bb)))
    return float(np.median(tilts)) if tilts else None


@needs_runs
@needs_parent
def test_the_portfolio_tilts_away_from_volatility():
    """The point of the index: hold less volatility than the parent."""
    free = PARAMS["exposures"]["unconstrained_styles"][0]
    tilt = _median_tilt("structural", free)
    if tilt is None:
        pytest.skip("structural weights unavailable")
    assert tilt < -0.10, (
        f"median {free} tilt is {tilt:+.3f}; a minimum-variance portfolio should "
        f"sit clearly below the parent on the volatility factor"
    )


@needs_parent
def test_the_exemption_actually_lets_a_style_leave_the_band():
    """An exempt style must be free to go where a bounded one cannot.

    In the base specification beta is bounded and settles against its band at
    about -0.25, so residual volatility never has to travel far. Freeing beta,
    the reading the 2025 methodology supports, sends it to roughly -0.72. That
    gap is the evidence that the exemption does something, and it is the whole
    reason the variant is run.
    """
    band = PARAMS["usmv"]["style_band"]
    variant = PARAMS["exposures"].get("unconstrained_styles_variant") or []
    if _weights("structural_betafree") is None or "beta" not in variant:
        pytest.skip("the beta-free variant has not been run")

    bounded = _median_tilt("structural", "beta")
    freed = _median_tilt("structural_betafree", "beta")
    assert bounded is not None and freed is not None

    assert abs(bounded) <= band + SLACK, (
        f"beta tilt of {bounded:+.3f} should respect the {band} band when bounded"
    )
    assert freed < -band, (
        f"beta tilt of {freed:+.3f} should exceed the {band} band once exempt; "
        f"otherwise the exemption is doing nothing"
    )
