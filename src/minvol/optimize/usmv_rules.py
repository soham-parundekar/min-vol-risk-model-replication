"""Module E: the USMV rule, as a constrained minimum-variance program.

The optimisation MSCI describes, solved every May and November:

    minimise    w' Sigma w
    subject to  1'w = 1
                0 <= w_i <= min(1.5%, 20 * b_i)
                |S'(w - b)| <= 5%      per sector
                |X'(w - b)| <= 0.25    per style, volatility excepted
                0.5 * ||w - w_prev||_1 <= 10%

with ``b`` the parent index weights. Every bound here is transcribed from the
MSCI Minimum Volatility Indexes methodology (REF-001), not inferred: the
document gives the 1.5%-or-20x cap, the 0.05% minimum holding, the plus or
minus 5% sector band, the plus or minus 0.25 standard deviation style band with
the volatility factor explicitly exempted, and the 10% one-way turnover cap.

Two features make this harder than it looks on the page.

**The 5bp floor is not a bound.** A holding is either zero or at least 0.05%;
nothing in between is allowed. That is a semi-continuous variable and it makes
the program mixed-integer, not quadratic. Solved as a plain QP you get a long
tail of 0.003% positions that no index provider would ever publish. The
treatment here is the iterative one index providers themselves use: solve the
relaxed problem, delete everything under the floor, re-solve on the survivors,
repeat. It is a heuristic and it is called one.

**The constraint set can be infeasible**, and MSCI publishes what it does about
that. Appendix III of the methodology relaxes turnover first, in five-point
steps to a ceiling of 30%, then walks the minimum holding down to 0.01%, and
only then loosens the weight cap and sector band together. Two details there
are easy to get wrong by intuition. The style band is never relaxed at any
stage. And when the whole sequence still fails, the index is *not rebalanced*
that review rather than rebalanced on looser terms, which this module
reproduces by carrying the previous weights forward.

Every rebalance records which rung of that ladder it needed, so the report can
say how often the published rule could actually be met.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, replace

import cvxpy as cp
import numpy as np
import pandas as pd

from ..config import PARAMS
from ..risk.covariance import nearest_psd


@dataclass(frozen=True)
class ConstraintSet:
    """One rung of the relaxation ladder."""

    max_weight_abs: float
    max_weight_parent_multiple: float
    min_holding: float
    sector_band: float
    style_band: float
    turnover_cap: float | None

    def describe(self, base: "ConstraintSet") -> dict[str, float]:
        """What this rung changed relative to the published rule."""
        out: dict[str, float] = {}
        if self.turnover_cap != base.turnover_cap:
            out["turnover_cap"] = self.turnover_cap
        if self.min_holding != base.min_holding:
            out["min_holding"] = self.min_holding
        if self.max_weight_abs != base.max_weight_abs:
            out["max_weight"] = self.max_weight_abs
        if self.sector_band != base.sector_band:
            out["sector_band"] = self.sector_band
        return out


def base_constraints() -> ConstraintSet:
    """The published rule, with no relaxation."""
    cfg = PARAMS["usmv"]
    return ConstraintSet(
        max_weight_abs=cfg["max_weight_abs"],
        max_weight_parent_multiple=cfg["max_weight_parent_multiple"],
        min_holding=cfg["min_holding"],
        sector_band=cfg["sector_band"],
        style_band=cfg["style_band"],
        turnover_cap=cfg["turnover_cap"],
    )


def constraint_ladder() -> list[ConstraintSet]:
    """The full relaxation sequence from MSCI Appendix III, rung by rung."""
    cfg = PARAMS["usmv"].get("relaxation", {})
    base = base_constraints()
    rungs = [base]

    current = base
    for cap in cfg.get("turnover_steps", []):
        current = replace(current, turnover_cap=float(cap))
        rungs.append(current)
    for floor in cfg.get("min_holding_steps", []):
        current = replace(current, min_holding=float(floor))
        rungs.append(current)
    for step in cfg.get("cap_sector_steps", []):
        current = replace(current, max_weight_abs=float(step["max_weight"]),
                          sector_band=float(step["sector_band"]))
        rungs.append(current)
    return rungs


def position_caps(parent: pd.Series, constraints: ConstraintSet) -> pd.Series:
    """Per-name upper bound: the lower of the absolute cap and a multiple of parent weight.

    The parent-multiple leg is what stops the optimiser taking a 1.5% position
    in a company that is 0.01% of the index. A minimum-variance objective has no
    opinion about liquidity, and without this it would happily build the
    portfolio out of the smallest, quietest names available.
    """
    return np.minimum(constraints.max_weight_abs,
                      constraints.max_weight_parent_multiple * parent.clip(lower=0.0))


def _solve(sigma: np.ndarray, caps: np.ndarray, parent: np.ndarray,
           sector: np.ndarray | None, style: np.ndarray | None,
           prev: np.ndarray | None, constraints: ConstraintSet,
           turnover_offset: float = 0.0) -> tuple[np.ndarray | None, str]:
    """One quadratic program. Returns weights and the solver status."""
    n = len(caps)
    w = cp.Variable(n, nonneg=True)
    cons = [cp.sum(w) == 1, w <= caps]

    if sector is not None and sector.size:
        active = sector.T @ (w - parent)
        cons += [active <= constraints.sector_band, active >= -constraints.sector_band]
    if style is not None and style.size:
        tilt = style.T @ (w - parent)
        cons += [tilt <= constraints.style_band, tilt >= -constraints.style_band]
    if prev is not None and constraints.turnover_cap is not None:
        budget = constraints.turnover_cap - turnover_offset
        if budget <= 0:
            # Names that left the index already consume the whole budget; the
            # only feasible reading is that the remaining book cannot move.
            budget = 1e-6
        cons.append(0.5 * cp.norm1(w - prev) <= budget)

    problem = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(sigma))), cons)
    # Iteration caps keep one hard rebalance from stalling the run. A
    # near-optimal minimum-variance portfolio is acceptable here; the
    # constraint tests verify the result independently of what the solver
    # called its status.
    attempts = (
        (cp.CLARABEL, {"max_iter": 2000}),
        (cp.OSQP, {"max_iter": 20000, "eps_abs": 1e-7, "eps_rel": 1e-7}),
        (cp.SCS, {"max_iters": 20000}),
    )
    status = "failed"
    for solver, opts in attempts:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem.solve(solver=solver, **opts)
        except (cp.SolverError, ValueError, TypeError):
            continue
        status = problem.status or status
        if status in ("optimal", "optimal_inaccurate") and w.value is not None:
            return np.clip(np.asarray(w.value, dtype="float64"), 0.0, None), status
    return None, status


def _semicontinuous(sigma, caps, parent, sector, style, prev,
                    constraints: ConstraintSet, max_passes: int,
                    turnover_offset: float = 0.0
                    ) -> tuple[np.ndarray | None, str, int, int]:
    """Solve, drop holdings under the floor, re-solve, until none are left.

    Offenders are removed smallest first and only half at a time. Deleting
    every sub-floor name at once repeatedly overshot in testing, emptying the
    book and turning a feasible rebalance into an infeasible one.
    """
    working = caps.copy()
    dropped = 0
    status = "not_attempted"
    w = None
    for attempt in range(1, max_passes + 1):
        w, status = _solve(sigma, working, parent, sector, style, prev,
                           constraints, turnover_offset)
        if w is None:
            return None, status, attempt, dropped

        offending = (w > 1e-9) & (w < constraints.min_holding)
        if not offending.any():
            w = np.where(w > 1e-9, w, 0.0)
            total = w.sum()
            return (w / total if total > 0 else w), status, attempt, dropped

        order = np.argsort(np.where(offending, w, np.inf))
        kill = order[: max(1, int(offending.sum() * 0.5))]
        working[kill] = 0.0
        dropped += len(kill)
        if (working > 0).sum() < 60:
            return None, "exhausted_universe", attempt, dropped
    return None, "floor_not_converged", max_passes, dropped


def solve_rebalance(sigma, parent: pd.Series, sector, style, prev,
                    turnover_offset: float = 0.0, max_passes: int | None = None
                    ) -> tuple[np.ndarray | None, str, int, ConstraintSet, int, int]:
    """Walk the relaxation ladder until a rung produces a feasible portfolio."""
    max_passes = max_passes or PARAMS["usmv"]["semicontinuous_max_passes"]
    ladder = constraint_ladder()
    parent_arr = parent.to_numpy(dtype="float64")
    status = "not_attempted"

    # Rungs are walked in order, including when the failure was the floor loop
    # rather than an infeasible program. Skipping ahead to the rungs that lower
    # the floor would be faster and would not be MSCI's ladder: a looser
    # turnover bound gives the optimiser more room, and more room can be enough
    # for the floor heuristic to settle at a rung that leaves the published 5bp
    # minimum intact. Fidelity wins over the saved solves.
    for rung, constraints in enumerate(ladder):
        caps = position_caps(parent, constraints).to_numpy(dtype="float64")
        caps = np.where(caps < constraints.min_holding, 0.0, caps)
        if caps.sum() < 1.0:
            # The cap structure cannot fund a fully invested portfolio, which
            # happens when the parent is very concentrated. Fall back to the
            # absolute cap so the rung is at least attempted.
            caps = np.full_like(caps, constraints.max_weight_abs)

        w, status, passes, dropped = _semicontinuous(
            sigma, caps, parent_arr, sector, style, prev, constraints,
            max_passes, turnover_offset,
        )
        if w is not None:
            return w, status, rung, constraints, passes, dropped
    return None, status, len(ladder) - 1, ladder[-1], 0, 0


def _sector_matrix(labels: pd.Series, names: list[str]) -> np.ndarray:
    lab = labels.reindex(names).fillna("Other")
    return pd.get_dummies(lab).to_numpy(dtype="float64")


def drift_weights(weights: pd.Series, daily_returns: pd.DataFrame,
                  start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Carry a weight vector forward with the market, with no trading.

    MSCI measures turnover against the "Initial Portfolio", which it defines as
    the current index weights at the close of the rebalancing date, before the
    rebalance. That is the *drifted* book, not the weights set six months
    earlier: a holding that doubled is a bigger share of the portfolio without
    anyone having traded it.

    Constraining against the stale target instead makes the 10% cap bind on the
    wrong quantity. It also understates what the index actually trades, because
    the drift has to be traded back before any deliberate change is made.
    """
    if weights.empty:
        return weights
    window = daily_returns.loc[(daily_returns.index > start)
                               & (daily_returns.index <= end)]
    if window.empty:
        return weights
    growth = (1.0 + window.reindex(columns=weights.index).fillna(0.0)).prod()
    drifted = weights * growth
    total = drifted.sum()
    return drifted / total if total > 0 else weights


def run(dates: pd.DatetimeIndex, estimator: str, covariances: dict,
        parent_weights: pd.DataFrame, industry_labels: pd.Series,
        exposures: dict[str, pd.DataFrame], verbose: bool = True,
        free: list[str] | None = None,
        daily_returns: pd.DataFrame | None = None
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the USMV rule across every rebalance date for one covariance estimator.

    ``covariances`` maps a rebalance date to the output of
    :func:`minvol.risk.covariance.estimate_all`, including the ``asof`` date the
    estimate was fitted to. Returns the weight panel and a per-rebalance
    diagnostic table.
    """
    base = base_constraints()
    free_styles = set(free or PARAMS["exposures"]["unconstrained_styles"])
    constrained_styles = [s for s in PARAMS["exposures"]["style_factors"]
                          if s not in free_styles]

    rows: dict[pd.Timestamp, pd.Series] = {}
    diagnostics: list[dict] = []
    prev_weights: pd.Series | None = None   # target set at the last rebalance
    prev_date: pd.Timestamp | None = None

    for date in dates:
        bundle = covariances.get(date)
        if not bundle or estimator not in bundle:
            continue
        asof = bundle.get("asof")
        if asof is None:
            continue

        sigma_df = bundle[estimator]
        names = [n for n in sigma_df.index if n in parent_weights.columns]
        if len(names) < 60:
            continue

        parent = parent_weights.loc[asof, names].astype("float64").fillna(0.0)
        if parent.sum() <= 0:
            continue
        parent = parent / parent.sum()

        sigma = nearest_psd(sigma_df.loc[names, names]).to_numpy()
        sector = _sector_matrix(industry_labels, names)
        style = np.column_stack([
            exposures[s].loc[asof, names].fillna(0.0).to_numpy()
            for s in constrained_styles
        ])

        # Drift the existing book up to the rebalance date; that drifted
        # portfolio is what the turnover cap is measured against.
        held = prev_weights
        if held is not None and daily_returns is not None and prev_date is not None:
            held = drift_weights(held, daily_returns, prev_date, date)

        prev = None
        offset = 0.0
        if held is not None:
            prev = held.reindex(names).fillna(0.0).to_numpy()
            # Positions in names no longer investable must be sold, and that
            # sale is turnover. It is a constant, so it comes out of the budget
            # rather than into the optimisation.
            gone = held.drop(index=[n for n in names if n in held.index],
                             errors="ignore")
            offset = 0.5 * float(gone.abs().sum())

        w, status, rung, used, passes, dropped = solve_rebalance(
            sigma, parent, sector, style, prev, offset,
        )

        if w is None:
            # MSCI's rule when the ladder is exhausted: do not rebalance.
            if prev_weights is not None:
                rows[date] = (held if held is not None else prev_weights).copy()
                prev_weights = rows[date]
                prev_date = date
            diagnostics.append({
                "date": date, "asof": asof, "status": "not_rebalanced",
                "n_holdings": int((prev_weights > 0).sum()) if prev_weights is not None else 0,
                "turnover": 0.0, "predicted_vol": np.nan, "passes": passes,
                "dropped_below_floor": dropped, "ladder_rung": rung,
                "relaxations": {}, "not_rebalanced": True,
            })
            if verbose:
                print(f"  {date.date()} {estimator}: NOT REBALANCED ({status})",
                      flush=True)
            continue

        weights = pd.Series(w, index=names)
        realised = np.nan
        if held is not None:
            union = weights.index.union(held.index)
            realised = 0.5 * float(
                (weights.reindex(union).fillna(0.0)
                 - held.reindex(union).fillna(0.0)).abs().sum()
            )
        predicted_vol = float(np.sqrt(max(w @ sigma @ w, 0.0)))

        rows[date] = weights
        diagnostics.append({
            "date": date, "asof": asof, "status": status,
            "n_holdings": int((weights > 0).sum()),
            "turnover": realised, "predicted_vol": predicted_vol,
            "passes": passes, "dropped_below_floor": dropped,
            "ladder_rung": rung, "relaxations": used.describe(base),
            "not_rebalanced": False,
        })
        prev_weights = weights
        prev_date = date
        if verbose:
            tag = f" rung={rung} {used.describe(base)}" if rung else ""
            print(f"  {date.date()} {estimator}: {int((weights > 0).sum()):3d} names, "
                  f"turnover={realised if np.isfinite(realised) else 0:.3f}, "
                  f"vol={predicted_vol * np.sqrt(52) * 100:.2f}%{tag}", flush=True)

    panel = pd.DataFrame(rows).T.sort_index()
    panel.index.name = "date"
    diag = pd.DataFrame(diagnostics).set_index("date") if diagnostics else pd.DataFrame()
    return panel, diag
