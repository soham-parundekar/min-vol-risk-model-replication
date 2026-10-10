"""Module G: replication accuracy, the estimator horse race, risk and alpha.

Produces the tables the report is written from:

    replication_accuracy     tracking error and correlation against USMV and SPLV
    performance_summary      return, volatility, drawdown for every portfolio
    estimator_horse_race     realised out-of-sample volatility and turnover by
                             covariance estimator
    risk_decomposition       factor, industry and specific shares of risk
    sector_tilts             active industry weights against the parent index
    alpha_tests              FF5+MOM and FF5+MOM+BAB regressions

    python scripts/05_evaluate.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minvol import pipeline as pl  # noqa: E402
from minvol.config import PARAMS, PATHS  # noqa: E402
from minvol.evaluate import alpha_tests, attribution, replication  # noqa: E402

ESTIMATORS = ["structural", "sample", "ledoit_wolf", "pca"]
# The beta-free reading of the style exemption, run as a variant.
VARIANTS = ["structural_betafree"]


def load_weight_panels() -> dict[str, pd.DataFrame]:
    out = {}
    for est in [*ESTIMATORS, *VARIANTS]:
        path = PATHS.processed / f"weights_usmv_{est}.parquet"
        if path.is_file():
            out[f"usmv_{est}"] = pd.read_parquet(path)
    splv = PATHS.processed / "weights_splv.parquet"
    if splv.is_file():
        out["splv"] = pd.read_parquet(splv)
    return out


def main() -> None:
    started = time.time()
    PATHS.ensure()
    panels = pl.load_panels()
    cfg = PARAMS["evaluation"]

    weights = load_weight_panels()
    print(f"Weight panels: {list(weights)}")

    etf_prices = pd.read_parquet(PATHS.interim / "etf_prices.parquet")
    etf = replication.etf_returns(etf_prices)
    external = pd.read_parquet(PATHS.processed / "external_factors.parquet")

    print("\nBuilding portfolio return series")
    gross, net, trades = {}, {}, {}
    for name, panel in weights.items():
        g, n, t = replication.portfolio_returns(panel, panels.daily_returns)
        gross[name], net[name], trades[name] = g, n, t
        print(f"  {name:22s} {len(g)} days, "
              f"{g.index.min().date()} to {g.index.max().date()}, "
              f"mean one-way turnover {t['one_way_turnover'].mean():.3f}")

    gross_df = pd.DataFrame(gross).sort_index()
    net_df = pd.DataFrame(net).sort_index()
    gross_df.to_parquet(PATHS.processed / "portfolio_returns_gross.parquet")
    net_df.to_parquet(PATHS.processed / "portfolio_returns_net.parquet")

    # --- performance ------------------------------------------------------
    rows = []
    for name in gross_df.columns:
        for label, series in (("gross", gross_df[name]), ("net", net_df[name])):
            stats = replication.performance_stats(series.dropna(), rf=external.get("RF"))
            if stats:
                stats.update({"portfolio": name, "basis": label})
                rows.append(stats)
    for ticker in [*cfg["replication_targets"], cfg["benchmark_ticker"]]:
        if ticker in etf.columns:
            stats = replication.performance_stats(etf[ticker].dropna(),
                                                  rf=external.get("RF"))
            stats.update({"portfolio": f"{ticker} (actual ETF)", "basis": "net"})
            rows.append(stats)
    performance = pd.DataFrame(rows).set_index(["portfolio", "basis"])
    performance.to_parquet(PATHS.tables / "performance_summary.parquet")
    print("\nPerformance (net basis)")
    print(performance.xs("net", level="basis")[
        ["start", "end", "cagr", "volatility", "sharpe", "max_drawdown"]
    ].round(4).to_string())

    # --- replication accuracy --------------------------------------------
    pairs = [(f"usmv_{e}", "USMV") for e in [*ESTIMATORS, *VARIANTS]] + [("splv", "SPLV")]
    acc = []
    for name, target in pairs:
        if name not in net_df.columns or target not in etf.columns:
            continue
        for label, frame in (("gross", gross_df), ("net", net_df)):
            stats = replication.tracking_error(frame[name].dropna(), etf[target].dropna())
            if stats:
                stats.update({"replica": name, "target": target, "basis": label})
                acc.append(stats)
        # Also against the common window, so the four estimators are comparable.
        stats = replication.tracking_error(
            net_df[name].loc[cfg["common_start"]:].dropna(),
            etf[target].loc[cfg["common_start"]:].dropna(),
        )
        if stats:
            stats.update({"replica": name, "target": target, "basis": "net, common window"})
            acc.append(stats)
    accuracy = pd.DataFrame(acc).set_index(["replica", "target", "basis"])
    accuracy.to_parquet(PATHS.tables / "replication_accuracy.parquet")
    print("\nReplication accuracy (net)")
    print(accuracy.xs("net", level="basis")[
        ["n_days", "tracking_error_ann", "correlation", "replica_vol_ann", "target_vol_ann"]
    ].round(4).to_string())

    # --- estimator horse race --------------------------------------------
    race = []
    common = net_df.loc[cfg["common_start"]:]
    for est in ESTIMATORS:
        name = f"usmv_{est}"
        if name not in common.columns:
            continue
        series = common[name].dropna()
        if series.empty:
            continue
        diag_path = PATHS.processed / f"diag_usmv_{est}.parquet"
        diag = pd.read_parquet(diag_path) if diag_path.is_file() else pd.DataFrame()
        turnover = trades[name]["one_way_turnover"]
        turnover = turnover.loc[turnover.index >= cfg["common_start"]]
        race.append({
            "estimator": est,
            "realised_vol_ann": series.std(ddof=1) * np.sqrt(cfg["trading_days_per_year"]),
            "cagr_net": (1 + series).prod() ** (cfg["trading_days_per_year"] / len(series)) - 1,
            "max_drawdown": float(
                ((1 + series).cumprod() / (1 + series).cumprod().cummax() - 1).min()
            ),
            "mean_one_way_turnover": float(turnover.mean()),
            "median_holdings": float(diag["n_holdings"].median()) if len(diag) else np.nan,
            "mean_predicted_vol_ann": float(
                diag["predicted_vol"].mean() * np.sqrt(cfg["weeks_per_year"])
            ) if len(diag) else np.nan,
            "n_relaxed_rebalances": int(
                (diag["relaxations"].astype(str) != "{}").sum()
            ) if len(diag) else 0,
        })
    horse_race = pd.DataFrame(race).set_index("estimator")
    # The ratio of what the model predicted to what actually happened is the
    # headline: a number far from one means the estimator was not describing
    # the portfolio it produced.
    horse_race["predicted_over_realised"] = (
        horse_race["mean_predicted_vol_ann"] / horse_race["realised_vol_ann"]
    )
    horse_race.to_parquet(PATHS.tables / "estimator_horse_race.parquet")
    print(f"\nEstimator horse race (from {cfg['common_start']})")
    print(horse_race.round(4).to_string())

    # --- risk decomposition and sector tilts ------------------------------
    decomp_rows, tilt_frames = [], []
    lag = PARAMS["usmv"].get("covariance_lag_months", 0)
    full_parent = panels.parent_weight_panel()
    used_path = PATHS.processed / "parent_weights_used.parquet"
    parent_used = pd.read_parquet(used_path) if used_path.is_file() else None

    for name, panel in weights.items():
        for date in panel.index[-6:]:
            w = panel.loc[date].dropna()
            w = w[w > 0]
            if w.empty:
                continue
            # Active weights are measured against the parent vector the
            # optimiser was actually constrained on, which is normalised over
            # the covariance's names rather than the whole tradable universe.
            # The wider vector gives tilts that look like band breaches and are
            # not. SPLV never saw either vector, so it uses the full universe.
            if name.startswith("usmv_") and parent_used is not None \
                    and date in parent_used.index:
                parent = parent_used.loc[date].dropna()
            else:
                asof = panels.asof_for_rebalance(date, lag)
                if asof is None:
                    continue
                parent = full_parent.loc[asof].dropna()
            tilts = attribution.active_sector_weights(w, parent, panels.industry_labels)
            tilts["portfolio"], tilts["date"] = name, date
            tilt_frames.append(tilts.reset_index(names="industry"))
            conc = attribution.concentration(w)
            conc.update({"portfolio": name, "date": date,
                         "max_abs_active_sector": float(tilts["active"].abs().max())})
            decomp_rows.append(conc)
    pd.DataFrame(decomp_rows).to_parquet(PATHS.tables / "concentration.parquet", index=False)
    if tilt_frames:
        pd.concat(tilt_frames, ignore_index=True).to_parquet(
            PATHS.tables / "sector_tilts.parquet", index=False)
    print("\nConcentration (most recent rebalances)")
    print(pd.DataFrame(decomp_rows).groupby("portfolio")[
        ["n_holdings", "max_weight", "top10_weight", "effective_n",
         "max_abs_active_sector"]
    ].mean().round(4).to_string())

    # --- alpha tests ------------------------------------------------------
    print("\nAlpha tests")
    tables = []
    for name in net_df.columns:
        tables.append(alpha_tests.alpha_with_and_without_bab(
            net_df[name].dropna(), external, label=name))
    for ticker in cfg["replication_targets"]:
        if ticker in etf.columns:
            tables.append(alpha_tests.alpha_with_and_without_bab(
                etf[ticker].dropna(), external, label=f"{ticker} (actual ETF)"))
    alpha = pd.concat([t for t in tables if not t.empty], ignore_index=True)
    alpha.to_parquet(PATHS.tables / "alpha_tests.parquet", index=False)
    print(alpha[["portfolio", "model", "n_obs", "alpha_ann", "alpha_t",
                 "r_squared"]].round(4).to_string(index=False))

    changes = alpha_tests.alpha_change(alpha)
    change_frame = pd.DataFrame(changes).T
    change_frame.to_parquet(PATHS.tables / "alpha_absorbed_by_bab.parquet")
    print("\nHow much of the intercept BAB absorbs")
    print(change_frame.round(4).to_string())

    print(f"\nFinished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
