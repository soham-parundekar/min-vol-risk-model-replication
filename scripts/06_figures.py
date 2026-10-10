"""Figures for the report.

Each one answers a question the text asks, and the title says what the answer
is rather than naming the axes again. Charts that would only restate a table
are not drawn.

    python scripts/06_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minvol.config import PARAMS, PATHS  # noqa: E402

# Fixed categorical order, never cycled. A sixth series would fold into
# "other" rather than take a generated colour.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e3e2de"
NEUTRAL = "#8d8c86"
SURFACE = "#fcfcfb"

PRETTY = {
    "usmv_structural": "USMV replica, structural",
    "usmv_sample": "USMV replica, sample",
    "usmv_ledoit_wolf": "USMV replica, Ledoit-Wolf",
    "usmv_pca": "USMV replica, PCA",
    "usmv_structural_betafree": "USMV replica, beta free",
    "splv": "SPLV replica",
}


def style_axes(ax, title: str, subtitle: str = "", ylabel: str = "") -> None:
    """Recessive frame, ink-coloured text, grid behind the marks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, labelsize=9, length=3)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.set_title(title, color=INK, fontsize=12, loc="left",
                 pad=22 if subtitle else 8)
    if subtitle:
        ax.text(0.0, 1.012, subtitle, transform=ax.transAxes, color=INK_SOFT,
                fontsize=9, va="bottom")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SOFT, fontsize=9)


def save(fig, name: str) -> None:
    PATHS.figures.mkdir(parents=True, exist_ok=True)
    out = PATHS.figures / f"{name}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote {out.relative_to(PATHS.root)}")


def growth_curves(net: pd.DataFrame, etf: pd.DataFrame, start: str) -> None:
    """Cumulative growth of one dollar, replicas against the funds they imitate."""
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    pairs = [("usmv_structural", "USMV", SERIES[0]), ("splv", "SPLV", SERIES[2])]

    for i, (name, target, colour) in enumerate(pairs):
        if name not in net.columns:
            continue
        s = net[name].loc[start:].dropna()
        if s.empty:
            continue
        curve = (1 + s).cumprod()
        ax.plot(curve.index, curve.values, color=colour, linewidth=2.0,
                label=PRETTY.get(name, name))
        if target in etf.columns:
            t = etf[target].reindex(s.index).dropna()
            tc = (1 + t).cumprod()
            ax.plot(tc.index, tc.values, color=colour, linewidth=1.6,
                    linestyle=(0, (4, 2)), alpha=0.85, label=f"{target}, actual fund")

    spy = etf["SPY"].loc[start:].dropna()
    ax.plot(spy.index, (1 + spy).cumprod().values, color=NEUTRAL, linewidth=1.6,
            label="SPY")

    style_axes(
        ax,
        "Both replicas track their funds' shape; the optimiser drifts further",
        f"growth of $1, net of {PARAMS['evaluation']['transaction_cost_bps']:.0f}bp "
        f"one-way costs, from {start}",
        "value of $1",
    )
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncols=2)
    save(fig, "growth_of_one_dollar")


def rolling_volatility(net: pd.DataFrame, etf: pd.DataFrame, start: str) -> None:
    """Realised volatility through time, the quantity both indexes exist to lower."""
    ppy = PARAMS["evaluation"]["trading_days_per_year"]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))

    series = [("usmv_structural", SERIES[0]), ("splv", SERIES[2])]
    for name, colour in series:
        if name not in net.columns:
            continue
        s = net[name].loc[start:].dropna()
        roll = s.rolling(ppy).std(ddof=1) * np.sqrt(ppy)
        ax.plot(roll.index, roll.values * 100, color=colour, linewidth=2.0,
                label=PRETTY.get(name, name))
    for target, colour in (("USMV", SERIES[0]), ("SPLV", SERIES[2])):
        if target not in etf.columns:
            continue
        t = etf[target].loc[start:].dropna()
        roll = t.rolling(ppy).std(ddof=1) * np.sqrt(ppy)
        ax.plot(roll.index, roll.values * 100, color=colour, linewidth=1.6,
                linestyle=(0, (4, 2)), alpha=0.85, label=f"{target}, actual fund")
    spy = etf["SPY"].loc[start:].dropna()
    roll = spy.rolling(ppy).std(ddof=1) * np.sqrt(ppy)
    ax.plot(roll.index, roll.values * 100, color=NEUTRAL, linewidth=1.6, label="SPY")

    style_axes(ax, "The replicas are less volatile than the market, and more volatile than the funds",
               "trailing 252-day realised volatility, annualised", "volatility, % a year")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncols=2)
    save(fig, "rolling_volatility")


def survivorship(start: str) -> None:
    """The rebuilt parent index against SPY: what the missing delistings are worth."""
    parent = pd.read_parquet(
        PATHS.processed / "parent_index_returns.parquet")["parent"]
    etf = pd.read_parquet(PATHS.interim / "etf_prices.parquet")
    spy = etf["SPY"].pct_change(fill_method=None)
    joined = pd.concat([parent.rename("parent"), spy.rename("spy")],
                       axis=1, sort=True).dropna().loc[start:]
    gap = (1 + joined["parent"]).cumprod() / (1 + joined["spy"]).cumprod() - 1

    fig, ax = plt.subplots(figsize=(9.5, 4.4))
    ax.plot(gap.index, gap.values * 100, color=SERIES[1], linewidth=2.0)
    ax.axhline(0, color=NEUTRAL, linewidth=1.0)
    ppy = PARAMS["evaluation"]["trading_days_per_year"]
    ann = (joined["parent"] - joined["spy"]).mean() * ppy * 100
    style_axes(
        ax,
        f"Rebuilding the index from surviving names flatters it by about {ann:.2f}% a year",
        "cumulative return of the rebuilt cap-weighted parent index minus SPY",
        "cumulative gap, %",
    )
    ax.text(gap.index[-1], gap.values[-1] * 100, f"  {gap.values[-1] * 100:.1f}%",
            color=INK_SOFT, fontsize=9, va="center")
    save(fig, "survivorship_gap")


def sector_tilts() -> None:
    """Where each rule puts its money relative to the parent index."""
    path = PATHS.tables / "sector_tilts.parquet"
    if not path.is_file():
        return
    tilts = pd.read_parquet(path)
    wanted = ["usmv_structural", "splv"]
    # Latest rebalance *per portfolio*. A global max would pick SPLV's date,
    # which is quarterly and therefore later than USMV's semi-annual one, and
    # the USMV series would silently vanish from the chart.
    latest = tilts[tilts["portfolio"].isin(wanted)].copy()
    if latest.empty:
        return
    keep = latest.groupby("portfolio")["date"].transform("max")
    sub = latest[latest["date"] == keep]
    pivot = sub.pivot_table(index="industry", columns="portfolio", values="active")
    pivot = pivot.reindex(pivot.abs().max(axis=1).sort_values().index)

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    y = np.arange(len(pivot))
    height = 0.38
    for i, name in enumerate(wanted):
        if name not in pivot.columns:
            continue
        # A 2px surface gap between adjacent bars keeps the pair readable.
        ax.barh(y + (i - 0.5) * (height + 0.02), pivot[name].values * 100,
                height=height, color=SERIES[0] if i == 0 else SERIES[2],
                label=PRETTY.get(name, name), linewidth=0)
    band = PARAMS["usmv"]["sector_band"] * 100
    ax.axvline(band, color=NEUTRAL, linewidth=1.0, linestyle=(0, (3, 3)))
    ax.axvline(-band, color=NEUTRAL, linewidth=1.0, linestyle=(0, (3, 3)))
    ax.axvline(0, color=NEUTRAL, linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(pivot.index, fontsize=9, color=INK_SOFT)

    style_axes(ax, "The sector band is what stops a low-volatility rule becoming a utilities bet",
               f"active industry weight against the parent index, latest rebalance; "
               f"dashed lines mark MSCI's +/-{band:.0f}% band", "")
    ax.set_xlabel("active weight, percentage points", color=INK_SOFT, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, loc="lower right")
    save(fig, "sector_tilts")


def estimator_race() -> None:
    """Predicted against realised volatility, by covariance estimator."""
    path = PATHS.tables / "estimator_horse_race.parquet"
    if not path.is_file():
        return
    race = pd.read_parquet(path)
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    x = np.arange(len(race))
    width = 0.38
    ax.bar(x - width / 2 - 0.01, race["mean_predicted_vol_ann"] * 100, width,
           color=SERIES[0], label="predicted by the model", linewidth=0)
    ax.bar(x + width / 2 + 0.01, race["realised_vol_ann"] * 100, width,
           color=SERIES[1], label="realised out of sample", linewidth=0)
    for i, (p, r) in enumerate(zip(race["mean_predicted_vol_ann"],
                                   race["realised_vol_ann"])):
        ax.text(i - width / 2 - 0.01, p * 100 + 0.2, f"{p * 100:.1f}", ha="center",
                fontsize=8, color=INK_SOFT)
        ax.text(i + width / 2 + 0.01, r * 100 + 0.2, f"{r * 100:.1f}", ha="center",
                fontsize=8, color=INK_SOFT)
    ax.set_xticks(x)
    ax.set_xticklabels(race.index, fontsize=9, color=INK_SOFT)
    style_axes(ax, "Every estimator understates the risk of the portfolio it builds",
               "USMV replica, common window", "volatility, % a year")
    # Headroom for the value labels, and the legend below the plot so it cannot
    # land on the rightmost bar's label.
    ax.set_ylim(0, max(race["realised_vol_ann"].max(),
                       race["mean_predicted_vol_ann"].max()) * 100 * 1.12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncols=2,
              loc="upper center", bbox_to_anchor=(0.5, -0.12))
    save(fig, "estimator_horse_race")


def bias_through_time() -> None:
    """Rolling bias statistic against its theoretical band."""
    path = PATHS.processed / "bias_rolling.parquet"
    if not path.is_file():
        return
    rolling = pd.read_parquet(path)
    cols = [c for c in rolling.columns if c.endswith("::min_variance")]
    if not cols:
        cols = list(rolling.columns)[:4]

    window = PARAMS["bias_tests"]["rolling_window_weeks"]
    half = np.sqrt(2.0 / window)
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.axhspan(1 - half, 1 + half, color=GRID, alpha=0.8, linewidth=0)
    ax.axhline(1.0, color=NEUTRAL, linewidth=1.0)
    for i, col in enumerate(cols[:5]):
        s = rolling[col].dropna()
        if s.empty:
            continue
        ax.plot(s.index, s.values, color=SERIES[i % len(SERIES)], linewidth=1.8,
                label=col.split("::")[0])
    style_axes(ax, "Calibration holds on average and breaks in the crisis",
               f"rolling {window}-week bias statistic; shaded band is the "
               f"1 +/- sqrt(2/T) interval under a correct model", "bias statistic")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SOFT, ncols=2)
    save(fig, "bias_through_time")


def main() -> None:
    PATHS.ensure()
    cfg = PARAMS["evaluation"]
    start = cfg["common_start"]

    net = pd.read_parquet(PATHS.processed / "portfolio_returns_net.parquet")
    etf_prices = pd.read_parquet(PATHS.interim / "etf_prices.parquet")
    etf = etf_prices.sort_index().pct_change(fill_method=None)

    print("Figures")
    growth_curves(net, etf, start)
    rolling_volatility(net, etf, start)
    survivorship(PARAMS["sample"]["exposure_start"])
    sector_tilts()
    estimator_race()
    bias_through_time()


if __name__ == "__main__":
    main()
