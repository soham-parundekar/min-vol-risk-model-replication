"""Render the manuscript's figures as vector PDF from the frozen outputs.

Separate from scripts/06_figures.py, which produces the raster figures the
internal report embeds. These are typeset for the paper: serif to match the
body text, sized to the text block, no chart furniture that a printed page does
not need. Both read the same frozen parquet files and neither computes a
statistic; cumulative growth is a display transformation of stored returns.

Run from the repository root:

    python3 paper/make_figures.py
"""
from __future__ import annotations

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

ROOT = pathlib.Path(__file__).resolve().parents[1]
T = ROOT / "reports" / "tables"
P = ROOT / "data" / "processed"
I = ROOT / "data" / "interim"
OUT = ROOT / "paper" / "figures"

# Categorical slots in fixed order. Never cycled: a fifth series would mean a
# different figure, not a fifth colour.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#1a1a19", "#5c5c58", "#d8d8d4"

# 0.94\linewidth on letter with 1in margins is about 6.1in.
WIDTH = 6.1


def _serif() -> str:
    """Nearest available serif to the document's Times, with a fallback."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Nimbus Roman", "Nimbus Roman No9 L", "FreeSerif",
                 "DejaVu Serif"):
        if name in have:
            return name
    return "serif"


plt.rcParams.update({
    "font.family": "serif",
    "font.serif": [_serif()],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})


def _tidy(ax, ygrid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if ygrid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.5, zorder=0)
        ax.set_axisbelow(True)


def _save(fig, name):
    """Write the figure, with the creation date pinned.

    Matplotlib stamps the current time into the PDF, so an unchanged figure
    rebuilds to different bytes every run and a reviewer diffing two checkouts
    sees four modified files that are visually identical. Pinning the date
    makes the output a function of the data alone, which is the property the
    rest of this directory already has.
    """
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02,
                metadata={"CreationDate": None})
    plt.close(fig)
    print(f"  wrote paper/figures/{name}.pdf")


def survivorship():
    """Annual gap between the rebuilt parent and the market ETF.

    One series, so no legend: the axis label says what it is. A bar per year
    because the reader's question is which years carry the bias, not what
    happens between them.
    """
    sg = pd.read_parquet(T / "survivorship_gap.parquet")
    years = sg.drop(index="full sample")
    full = float(sg.loc["full sample", "gap_ann"]) * 100
    vals = years["gap_ann"].to_numpy() * 100
    labels = [str(i) for i in years.index]

    fig, ax = plt.subplots(figsize=(WIDTH, 2.5))
    ax.bar(labels, vals, color=BLUE, width=0.66, zorder=2)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=3)
    ax.axhline(full, color=ORANGE, linewidth=1.1, linestyle="--", zorder=4)
    ax.annotate(f"full sample  {full:+.2f}", xy=(len(labels) - 0.4, full),
                xytext=(0, 4), textcoords="offset points", ha="right",
                va="bottom", color=ORANGE, fontsize=8)
    ax.set_ylabel("Gap, percentage points a year")
    _tidy(ax)
    plt.setp(ax.get_xticklabels(), rotation=0)
    _save(fig, "survivorship")


def replication():
    """Growth of one dollar: the fund against two readings of the exemption."""
    rets = pd.read_parquet(P / "portfolio_returns_net.parquet")
    etf = pd.read_parquet(I / "etf_prices.parquet")
    fund = etf["USMV"].dropna().pct_change().dropna()

    cols = {
        "usmv_structural": ("USMV replica, residual volatility exempt", ORANGE),
        "usmv_structural_betafree": ("USMV replica, beta also exempt", AQUA),
    }
    start = max(rets.index.min(), fund.index.min(),
                pd.Timestamp("2012-05-31"))
    fig, ax = plt.subplots(figsize=(WIDTH, 2.9))

    f = fund.loc[fund.index >= start]
    ax.plot(f.index, (1 + f).cumprod(), color=BLUE, linewidth=1.3,
            label="USMV, the fund", zorder=4)
    for key, (label, colour) in cols.items():
        s = rets[key].dropna()
        s = s.loc[s.index >= start]
        ax.plot(s.index, (1 + s).cumprod(), color=colour, linewidth=1.1,
                label=label, zorder=3)

    ax.set_ylabel("Growth of one dollar, net")
    _tidy(ax)
    ax.legend(frameon=False, loc="upper left", handlelength=1.6,
              borderpad=0.2, labelspacing=0.35)
    _save(fig, "replication")


def bias():
    """Rolling bias statistic per estimator, as small multiples.

    One axis with four lines was the obvious design and it was wrong: the three
    non-structural estimators trace almost the same path, so whichever is drawn
    last hides the other two and the legend promises four series the reader can
    only find two of. The near-identity is the finding, so the figure has to
    make it visible rather than let it destroy the chart. Four panels on a
    shared scale do that, and comparing panels is how the reader sees it.
    """
    roll = pd.read_parquet(P / "bias_rolling.parquet")
    bs = pd.read_parquet(T / "bias_statistics.parquet")
    mv = bs[bs.family == "min_variance"].set_index("estimator")
    lo, hi = float(mv.ci_low.iloc[0]), float(mv.ci_high.iloc[0])

    panels = [
        ("structural", "Structural", BLUE),
        ("sample", "Sample", ORANGE),
        ("ledoit_wolf", "Ledoit–Wolf", AQUA),
        ("pca", "PCA", YELLOW),
    ]
    series = {k: roll[f"{k}::min_variance"].dropna() for k, _, _ in panels}
    top = max(s.max() for s in series.values())

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 3.2), sharex=True,
                             sharey=True)
    for ax, (key, label, colour) in zip(axes.ravel(), panels):
        s = series[key]
        ax.axhspan(lo, hi, color=GRID, alpha=0.8, zorder=1)
        ax.axhline(1.0, color=MUTED, linewidth=0.5, zorder=2)
        ax.plot(s.index, s.to_numpy(), color=colour, linewidth=1.0, zorder=3)
        ax.set_title(f"{label}   $b$ = {mv.loc[key, 'bias_mean']:.3f}",
                     fontsize=8.5, pad=4)
        ax.set_ylim(0.4, top * 1.06)
        _tidy(ax)

    for ax in axes[:, 0]:
        ax.set_ylabel("Rolling $b$")
    fig.subplots_adjust(hspace=0.42, wspace=0.14)
    _save(fig, "bias")


def tilts():
    """Largest active industry weight, one panel each.

    Small multiples rather than one axis, because the two portfolios review on
    different calendars and their six most recent reviews share no dates. Two
    panels make the mismatch visible instead of implying a comparison the data
    does not support. The y-scales differ by an order of magnitude and are
    labelled as such.
    """
    st = pd.read_parquet(T / "sector_tilts.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.4))

    for ax, (port, title, colour) in zip(axes, [
        ("usmv_structural", "USMV replica, sector band $\\pm$5%", BLUE),
        ("splv", "SPLV replica, no sector constraint", ORANGE),
    ]):
        s = st[st.portfolio == port].copy()
        s["a"] = s.active.abs()
        top = s.loc[s.groupby("date").a.idxmax()].sort_values("date")
        x = [str(d)[:7] for d in top.date]
        ax.bar(x, top.active.to_numpy() * 100, color=colour, width=0.6,
               zorder=2)
        ax.axhline(0, color=MUTED, linewidth=0.6, zorder=3)
        if port == "usmv_structural":
            for y in (5, -5):
                ax.axhline(y, color=MUTED, linewidth=0.7, linestyle=":",
                           zorder=3)
        ax.set_title(title, fontsize=8.5, pad=6)
        _tidy(ax)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=7)

    axes[0].set_ylabel("Largest active industry weight, %")
    fig.subplots_adjust(wspace=0.28)
    _save(fig, "tilts")


def main():
    for fn in (survivorship, replication, bias, tilts):
        fn()


if __name__ == "__main__":
    main()
