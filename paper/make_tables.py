"""Render the manuscript's empirical tables straight from the frozen outputs.

Nothing in paper/tables/ is typed by hand. Every cell here is read from a
parquet file under reports/tables/ and formatted by an explicit rule, so a
reviewer can point at a number in the PDF and find the column it came from.
The formats match the evidence ledger: percentages to two decimals, ratios to
three, t-statistics to one or two depending on how the text quotes them.

Run from the repository root:

    python3 paper/make_tables.py

Writing a table by hand would be faster once and wrong eventually. The whole
reason this file exists is that the research record already disagreed with its
own tables in two places, and a generated table cannot drift from its source.
"""
from __future__ import annotations

import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "tables"
OUT = ROOT / "paper" / "tables"

# Display order and display names. The dictionary is the single place a label
# changes, so a rename cannot leave one table disagreeing with another.
REPLICAS = {
    "splv": "SPLV replica",
    "usmv_structural": "USMV, structural",
    "usmv_sample": "USMV, sample",
    "usmv_ledoit_wolf": "USMV, Ledoit--Wolf",
    "usmv_pca": "USMV, PCA",
    "usmv_structural_betafree": "USMV, structural, beta exempt",
}
ESTIMATORS = {
    "structural": "Structural factor model",
    "sample": "Sample",
    "ledoit_wolf": "Ledoit--Wolf",
    "pca": "PCA, Bai--Ng",
}


def pc(x, dp=2):
    """Percent with an escaped sign, or an en dash when the cell is empty."""
    return "--" if pd.isna(x) else f"{x * 100:.{dp}f}"


def num(x, dp=3):
    return "--" if pd.isna(x) else f"{x:.{dp}f}"


def _header(caption: str, label: str, colspec: str) -> list[str]:
    return [
        r"\begin{table}[tbp]",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]


def _footer(note: str) -> list[str]:
    return [
        r"\bottomrule",
        r"\end{tabular}",
        rf"\notetext{{{note}}}",
        r"\end{table}",
        "",
    ]


def replication_table() -> str:
    ra = pd.read_parquet(SRC / "replication_accuracy.parquet")
    net = ra.xs("net", level="basis")

    lines = _header(
        "Replication accuracy against the live funds.",
        "tab:replication",
        "lrrrrrr",
    )
    lines.append(
        r"Replica & TE (\%) & Corr. & $\beta$ & Replica vol (\%) & "
        r"Fund vol (\%) & Days\\"
    )
    lines.append(r"\midrule")

    for key, label in REPLICAS.items():
        if key not in net.index.get_level_values("replica"):
            continue
        r = net.xs(key, level="replica").iloc[0]
        lines.append(
            f"{label} & {pc(r.tracking_error_ann)} & {num(r.correlation, 4)} & "
            f"{num(r.beta_to_target)} & {pc(r.replica_vol_ann)} & "
            f"{pc(r.target_vol_ann)} & {int(r.n_days):,}\\\\"
        )
        if key == "splv":
            lines.append(r"\addlinespace")

    windows = sorted({(str(r.start)[:10], str(r.end)[:10])
                      for _, r in net.iterrows()})
    note = (
        "Daily total returns, net of the replicas' modelled trading costs, "
        "against each fund's own history from its inception. Tracking error is "
        "the annualised standard deviation of the return difference and "
        "$\\beta$ is the replica's beta to the fund it tracks, so a replica "
        "that matched the fund exactly would show a tracking error of zero, a "
        "correlation of one and a beta of one. The SPLV replica has the longer "
        "window because SPLV began trading before USMV; windows run "
        + " and ".join(f"{a} to {b}" for a, b in windows)
        + ". The last row repeats the structural specification with the beta "
        "factor added to the style exemption, which is the one specification "
        "choice Section~\\ref{sec:rules} shows the published methodology does "
        "not settle."
    )
    return "\n".join(lines + _footer(note))


def _ladder_counts() -> dict[str, tuple[int, int]]:
    """Per estimator: rebalances that loosened a bound, and that exhausted the ladder.

    These are different events and the distinction matters. A loosened bound
    still produces an optimised book. An exhausted ladder produces no trade at
    all, so the previous weights are carried forward and then drift, which is
    how a portfolio ends a period further outside the sector band than the
    optimiser would ever have placed it.
    """
    out: dict[str, tuple[int, int]] = {}
    for key in ESTIMATORS:
        path = ROOT / "data" / "processed" / f"diag_usmv_{key}.parquet"
        if not path.is_file():
            continue
        d = pd.read_parquet(path)
        loosened = int((d["relaxations"].astype(str) != "{}").sum())
        exhausted = int((d["status"] == "not_rebalanced").sum())
        out[key] = (loosened, exhausted)
    return out


def horse_race_table() -> str:
    hr = pd.read_parquet(SRC / "estimator_horse_race.parquet")
    bs = pd.read_parquet(SRC / "bias_statistics.parquet")
    mv = bs[bs.family == "min_variance"].set_index("estimator")
    ladder = _ladder_counts()

    lines = _header(
        "The four covariance estimators under an identical constraint set.",
        "tab:horserace",
        "lrrrrrrr",
    )
    lines.append(
        r"Estimator & Realised vol (\%) & Pred./real. & Bias $b$ & "
        r"Turnover (\%) & Holdings & Loosened & No trade\\"
    )
    lines.append(r"\midrule")

    for key, label in ESTIMATORS.items():
        r = hr.loc[key]
        b = mv.loc[key, "bias_mean"] if key in mv.index else float("nan")
        loosened, exhausted = ladder.get(key, (int(r.n_relaxed_rebalances), 0))
        lines.append(
            f"{label} & {pc(r.realised_vol_ann)} & "
            f"{num(r.predicted_over_realised)} & {num(b)} & "
            f"{pc(r.mean_one_way_turnover)} & {int(r.median_holdings)} & "
            f"{loosened} & {exhausted}\\\\"
        )

    spread = hr.realised_vol_ann.max() - hr.realised_vol_ann.min()
    lo = float(mv.ci_low.iloc[0])
    hi = float(mv.ci_high.iloc[0])
    nobs = int(mv.n_obs.iloc[0])
    note = (
        "Each estimator is run through the same constrained optimiser on the "
        f"same rebalance calendar, so the rows differ only in $\\Sigma$. The "
        f"four realised volatilities span {spread * 1e4:.0f} basis points. "
        "Pred./real. is the ratio of mean predicted to realised annualised "
        "volatility, and bias $b$ is the standard deviation of realised return "
        "divided by the previous period's predicted standard deviation, "
        f"computed on the minimum-variance portfolio over {nobs:,} "
        "observations. The two measure calibration differently and are not "
        "interchangeable: $b$ carries a well-specified band of "
        f"[{lo:.3f}, {hi:.3f}], and all four estimators sit above it. Turnover "
        "is the mean one-way figure per rebalance. Loosened counts the "
        "rebalances of 29 at which the published ladder put a weaker bound in "
        "force, and No trade those at which the ladder was exhausted without "
        "reaching a feasible book, so the previous weights were carried "
        "forward unchanged. The turnover cap is what binds in almost every "
        "case; the sector band was never the bound that had to be loosened."
    )
    return "\n".join(lines + _footer(note))


def alpha_table() -> str:
    at = pd.read_parquet(SRC / "alpha_tests.parquet")
    base = at[~at.model.str.contains("BAB")].set_index("portfolio")
    bab = at[at.model.str.contains("BAB")].set_index("portfolio")

    order = [
        "USMV (actual ETF)", "usmv_structural", "usmv_structural_betafree",
        "SPLV (actual ETF)", "splv",
    ]
    labels = {
        "USMV (actual ETF)": "USMV, the fund",
        "usmv_structural": "USMV replica, structural",
        "usmv_structural_betafree": "USMV replica, beta exempt",
        "SPLV (actual ETF)": "SPLV, the fund",
        "splv": "SPLV replica",
    }

    lines = _header(
        "Six-factor alphas before and after the betting-against-beta factor.",
        "tab:alpha",
        "lrrrrrr",
    )
    lines.append(
        r"Portfolio & $\alpha$ (\%) & $t$ & $\alpha$+BAB (\%) & $t$ & "
        r"$\beta_{\rm BAB}$ & $t$\\"
    )
    lines.append(r"\midrule")

    for key in order:
        if key not in base.index:
            continue
        a, c = base.loc[key], bab.loc[key]
        lines.append(
            f"{labels[key]} & {pc(a.alpha_ann)} & {num(a.alpha_t, 2)} & "
            f"{pc(c.alpha_ann)} & {num(c.alpha_t, 2)} & "
            f"{num(c.beta_BAB)} & {num(c.t_BAB, 1)}\\\\"
        )
        if key == "usmv_structural_betafree":
            lines.append(r"\addlinespace")

    lags = int(bab.nw_lags.iloc[0])
    b0, b1 = str(base.start.min())[:10], str(base.end.max())[:10]
    c0, c1 = str(bab.start.min())[:10], str(bab.end.max())[:10]
    note = (
        "Daily excess returns regressed on the Fama--French five factors plus "
        "momentum, then re-run with the betting-against-beta factor added. "
        "Annualised intercepts, with Newey--West standard errors at "
        f"{lags} lags. The two blocks end on different dates because the "
        "factor vintages do: the six-factor window runs {b0} to {b1} and the "
        "seven-factor window to {c1}, so the columns are not a paired test on "
        "an identical sample. No intercept reaches significance at the five "
        "per cent level in either block, though the largest, the SPLV fund's "
        "own at $t=\\mathrm{SPLVT}$, falls only just short at $p=\\mathrm{SPLVP}$ "
        "and would clear a ten per cent threshold. Every "
        "betting-against-beta loading is significant by a wide margin, which "
        "is why the paper reads the pattern as absorption rather than as an "
        "absence of exposure."
    ).replace("{b0}", b0).replace("{b1}", b1).replace("{c0}", c0).replace("{c1}", c1)

    worst = bab.loc[bab.alpha_t.idxmin()]
    note = (note
            .replace("\\mathrm{SPLVT}", f"{worst.alpha_t:.2f}")
            .replace("\\mathrm{SPLVP}", f"{worst.alpha_p:.3f}"))
    return "\n".join(lines + _footer(note))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in [
        ("replication.tex", replication_table),
        ("horserace.tex", horse_race_table),
        ("alpha.tex", alpha_table),
    ]:
        (OUT / name).write_text(builder(), encoding="utf-8")
        print(f"  wrote paper/tables/{name}")


if __name__ == "__main__":
    main()
