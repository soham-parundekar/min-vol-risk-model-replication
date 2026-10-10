"""Daily market capitalisation and book-to-price, built point-in-time.

Three inputs meet here: XBRL facts keyed by filing date, split-back-adjusted
prices from Yahoo, and the split ratios that reconcile the two. The output is a
pair of daily panels, market cap and book equity, that contain on every date
only what a person sitting at a terminal that morning could actually have read.

The step-function construction is the part worth reading twice. A naive
implementation asks, for each of 4,700 dates and 618 tickers, "what was the
latest filed fact?" and takes minutes. The version here walks each ticker's
facts once in filing order, carrying the most recent fiscal period seen so far,
and emits a step series that is then reindexed onto the trading calendar. Same
answer, two orders of magnitude faster, and the ordering rule is explicit
rather than buried in a sort.

One wrinkle that rule has to handle: a 10-K filed in February 2015 carries
comparative balance sheets for 2014 *and* 2013. Both arrive with the same
filing date. Taking the last row of a naive sort could hand back the 2013
figure. The walk keeps a fact only when its fiscal period is at least as recent
as the best one already seen, so the comparatives are ignored and a genuine
restatement of the latest period still lands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import PARAMS, PATHS


def point_in_time_series(facts: pd.DataFrame) -> pd.Series:
    """Collapse one ticker's facts for one field into a step function of filing dates.

    Input needs ``filed``, ``end`` and ``val`` columns. The result is indexed by
    filing date and holds the value that became public on that date, so that a
    forward-fill onto the trading calendar reproduces what was knowable.
    """
    if facts.empty:
        return pd.Series(dtype="float64")

    ordered = facts.sort_values(["filed", "end"], kind="mergesort")
    best_end: pd.Timestamp | None = None
    out: dict[pd.Timestamp, float] = {}
    for filed, end, val in zip(ordered["filed"], ordered["end"], ordered["val"]):
        if best_end is None or end >= best_end:
            best_end = end
            out[filed] = float(val)
    return pd.Series(out).sort_index()


def daily_fact_panel(facts: pd.DataFrame, field: str, dates: pd.DatetimeIndex,
                     tickers: list[str], max_staleness_days: int | None = None
                     ) -> pd.DataFrame:
    """Forward-filled daily panel of one point-in-time field.

    Where a field has several source tags, the preferred tag supplies every date
    it covers and the others only fill the gaps it leaves. Coalescing by date
    rather than choosing one tag per ticker matters for companies that switched
    tags mid-sample: the preferred series would otherwise either lose its tail
    or shadow the fallback that covers the rest of the history.
    """
    sub = facts[facts["field"] == field]
    panel = pd.DataFrame(np.nan, index=dates, columns=tickers, dtype="float64")
    has_rank = "tag_rank" in sub.columns
    # Datetime arithmetic rather than integer arithmetic on the index. pandas
    # indexes are not always nanosecond resolution, and dividing a raw int64
    # difference by a hard-coded nanoseconds-per-day constant silently yields a
    # near-zero age on a second-resolution index, which lets every stale value
    # through.
    day_index = pd.Series(dates, index=dates)

    for ticker, grp in sub.groupby("ticker"):
        if ticker not in panel.columns:
            continue
        ranks = sorted(grp["tag_rank"].unique()) if has_rank else [0]
        combined: pd.Series | None = None
        for rank in ranks:
            part = grp[grp["tag_rank"] == rank] if has_rank else grp
            step = point_in_time_series(part)
            if step.empty:
                continue
            # Reindex onto the union so a filing that lands on a non-trading day
            # is still picked up by the next session, then restrict to trading
            # dates.
            union = step.index.union(dates)
            joined = step.reindex(union).ffill().reindex(dates)

            # Staleness is applied per tag, before the tags are combined. The
            # other order looks equivalent and is not: a preferred tag that a
            # filer abandoned years ago would forward-fill forever, block the
            # fallback tag from ever filling the gap, and then get blanked by
            # the staleness cut, leaving nothing where a perfectly good
            # fallback existed. Alphabet, UPS and CME all behave this way.
            if max_staleness_days is not None:
                stamps = pd.Series(step.index, index=step.index)
                stamps = stamps.reindex(union).ffill().reindex(dates)
                age = (day_index - pd.to_datetime(stamps)).dt.days
                joined = joined.where(age <= max_staleness_days)

            combined = joined if combined is None else combined.fillna(joined)

        if combined is not None:
            panel[ticker] = combined.to_numpy()
    return panel


def clean_share_facts(facts: pd.DataFrame, split_factors: pd.DataFrame
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop share counts that disagree wildly with their neighbours.

    EDGAR republishes what filers tag, mistakes included. Yum Brands' October
    2016 10-Q carries 3.67e14 shares where the real figure is about 3.5e8, and
    Garmin's the same month carries 1.98e11 against 1.89e8. One such fact is
    enough to give a single stock a market cap in the quadrillions and turn a
    cap-weighted index into that stock.

    The comparison runs on a split-adjusted basis. Raw reported counts jump by
    the split ratio at every split, which a naive filter would read as an error
    and discard; once both the fact and its neighbours are expressed in today's
    share units, the series is smooth and a genuine outlier stands out.

    A rolling median over neighbouring filings is the reference rather than a
    mean, so a single bad fact cannot drag the thing it is being tested
    against. Returns the surviving facts and a log of what was removed.
    """
    cfg = PARAMS.get("fundamentals", {})
    window = int(cfg.get("share_outlier_window", 9))
    limit = float(cfg.get("share_outlier_factor", 5.0))

    shares = facts[facts["field"] == "shares_outstanding"].copy()
    if shares.empty:
        return facts, pd.DataFrame(columns=["ticker", "filed", "end", "val", "tag",
                                            "adjusted", "local_median", "ratio"])

    factor_lookup = split_factors.reindex(
        split_factors.index.union(pd.DatetimeIndex(shares["filed"].unique()))
    ).ffill().bfill()

    keep_flags: list[pd.Series] = []
    dropped: list[pd.DataFrame] = []
    for ticker, grp in shares.groupby("ticker"):
        grp = grp.sort_values("filed")
        if ticker in factor_lookup.columns:
            fac = factor_lookup[ticker].reindex(grp["filed"]).to_numpy()
        else:
            fac = np.ones(len(grp))
        adj = grp["val"].to_numpy(dtype="float64") * np.where(np.isfinite(fac), fac, 1.0)
        adj_s = pd.Series(adj, index=grp.index)

        med = adj_s.rolling(window, center=True, min_periods=3).median()
        med = med.fillna(adj_s.median())
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = adj_s / med
        bad = (~np.isfinite(ratio)) | (ratio > limit) | (ratio < 1.0 / limit)
        bad &= adj_s.notna()

        keep_flags.append(pd.Series(~bad.to_numpy(), index=grp.index))
        if bad.any():
            log = grp.loc[bad, ["ticker", "filed", "end", "val", "tag"]].copy()
            log["adjusted"] = adj_s[bad].to_numpy()
            log["local_median"] = med[bad].to_numpy()
            log["ratio"] = ratio[bad].to_numpy()
            dropped.append(log)

    keep = pd.concat(keep_flags) if keep_flags else pd.Series(dtype=bool)
    bad_index = keep[~keep].index
    cleaned = facts.drop(index=bad_index)
    log = (pd.concat(dropped, ignore_index=True) if dropped
           else pd.DataFrame(columns=["ticker", "filed", "end", "val", "tag",
                                      "adjusted", "local_median", "ratio"]))
    return cleaned, log


def build_market_cap(shares_reported: pd.DataFrame, split_factors: pd.DataFrame,
                     close: pd.DataFrame) -> pd.DataFrame:
    """Market cap in dollars, with share counts rebased to today's split units.

    ``close`` is Yahoo's split-back-adjusted close, so share counts have to move
    onto the same basis before the multiplication means anything.
    """
    common_cols = [c for c in close.columns if c in shares_reported.columns]
    idx = close.index
    shares = shares_reported.reindex(index=idx, columns=common_cols)
    factors = split_factors.reindex(index=idx, columns=common_cols).fillna(1.0)
    px = close.reindex(index=idx, columns=common_cols)

    return (shares * factors * px).astype("float64")


def sanitise_market_cap(market_cap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove market caps that cannot be right, judged against their own cross-section.

    This is the backstop behind :func:`clean_share_facts`. That function
    compares a fact with its neighbours, which works when one filing is wrong
    and the ones around it are fine. It is blind to a scale error repeated
    through a whole filing, and those are the expensive ones: AIG's Q2 2009
    report carries 1.35e14 shares in every tag, and the resulting market cap of
    4,548 trillion dollars made the rebuilt parent index in late 2009
    essentially one stock.

    Comparing against the day's median market cap catches it. The comparison is
    cross-sectional rather than absolute so it needs no view about price levels
    or how large the index should be in a given decade, and the thresholds are
    loose enough that no real constituent has ever come close.

    The values are dropped rather than rescaled. A number that is off by a
    factor of a thousand is probably a units mistake, but repairing it would
    mean guessing at the filer's intent, and a guessed market cap sets an index
    weight. Dropping leaves the name out of that week's cross-section, which is
    visible in the coverage table.
    """
    cfg = PARAMS.get("fundamentals", {})
    high = float(cfg.get("mcap_median_ratio_high", 500.0))
    low = float(cfg.get("mcap_median_ratio_low", 0.0002))

    positive = market_cap.where(market_cap > 0)
    median = positive.median(axis=1)
    ratio = positive.div(median, axis=0)
    bad = (ratio > high) | (ratio < low)

    cleaned = market_cap.where(~bad)
    idx = np.argwhere(bad.to_numpy())
    log = pd.DataFrame({
        "date": market_cap.index[idx[:, 0]] if len(idx) else [],
        "ticker": market_cap.columns[idx[:, 1]] if len(idx) else [],
        "market_cap": positive.to_numpy()[idx[:, 0], idx[:, 1]] if len(idx) else [],
        "cross_section_median": median.to_numpy()[idx[:, 0]] if len(idx) else [],
        "ratio": ratio.to_numpy()[idx[:, 0], idx[:, 1]] if len(idx) else [],
    })
    return cleaned, log


def build_book_to_price(book_equity: pd.DataFrame, market_cap: pd.DataFrame
                        ) -> pd.DataFrame:
    """Book equity over market cap, with non-positive book values dropped.

    Negative book equity is a real state of the world. A company that has
    bought back more stock than it has retained earnings reports it, and so do
    companies after large write-downs. It is not, however, a meaningful
    *value* signal: the ratio flips sign and a deeply negative book value would
    rank as the cheapest stock in the index. The standard treatment is to leave
    those names without a book-to-price reading and let the cross-sectional
    standardisation put them at the mean.
    """
    cols = [c for c in market_cap.columns if c in book_equity.columns]
    be = book_equity.reindex(index=market_cap.index, columns=cols)
    mc = market_cap.reindex(columns=cols)
    ratio = be.where(be > 0) / mc.where(mc > 0)
    return ratio.replace([np.inf, -np.inf], np.nan)


def build(facts: pd.DataFrame, close: pd.DataFrame, split_factors: pd.DataFrame,
          cache: bool = True) -> dict[str, pd.DataFrame]:
    """Assemble the daily shares, market-cap and book-to-price panels."""
    PATHS.ensure()
    dates = close.index
    tickers = list(close.columns)

    stale_limit = PARAMS.get("fundamentals", {}).get("max_share_staleness_days")
    facts, outlier_log = clean_share_facts(facts, split_factors)
    shares = daily_fact_panel(facts, "shares_outstanding", dates, tickers,
                              max_staleness_days=stale_limit)
    # Book equity is not subject to the staleness cut. It feeds book-to-price,
    # where a stale value degrades one style descriptor, rather than market cap,
    # where it would misweight the benchmark everything else is measured against.
    book = daily_fact_panel(facts, "book_equity", dates, tickers)
    mcap = build_market_cap(shares, split_factors, close)
    mcap, mcap_log = sanitise_market_cap(mcap)
    btp = build_book_to_price(book, mcap)

    out = {"shares_outstanding": shares, "book_equity": book,
           "market_cap": mcap, "book_to_price": btp}
    if cache:
        for name, df in out.items():
            df.to_parquet(PATHS.processed / f"{name}.parquet")
        outlier_log.to_parquet(PATHS.processed / "share_outliers_removed.parquet",
                               index=False)
        mcap_log.to_parquet(PATHS.processed / "market_cap_outliers_removed.parquet",
                            index=False)
    out["share_outliers"] = outlier_log
    out["market_cap_outliers"] = mcap_log
    return out
