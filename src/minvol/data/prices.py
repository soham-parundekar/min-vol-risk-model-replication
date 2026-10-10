"""Daily prices from Yahoo, with the two traps that come with free price data.

**Survivorship.** Yahoo keeps almost nothing for companies that were acquired or
went through bankruptcy. Of the names that passed through the S&P 500 since
2008, a meaningful minority return an empty series. They are not dropped
quietly: the list is written to disk and the size of the gap it creates is
measured later by rebuilding the cap-weighted parent index and comparing it
against SPY.

**Ticker reuse.** This one is nastier because it fails silently and looks like
clean data. A ticker freed up by a delisting gets handed to an unrelated company
years later. Ask Yahoo for ``ABI`` and it answers with a company that listed in
2025, even though the S&P 500 member of that name left the index in 2008. Joined
naively, a 2025 IPO's returns would be stitched onto a 2008 index constituent.

The guard is to intersect every price series with the window in which that
ticker was actually an index member, and to treat a series whose history begins
after its membership ended as a different company.
"""

from __future__ import annotations

import time
import warnings

import pandas as pd

from ..config import PARAMS, PATHS
from .universe import Universe, to_yahoo_ticker

FIELDS = {"Close": "close", "Adj Close": "adj_close", "Volume": "volume"}

# Price history is kept for this long before a name joins the index, because a
# 252-day beta computed on the day it enters needs a year of prior returns.
PRE_MEMBERSHIP_BUFFER = pd.Timedelta(days=550)
# Trading days of price data that must land inside the membership window before
# a series is trusted.
MIN_OVERLAP_DAYS = 20


def membership_intervals(universe: Universe) -> pd.DataFrame:
    """First and last date each ticker counts as an index member.

    The subtlety is the upper edge. The panel records one snapshot per change
    date, and a ticker's last *appearance* is the final snapshot that still
    lists it. It remains a member for every day until the next snapshot, which
    is when the removal actually takes effect. Teleflex last appears in the
    snapshot of 2024-12-23 and is gone from the following one, so it is an index
    member through the end of December, not through 23 December.

    Taking the last appearance as the end date puts the price mask a few days
    out of step with :meth:`Universe.members_on`, and a stock then sits in the
    cross-section with its price blanked. ``member_end`` is therefore the day
    before the next snapshot.

    A handful of companies leave the index and return years later. Collapsing
    them to a single span is generous at the edges, and the generosity runs the
    safe way: it widens the window in which price data is accepted, while the
    exposures themselves always come from the week's actual constituent list.
    """
    snapshot_dates = (
        universe.snapshots["date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    g = universe.snapshots.groupby("ticker")["date"]
    out = pd.DataFrame({"member_start": g.min(), "last_seen": g.max()}).reset_index()

    nxt = snapshot_dates.searchsorted(out["last_seen"].to_numpy(), side="right")
    end: list[pd.Timestamp] = []
    for last_seen, idx in zip(out["last_seen"], nxt):
        if idx >= len(snapshot_dates):
            # Still a member as of the final snapshot; leave the window open.
            end.append(snapshot_dates.iloc[-1] + pd.Timedelta(days=370))
        else:
            end.append(snapshot_dates.iloc[idx] - pd.Timedelta(days=1))
    out["member_end"] = end
    return out.sort_values("ticker").reset_index(drop=True)


def _download_batch(tickers: list[str], start: str, end: str,
                    retries: int = 3) -> pd.DataFrame | None:
    """One yfinance call with a couple of retries for transient failures."""
    import yfinance as yf

    for attempt in range(retries):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df = yf.download(
                    tickers, start=start, end=end, auto_adjust=False,
                    progress=False, threads=True, timeout=60,
                )
            if df is not None and len(df):
                return df
        except Exception as exc:  # noqa: BLE001 - we want to retry on anything transient
            if attempt == retries - 1:
                print(f"  batch failed after {retries} attempts: {type(exc).__name__}: {exc}")
                return None
        time.sleep(2 * (attempt + 1))
    return None


def download_prices(tickers: list[str], start: str, end: str,
                    batch_size: int = 60, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch raw close, adjusted close and volume for a list of tickers.

    Returns wide frames indexed by date with one column per ticker, in the
    project's dotted ticker convention rather than Yahoo's hyphenated one.
    """
    yahoo_map = {to_yahoo_ticker(t): t for t in tickers}
    symbols = sorted(yahoo_map)
    frames: dict[str, list[pd.DataFrame]] = {v: [] for v in FIELDS.values()}

    for i in range(0, len(symbols), batch_size):
        chunk = symbols[i:i + batch_size]
        if verbose:
            print(f"  [{i + len(chunk):4d}/{len(symbols)}] {chunk[0]} ... {chunk[-1]}", flush=True)
        df = _download_batch(chunk, start, end)
        if df is None:
            continue
        if not isinstance(df.columns, pd.MultiIndex):
            # yfinance flattens the column index when a single symbol survives.
            df.columns = pd.MultiIndex.from_product([df.columns, [chunk[0]]])
        for yf_field, name in FIELDS.items():
            if yf_field not in df.columns.get_level_values(0):
                continue
            sub = df[yf_field].copy()
            sub = sub.rename(columns=yahoo_map)
            frames[name].append(sub)

    out: dict[str, pd.DataFrame] = {}
    for name, parts in frames.items():
        if not parts:
            out[name] = pd.DataFrame()
            continue
        merged = pd.concat(parts, axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        merged.index = pd.to_datetime(merged.index).tz_localize(None)
        out[name] = merged.sort_index().sort_index(axis=1)
    return out


def audit_coverage(prices: dict[str, pd.DataFrame], intervals: pd.DataFrame) -> pd.DataFrame:
    """Classify every requested ticker by how its price history lines up with membership.

    Four outcomes:

    ``ok``                 enough observations inside the membership window.
    ``no_data``            Yahoo returned nothing. Almost always an acquisition
                           or a bankruptcy, and the core of the survivorship gap.
    ``reused_ticker``      data exists, but all of it postdates the end of
                           membership. A different company now owns the symbol.
    ``thin_overlap``       fewer than MIN_OVERLAP_DAYS observations inside the
                           window. Too little to compute anything on.
    """
    adj = prices["adj_close"]
    rows = []
    for rec in intervals.itertuples(index=False):
        t = rec.ticker
        if t not in adj.columns:
            rows.append((t, rec.member_start, rec.member_end, 0, pd.NaT, pd.NaT, "no_data"))
            continue
        s = adj[t].dropna()
        if s.empty:
            rows.append((t, rec.member_start, rec.member_end, 0, pd.NaT, pd.NaT, "no_data"))
            continue
        first, last = s.index.min(), s.index.max()
        inside = s.loc[(s.index >= rec.member_start) & (s.index <= rec.member_end)]
        if first > rec.member_end:
            status = "reused_ticker"
        elif len(inside) < MIN_OVERLAP_DAYS:
            status = "thin_overlap"
        else:
            status = "ok"
        rows.append((t, rec.member_start, rec.member_end, len(inside), first, last, status))

    return pd.DataFrame(
        rows,
        columns=["ticker", "member_start", "member_end", "obs_in_window",
                 "price_first", "price_last", "status"],
    )


def apply_coverage_mask(prices: dict[str, pd.DataFrame], audit: pd.DataFrame,
                        ) -> dict[str, pd.DataFrame]:
    """Drop unusable series and blank out observations outside membership.

    Data from before a name joins the index is kept, within the buffer, because
    trailing statistics need it. Data from after it leaves is discarded: once a
    company is out of the parent universe its returns are irrelevant to the
    study, and keeping them is how a reused ticker sneaks back in.
    """
    usable = audit[audit["status"] == "ok"].set_index("ticker")
    out: dict[str, pd.DataFrame] = {}
    for name, df in prices.items():
        if df.empty:
            out[name] = df
            continue
        cols = [c for c in df.columns if c in usable.index]
        sub = df.loc[:, cols].copy()
        for t in cols:
            lo = usable.at[t, "member_start"] - PRE_MEMBERSHIP_BUFFER
            hi = usable.at[t, "member_end"]
            sub.loc[(sub.index < lo) | (sub.index > hi), t] = pd.NA
        out[name] = sub.astype("float64")
    return out


def total_returns(adj_close: pd.DataFrame) -> pd.DataFrame:
    """Daily total returns from the dividend- and split-adjusted close."""
    return adj_close.sort_index().pct_change(fill_method=None)


def build(universe: Universe, force: bool = False, verbose: bool = True
          ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Download, audit and mask the constituent price panel; cache to parquet."""
    PATHS.ensure()
    s = PARAMS["sample"]
    start, end = s["price_start"], s["price_end"]
    cache = {name: PATHS.interim / f"prices_{name}.parquet" for name in FIELDS.values()}
    audit_path = PATHS.processed / "price_coverage_audit.parquet"

    if not force and all(p.is_file() for p in cache.values()) and audit_path.is_file():
        prices = {n: pd.read_parquet(p) for n, p in cache.items()}
        return prices, pd.read_parquet(audit_path)

    tickers = universe.all_tickers(start=start, end=end)
    if verbose:
        print(f"Downloading {len(tickers)} tickers, {start} to {end}")
    raw = download_prices(tickers, start, end, verbose=verbose)

    intervals = membership_intervals(universe)
    intervals = intervals[intervals["ticker"].isin(tickers)]
    audit = audit_coverage(raw, intervals)
    masked = apply_coverage_mask(raw, audit)

    for name, df in masked.items():
        df.to_parquet(cache[name])
    audit.to_parquet(audit_path, index=False)
    # The unmasked panel is kept so the masking step can be re-run or inspected
    # without paying for the download again.
    for name, df in raw.items():
        if not df.empty:
            df.to_parquet(PATHS.raw / f"prices_unmasked_{name}.parquet")
    return masked, audit
