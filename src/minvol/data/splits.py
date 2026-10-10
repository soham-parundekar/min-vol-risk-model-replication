"""Split factors, needed to put share counts and prices on the same basis.

Yahoo back-adjusts its close series for splits. Apple's close on 2019-06-28
reads 49.48, not the 197.92 that actually printed that day, because the 4-for-1
split in August 2020 is carried backwards through the whole history. Every
historical price is therefore quoted in *today's* share units.

Share counts are not. ``EntityCommonStockSharesOutstanding`` from a 2019 10-Q
says roughly 4.5 billion shares, which is what existed in 2019.

Multiply those two together and the 2019 market cap comes out four times too
small. The fix is to express historical share counts in today's units by
multiplying them by every split that happened afterwards:

    shares_today_basis(t) = shares_reported(t) * prod(ratio_s for s > t)

Apple in mid-2019 then gives 4.5bn * 4 * 49.48, about 896 billion dollars, which
is roughly what Apple was worth. Skip this step and size, book-to-price and the
parent index weights are all wrong, with the error concentrated in exactly the
mega-cap names that dominate a cap-weighted benchmark.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import PARAMS, PATHS
from .universe import to_yahoo_ticker

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
BROWSER_UA = "Mozilla/5.0 (compatible; SohamParundekar-MFEResearch; research)"


def fetch_splits(tickers: list[str], start: str | None = None, end: str | None = None,
                 cache_path: Path | None = None, force: bool = False,
                 verbose: bool = True) -> pd.DataFrame:
    """Split events per ticker, read from the Yahoo chart endpoint.

    yfinance's bulk download drops the events payload, so this goes to the chart
    endpoint directly, which returns splits alongside the price series. One
    request per ticker, cached to a single JSON so a rerun costs nothing.
    """
    s = PARAMS["sample"]
    start = start or s["price_start"]
    end = end or s["price_end"]
    cache_path = cache_path or (PATHS.raw / "yahoo_splits.json")

    cached: dict[str, list] = {}
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text(encoding="utf-8"))

    todo = [t for t in tickers if t not in cached]
    if todo:
        p1 = int(dt.datetime.fromisoformat(start).timestamp())
        p2 = int(dt.datetime.fromisoformat(end).timestamp())
        session = requests.Session()
        session.headers.update({"User-Agent": BROWSER_UA})
        for i, ticker in enumerate(todo, 1):
            if verbose and i % 100 == 0:
                print(f"  splits [{i}/{len(todo)}] {ticker}", flush=True)
            url = CHART_URL.format(symbol=to_yahoo_ticker(ticker))
            try:
                resp = session.get(
                    url,
                    params={"period1": p1, "period2": p2, "interval": "1d",
                            "events": "split"},
                    timeout=30,
                )
                payload = resp.json()
            except (requests.RequestException, ValueError):
                cached[ticker] = []
                continue
            result = (payload.get("chart") or {}).get("result")
            if not result:
                cached[ticker] = []
                continue
            events = (result[0].get("events") or {}).get("splits") or {}
            cached[ticker] = [
                {"date": int(ev["date"]),
                 "numerator": float(ev["numerator"]),
                 "denominator": float(ev["denominator"])}
                for ev in events.values()
            ]
            time.sleep(0.05)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cached), encoding="utf-8")

    rows = []
    for ticker, events in cached.items():
        for ev in events:
            denom = ev["denominator"] or 1.0
            rows.append({
                "ticker": ticker,
                "date": pd.Timestamp(dt.datetime.utcfromtimestamp(ev["date"]).date()),
                "ratio": ev["numerator"] / denom,
            })
    if not rows:
        return pd.DataFrame(columns=["ticker", "date", "ratio"])
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


def cumulative_factor_panel(splits: pd.DataFrame, dates: pd.DatetimeIndex,
                            tickers: list[str]) -> pd.DataFrame:
    """Factor converting a share count reported on date t into today's units.

    The value on date t is the product of every split ratio strictly after t, so
    the series steps *down* through time and reaches 1.0 once no splits remain.
    Tickers with no splits are all ones.
    """
    panel = pd.DataFrame(1.0, index=dates, columns=tickers)
    if splits.empty:
        return panel
    for ticker, grp in splits.groupby("ticker"):
        if ticker not in panel.columns:
            continue
        col = pd.Series(1.0, index=dates)
        for ev in grp.sort_values("date", ascending=False).itertuples(index=False):
            # A share count reported before the split date needs scaling up.
            col.loc[col.index < ev.date] *= ev.ratio
        panel[ticker] = col
    return panel
