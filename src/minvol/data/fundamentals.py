"""Point-in-time fundamentals from the SEC XBRL company-facts API.

Every XBRL fact carries two dates, and confusing them is the most expensive
mistake available in this kind of study.

``end``    the last day of the fiscal period the number describes.
``filed``  the day the filing that contained it reached EDGAR.

Only ``filed`` says when a number became public. Apple's oldest
``StockholdersEquity`` fact describes the period ending 2006-09-30 and was filed
on 2009-10-27, because that is when Apple's first XBRL exhibit appeared with
comparative history attached. Key the panel off ``end`` and the 2006
cross-section quietly receives a number that nobody could read until three
years later. Book-to-price looks sharper, the optimiser leans on it, and the
backtest is wrong in a way that never raises an error.

Nothing here exposes a lookup by fiscal period. The only retrieval function is
:func:`as_of`, which filters on ``filed``, and ``tests/`` asserts the property
directly.

Three things about EDGAR that this module exists to absorb:

**companyfacts, not companyconcept.** The per-concept endpoint is not reliable.
Ask it for Coca-Cola's ``EntityCommonStockSharesOutstanding`` and it returns a
200 with an empty fact list; the same concept pulled from that company's
companyfacts document has 71 facts going back to 2009. One request per
registrant against companyfacts avoids the whole class of problem.

**No single share-count tag covers the index.** Most filers report
``dei:EntityCommonStockSharesOutstanding``. Meta has never used it, or
``us-gaap:CommonStockSharesOutstanding`` either, and only ever reports a
weighted average share count. So the share count is a cascade: take the
preferred tag where it exists on a given date and fall back only to fill gaps.
``CommonStockSharesIssued`` is deliberately *not* in the cascade. Coca-Cola
reports 7.04bn issued against 4.30bn outstanding, a 64% overstatement, because
issued includes treasury stock.

**A ticker can point at the wrong company.** EDGAR's ticker file maps to
whichever registrant holds the symbol now. ``XOM`` resolves to ExxonMobil
Holdings Corp, a successor entity with three filings, while seventeen years of
Exxon history sit under the predecessor CIK. Facts are therefore pulled for
every candidate CIK a ticker resolves to and merged, which is the right
treatment for a holding-company reorganisation: the economic entity is
continuous even though the filer id is not.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

from ..config import PARAMS, PATHS

SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SEC_TICKERS = "https://www.sec.gov/files/company_tickers.json"

# Ordered by preference. Earlier entries win on any date where they have a
# value; later entries only fill gaps.
EQUITY_TAGS = [
    ("us-gaap", "StockholdersEquity"),
    ("us-gaap", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
]

SHARE_TAGS = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
    # A period average rather than a point-in-time count, so it moves slightly
    # differently across a buyback. For index weights that difference is far
    # smaller than having no share count at all.
    ("us-gaap", "WeightedAverageNumberOfSharesOutstandingBasic"),
    # Last resort, and the only consolidated count some dual-class filers
    # publish without a class dimension. Tyson reports nothing else at the top
    # level. Diluted overstates the count by the option overhang, typically
    # well under one percent for a large cap.
    ("us-gaap", "WeightedAverageNumberOfDilutedSharesOutstanding"),
]

FIELD_TAGS = {"book_equity": EQUITY_TAGS, "shares_outstanding": SHARE_TAGS}


class SecClient:
    """A rate-limited EDGAR session.

    The SEC asks for a descriptive User-Agent and caps requests at ten per
    second. Exceeding it earns a block on the whole network, so the limiter sits
    below the published ceiling. Extracted facts are cached per CIK; the raw
    companyfacts documents run to tens of megabytes each and are not kept.
    """

    def __init__(self, cache_dir: Path | None = None, rate_per_sec: float | None = None):
        cfg = PARAMS["data_sources"]
        self.cache_dir = cache_dir or (PATHS.raw / "sec_facts_extracted")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = 1.0 / float(rate_per_sec or cfg["sec_rate_limit_per_sec"])
        self._last = 0.0
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": cfg["sec_user_agent"],
            "Accept-Encoding": "gzip, deflate",
        })

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def company_facts(self, cik: int) -> list[dict] | None:
        """Extracted facts for one registrant, or None if EDGAR has no document.

        The response is filtered down to the handful of tags this study uses
        before anything touches disk.
        """
        cache = self.cache_dir / f"{cik:010d}.json"
        if cache.is_file():
            txt = cache.read_text(encoding="utf-8")
            return None if txt == "null" else json.loads(txt)

        self._throttle()
        try:
            resp = self.session.get(SEC_FACTS.format(cik=cik), timeout=120)
        except requests.RequestException:
            return None
        if resp.status_code != 200:
            if resp.status_code == 404:
                cache.write_text("null", encoding="utf-8")
            return None
        try:
            payload = resp.json()
        except ValueError:
            return None

        rows = _extract_tags(payload, cik)
        cache.write_text(json.dumps(rows), encoding="utf-8")
        return rows


def _extract_tags(payload: dict, cik: int) -> list[dict]:
    """Pull the tags this study needs out of a full companyfacts document."""
    facts = payload.get("facts", {})
    rows: list[dict] = []
    for field, tags in FIELD_TAGS.items():
        for rank, (taxonomy, tag) in enumerate(tags):
            block = facts.get(taxonomy, {}).get(tag)
            if not block:
                continue
            for unit_name, entries in block.get("units", {}).items():
                for f in entries:
                    if "filed" not in f or "val" not in f or "end" not in f:
                        continue
                    rows.append({
                        "cik": cik, "field": field, "tag": tag, "tag_rank": rank,
                        "unit": unit_name, "end": f["end"], "filed": f["filed"],
                        "val": f["val"], "form": f.get("form"), "accn": f.get("accn"),
                    })
    return rows


def load_cik_candidates(client: SecClient, extra: pd.DataFrame | None = None
                        ) -> dict[str, list[int]]:
    """Every CIK a ticker plausibly maps to, most authoritative first.

    EDGAR's own file goes first because it is the current, canonical mapping.
    The Wikipedia constituents table contributes a second candidate for names
    where the two disagree, which is how the predecessor filer for a
    reorganised company gets picked up.
    """
    cache = PATHS.raw / "sec_company_tickers.json"
    if cache.is_file():
        payload = json.loads(cache.read_text(encoding="utf-8"))
    else:
        client._throttle()
        resp = client.session.get(SEC_TICKERS, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        cache.write_text(json.dumps(payload), encoding="utf-8")

    out: dict[str, list[int]] = {}
    for rec in payload.values():
        tick = str(rec["ticker"]).strip().upper().replace("-", ".")
        cik = int(rec["cik_str"])
        out.setdefault(tick, [])
        if cik not in out[tick]:
            out[tick].append(cik)

    if extra is not None and {"ticker", "cik"}.issubset(extra.columns):
        for rec in extra.dropna(subset=["cik"]).itertuples(index=False):
            tick = str(rec.ticker).upper()
            cik = int(rec.cik)
            out.setdefault(tick, [])
            if cik not in out[tick]:
                out[tick].append(cik)

    # Predecessor filers that neither EDGAR's current mapping nor Wikipedia
    # will surface, listed with their reason in config/params.yaml.
    for tick, ciks in (PARAMS.get("fundamentals", {}).get("extra_ciks") or {}).items():
        tick = str(tick).upper()
        out.setdefault(tick, [])
        for cik in ciks:
            if int(cik) not in out[tick]:
                out[tick].append(int(cik))
    return out


def primary_cik_map(candidates: dict[str, list[int]]) -> dict[str, int]:
    """First candidate per ticker, for callers that need a single id."""
    return {t: c[0] for t, c in candidates.items() if c}


def fetch_fundamentals(tickers: list[str], candidates: dict[str, list[int]],
                       client: SecClient | None = None,
                       verbose: bool = True) -> pd.DataFrame:
    """Pull book equity and share counts for every ticker we can resolve."""
    client = client or SecClient()
    frames: list[pd.DataFrame] = []
    unresolved: list[str] = []

    for i, ticker in enumerate(tickers, 1):
        ciks = candidates.get(ticker) or []
        if not ciks:
            unresolved.append(ticker)
            continue
        if verbose and i % 100 == 0:
            print(f"  [{i}/{len(tickers)}] {ticker}", flush=True)

        rows: list[dict] = []
        for cik in ciks:
            got = client.company_facts(cik)
            if got:
                rows.extend(got)
        if not rows:
            unresolved.append(ticker)
            continue
        df = pd.DataFrame(rows)
        df["ticker"] = ticker
        frames.append(df)

    if verbose and unresolved:
        print(f"  no facts for {len(unresolved)} tickers (e.g. {unresolved[:8]})")
    if not frames:
        return pd.DataFrame(columns=["ticker", "cik", "field", "tag", "tag_rank",
                                     "unit", "end", "filed", "val", "form", "accn"])

    out = pd.concat(frames, ignore_index=True)
    out["end"] = pd.to_datetime(out["end"])
    out["filed"] = pd.to_datetime(out["filed"])
    out["val"] = pd.to_numeric(out["val"], errors="coerce")
    out = out.dropna(subset=["val"])
    # A predecessor and a successor filer can report the same fact; keep one.
    return out.drop_duplicates(
        subset=["ticker", "field", "tag", "end", "filed", "val"]
    ).reset_index(drop=True)


def as_of(facts: pd.DataFrame, date, field: str) -> pd.Series:
    """Latest value of ``field`` per ticker that was public on ``date``.

    The filter is on ``filed``, never on ``end``. Among what had been filed by
    ``date``, the preferred tag wins; within a tag, the most recent fiscal
    period wins; and a restatement of that period beats the original, because
    by ``date`` the correction was public.
    """
    date = pd.Timestamp(date)
    sub = facts[(facts["field"] == field) & (facts["filed"] <= date)]
    if sub.empty:
        return pd.Series(dtype="float64", name=field)
    best_rank = sub.groupby("ticker")["tag_rank"].transform("min")
    sub = sub[sub["tag_rank"] == best_rank]
    sub = sub.sort_values(["ticker", "end", "filed"])
    return sub.groupby("ticker")["val"].last().rename(field)


def build(tickers: list[str], wiki: pd.DataFrame | None = None,
          force: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Fetch and cache the raw fact table. Point-in-time slicing happens later."""
    PATHS.ensure()
    cache = PATHS.interim / "sec_facts.parquet"
    if cache.is_file() and not force:
        return pd.read_parquet(cache)

    client = SecClient()
    candidates = load_cik_candidates(client, extra=wiki)
    facts = fetch_fundamentals(tickers, candidates, client=client, verbose=verbose)
    facts.to_parquet(cache, index=False)
    return facts
