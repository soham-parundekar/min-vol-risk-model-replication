"""Industry groups from SIC codes, using the Fama-French 12-industry scheme.

MSCI runs its sector constraint on GICS. We cannot. GICS is licensed, and the
only free source of it is a table of companies that are in the index *today*,
which is precisely the set that would reintroduce survivorship bias if we used
it to classify history.

So the study runs on SIC instead, read from each registrant's EDGAR submissions
record and bucketed into Ken French's 12 industries. Two things this costs us,
both of which stay in the report rather than being smoothed over:

SIC is coarser and older than GICS. It predates the software and biotech
industries as they exist now, and it files a lot of modern companies under
codes that made sense in 1987. The industry factors and the sector constraint
are therefore both approximations of what MSCI does, not equivalents.

EDGAR publishes one current SIC per filer, not a history. A company that
reinvented itself carries today's code backwards through the whole sample. For
the S&P 500 the effect is small, because large-cap industry migration is rare,
but it is not zero.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ..config import PARAMS, PATHS

KF_SICCODES12 = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/Siccodes12.zip"
)
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"

# Ken French's file labels the catch-all group "Other" and gives it no explicit
# ranges, so anything that matches nothing lands here by construction.
FALLBACK_GROUP = ("12", "Other")

_HEADER = re.compile(r"^\s*(\d+)\s+(\w+)\s+(.*)$")
_RANGE = re.compile(r"^\s+(\d{4})-(\d{4})")


def download_siccodes12(dest: Path | None = None, force: bool = False) -> Path:
    """Fetch and unpack Ken French's Siccodes12 definition file."""
    dest = dest or (PATHS.raw / "Siccodes12.txt")
    if dest.is_file() and not force:
        return dest
    resp = requests.get(
        KF_SICCODES12,
        headers={"User-Agent": PARAMS["data_sources"]["sec_user_agent"]},
        timeout=90,
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))
    return dest


def parse_siccodes12(path: Path | None = None) -> pd.DataFrame:
    """Turn the definition file into one row per SIC range.

    The file is a fixed-layout listing: an unindented line opens a group, and
    the indented four-digit ranges beneath it belong to that group until the
    next header.
    """
    path = path or download_siccodes12()
    rows = []
    code = abbrev = None
    for line in path.read_text(encoding="latin-1").splitlines():
        if not line.strip():
            continue
        rng = _RANGE.match(line)
        if rng and code is not None:
            rows.append({
                "industry_code": code,
                "industry": abbrev,
                "sic_lo": int(rng.group(1)),
                "sic_hi": int(rng.group(2)),
            })
            continue
        head = _HEADER.match(line)
        if head and not line.startswith("  "):
            code, abbrev = head.group(1), head.group(2)
    if not rows:
        raise ValueError(f"No SIC ranges parsed from {path}; the layout has changed.")
    return pd.DataFrame(rows)


def fetch_sic_codes(cik_map: dict[str, int], client=None, verbose: bool = True
                    ) -> pd.DataFrame:
    """Read the current SIC code for each resolvable ticker from EDGAR."""
    from .fundamentals import SecClient

    client = client or SecClient()
    cache_dir = PATHS.raw / "sec_submissions"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, (ticker, cik) in enumerate(sorted(cik_map.items()), 1):
        if verbose and i % 200 == 0:
            print(f"  [{i}/{len(cik_map)}] {ticker}", flush=True)
        cache = cache_dir / f"{cik:010d}.json"
        if cache.is_file():
            try:
                payload = json.loads(cache.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
        else:
            client._throttle()
            try:
                resp = client.session.get(SEC_SUBMISSIONS.format(cik=cik), timeout=30)
            except requests.RequestException:
                continue
            if resp.status_code != 200:
                continue
            payload = resp.json()
            # The filing index is large and we need four fields from it.
            slim = {k: payload.get(k) for k in
                    ("cik", "name", "sic", "sicDescription", "tickers", "exchanges")}
            cache.write_text(json.dumps(slim), encoding="utf-8")
            payload = slim

        if not isinstance(payload, dict):
            continue
        sic = payload.get("sic")
        if sic in (None, "", "0000"):
            continue
        rows.append({
            "ticker": ticker,
            "cik": cik,
            "sic": int(sic),
            "sic_description": payload.get("sicDescription"),
            "company_name": payload.get("name"),
        })
    return pd.DataFrame(rows)


def assign_industries(sic_table: pd.DataFrame, ranges: pd.DataFrame | None = None
                      ) -> pd.DataFrame:
    """Attach a Fama-French 12 industry label to every ticker with a SIC code."""
    ranges = ranges if ranges is not None else parse_siccodes12()
    lo = ranges["sic_lo"].to_numpy()
    hi = ranges["sic_hi"].to_numpy()
    codes = ranges["industry_code"].to_numpy()
    names = ranges["industry"].to_numpy()

    out = sic_table.copy()
    assigned_code, assigned_name = [], []
    for sic in out["sic"].to_numpy():
        hit = (sic >= lo) & (sic <= hi)
        if hit.any():
            idx = hit.argmax()
            assigned_code.append(codes[idx])
            assigned_name.append(names[idx])
        else:
            assigned_code.append(FALLBACK_GROUP[0])
            assigned_name.append(FALLBACK_GROUP[1])
    out["industry_code"] = assigned_code
    out["industry"] = assigned_name
    return out


def build(cik_map: dict[str, int], force: bool = False, verbose: bool = True
          ) -> pd.DataFrame:
    """Fetch SIC codes, map them to FF12, and cache the result."""
    PATHS.ensure()
    cache = PATHS.processed / "industries.parquet"
    if cache.is_file() and not force:
        return pd.read_parquet(cache)
    sic_table = fetch_sic_codes(cik_map, verbose=verbose)
    industries = assign_industries(sic_table)
    industries.to_parquet(cache, index=False)
    return industries
