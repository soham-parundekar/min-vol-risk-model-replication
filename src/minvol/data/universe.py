"""Point-in-time S&P 500 membership.

The study needs to know which stocks were actually in the index on a given
Wednesday in 2014, not which stocks are in it today. Using today's membership
to run a 2014 cross-section is survivorship bias in its purest form: every name
that failed, was acquired, or was demoted would silently vanish from history.

Two independent sources are pulled and compared. Neither is an official S&P
product, and the comparison is reported rather than assumed away.

Source A  fja05680/sp500, a public repository that maintains a dated panel of
          index constituents reconstructed from the index change announcements.
Source B  the current constituents table on Wikipedia's "List of S&P 500
          companies", which gives an independent snapshot of membership today.

B cannot validate A's history, only its endpoint. That is a real limitation and
it is stated in the reconciliation report rather than left for a reader to work
out.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import PARAMS, PATHS

FJA_REPO = "https://github.com/fja05680/sp500.git"
FJA_PANEL = "S&P 500 Historical Components & Changes (Updated).csv"
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
USER_AGENT = PARAMS["data_sources"]["sec_user_agent"]


def normalise_ticker(ticker: str) -> str:
    """Put a ticker into the dotted share-class form used throughout the study.

    The two membership sources and Yahoo disagree about share classes. Wikipedia
    writes ``BRK.B``, Yahoo wants ``BRK-B``, and some vendors use ``BRKB``. We
    standardise on the dotted form internally and convert at the point of each
    price request, so that exactly one convention travels through the pipeline.
    """
    t = str(ticker).strip().upper()
    return t.replace("-", ".")


def to_yahoo_ticker(ticker: str) -> str:
    """Convert the internal dotted form to the hyphenated form Yahoo expects."""
    return normalise_ticker(ticker).replace(".", "-")


def download_fja_panel(dest: Path | None = None, force: bool = False) -> Path:
    """Clone the membership repository into data/raw and return the panel path.

    A shallow clone is used rather than a direct file download because the raw
    file host is not reachable from every environment, while plain git over
    HTTPS is. The clone is kept so the commit hash can be recorded as the
    dataset version.
    """
    dest = dest or (PATHS.raw / "sp500_membership")
    if dest.exists() and not force:
        panel = dest / FJA_PANEL
        if panel.is_file():
            return panel

    if dest.exists():
        subprocess.run(["rm", "-rf", str(dest)], check=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "--quiet", FJA_REPO, str(dest)],
        check=True,
    )
    panel = dest / FJA_PANEL
    if not panel.is_file():
        raise FileNotFoundError(
            f"{FJA_PANEL} is missing from the clone at {dest}. "
            "The upstream repository may have renamed it."
        )
    return panel


def source_version(repo_dir: Path | None = None) -> dict[str, str]:
    """Record the commit the membership panel came from.

    Pinning the version matters because the upstream panel is revised: index
    changes get backfilled and ticker corrections land weeks later. Two runs
    against different commits can legitimately produce different universes, and
    without this a reader has no way to tell that apart from a bug.
    """
    repo_dir = repo_dir or (PATHS.raw / "sp500_membership")
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        when = subprocess.run(
            ["git", "-C", str(repo_dir), "log", "-1", "--format=%cI"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return {"commit": "unknown", "commit_date": "unknown"}
    return {"commit": sha, "commit_date": when}


def load_membership_snapshots(panel_path: Path | None = None) -> pd.DataFrame:
    """Read the dated constituent panel into a long (date, ticker) frame.

    The upstream file stores one row per change date with the full constituent
    list in a single comma-separated cell. Exploding it to long form costs a
    little memory and makes every downstream join trivial.
    """
    panel_path = panel_path or download_fja_panel()
    raw = pd.read_csv(panel_path)
    if not {"date", "tickers"}.issubset(raw.columns):
        raise ValueError(f"Unexpected columns in {panel_path.name}: {list(raw.columns)}")

    raw["date"] = pd.to_datetime(raw["date"])
    long = (
        raw.assign(ticker=raw["tickers"].str.split(","))
        .explode("ticker")
        .loc[:, ["date", "ticker"]]
    )
    long["ticker"] = long["ticker"].map(normalise_ticker)
    long = long[long["ticker"].str.len() > 0]
    return long.drop_duplicates().sort_values(["date", "ticker"]).reset_index(drop=True)


@dataclass
class Universe:
    """As-of membership lookup over a set of dated snapshots.

    ``snapshots`` holds one row per (change date, ticker). Membership on any
    other date is the most recent snapshot on or before it, which is the only
    correct reading: the index composition does not change between
    announcements.
    """

    snapshots: pd.DataFrame

    def __post_init__(self) -> None:
        self._dates = (
            self.snapshots["date"].drop_duplicates().sort_values().reset_index(drop=True)
        )
        self._by_date = {d: frozenset(g["ticker"]) for d, g in self.snapshots.groupby("date")}

    @property
    def first_date(self) -> pd.Timestamp:
        return self._dates.iloc[0]

    @property
    def last_date(self) -> pd.Timestamp:
        return self._dates.iloc[-1]

    def members_on(self, date) -> frozenset[str]:
        """Constituents in effect on ``date``.

        Returns an empty set before the panel starts rather than raising, so a
        caller sweeping a long date range does not have to special-case the
        beginning of history.
        """
        date = pd.Timestamp(date)
        idx = self._dates.searchsorted(date, side="right") - 1
        if idx < 0:
            return frozenset()
        return self._by_date[self._dates.iloc[idx]]

    def panel_for_dates(self, dates) -> pd.DataFrame:
        """Long (date, ticker) membership for an arbitrary list of dates."""
        rows = []
        for d in pd.DatetimeIndex(dates):
            for t in sorted(self.members_on(d)):
                rows.append((d, t))
        return pd.DataFrame(rows, columns=["date", "ticker"])

    def all_tickers(self, start=None, end=None) -> list[str]:
        """Every ticker that was ever a member inside the window.

        This is the download list. It is deliberately the union over time and
        not today's membership, which is the whole point of doing this.
        """
        df = self.snapshots
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return sorted(df["ticker"].unique())


def load_wikipedia_current(cache: Path | None = None, force: bool = False) -> pd.DataFrame:
    """Current constituents plus GICS labels from Wikipedia.

    GICS sector is collected for reference only. The risk model and the sector
    constraint both run on SIC-derived groups, because GICS is unavailable for
    companies that have already left the index and a classification that only
    covers survivors would reintroduce the bias we are trying to measure.
    """
    cache = cache or (PATHS.raw / "wikipedia_sp500_current.html")
    if force or not cache.is_file():
        import requests

        resp = requests.get(WIKI_URL, headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(resp.text, encoding="utf-8")

    tables = pd.read_html(cache)
    constituents = next(
        (t for t in tables if "Symbol" in t.columns and "GICS Sector" in t.columns), None
    )
    if constituents is None:
        raise ValueError(
            "No constituents table on the cached Wikipedia page. "
            "The article layout has probably changed."
        )
    out = constituents.rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "gics_sector",
            "GICS Sub-Industry": "gics_sub_industry",
            "CIK": "cik",
            "Date added": "date_added",
        }
    )
    out["ticker"] = out["ticker"].map(normalise_ticker)
    out["cik"] = pd.to_numeric(out["cik"], errors="coerce").astype("Int64")
    keep = ["ticker", "name", "gics_sector", "gics_sub_industry", "cik", "date_added"]
    return out.loc[:, [c for c in keep if c in out.columns]]


def reconcile(universe: Universe, wiki: pd.DataFrame) -> dict[str, object]:
    """Compare the panel's most recent snapshot against Wikipedia's live list.

    Perfect agreement is not expected and would in fact be mildly suspicious:
    the panel is rebuilt periodically and Wikipedia is edited within hours of an
    index announcement, so the two are measured on different days. What matters
    is that the disagreement is small and consists of names that changed hands
    in the gap, rather than a systematic difference in what either source thinks
    the index is.
    """
    latest = set(universe.members_on(universe.last_date))
    current = set(wiki["ticker"])
    only_panel = sorted(latest - current)
    only_wiki = sorted(current - latest)
    overlap = len(latest & current)
    return {
        "panel_as_of": str(universe.last_date.date()),
        "panel_count": len(latest),
        "wikipedia_count": len(current),
        "overlap": overlap,
        "agreement_pct": round(100.0 * overlap / max(len(current), 1), 2),
        "in_panel_not_wikipedia": only_panel,
        "in_wikipedia_not_panel": only_wiki,
    }


def build(force: bool = False) -> tuple[Universe, dict[str, object]]:
    """Fetch both sources, write the processed panel, and return it with the check."""
    PATHS.ensure()
    panel_path = download_fja_panel(force=force)
    snapshots = load_membership_snapshots(panel_path)
    universe = Universe(snapshots)

    wiki = load_wikipedia_current(force=force)
    check = reconcile(universe, wiki)
    check.update(source_version())

    snapshots.to_parquet(PATHS.processed / "universe_snapshots.parquet", index=False)
    wiki.to_parquet(PATHS.processed / "wikipedia_current.parquet", index=False)
    return universe, check
