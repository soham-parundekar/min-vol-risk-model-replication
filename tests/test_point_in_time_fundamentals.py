"""The look-ahead tests. These are the ones that matter most.

Every other bug in this project makes a number wrong in a way that eventually
looks wrong. A look-ahead bug makes the numbers look *better*, which is why it
survives review: a backtest that improves does not invite scrutiny.

The specific trap is that XBRL facts carry two dates and only one of them says
when the number became public. Apple's oldest ``StockholdersEquity`` fact
describes the period ending 2006-09-30 and was filed on 2009-10-27, so keying
off the period end would hand a 2006 cross-section a figure nobody could read
for another three years.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minvol.config import PATHS  # noqa: E402
from minvol.data import fundamentals, marketcap  # noqa: E402

FACTS_PATH = PATHS.interim / "sec_facts.parquet"
needs_data = pytest.mark.skipif(
    not FACTS_PATH.is_file(),
    reason="run scripts/01_build_data.py first",
)


@pytest.fixture(scope="module")
def facts() -> pd.DataFrame:
    return pd.read_parquet(FACTS_PATH)


def _synthetic_facts() -> pd.DataFrame:
    """A fact whose period end long predates its filing date, as Apple's does."""
    return pd.DataFrame({
        "ticker": ["TEST", "TEST", "TEST"],
        "cik": [1, 1, 1],
        "field": ["book_equity"] * 3,
        "tag": ["StockholdersEquity"] * 3,
        "tag_rank": [0, 0, 0],
        "unit": ["USD"] * 3,
        "end": pd.to_datetime(["2006-09-30", "2009-09-30", "2010-09-30"]),
        "filed": pd.to_datetime(["2009-10-27", "2009-10-27", "2010-10-27"]),
        "val": [100.0, 200.0, 300.0],
        "form": ["10-K"] * 3,
        "accn": ["x"] * 3,
    })


def test_as_of_ignores_facts_filed_later():
    """Nothing filed after the as-of date may appear in the answer."""
    df = _synthetic_facts()

    # Before anything was filed, there is no value at all, even though a fact
    # describing a 2006 period exists in the table.
    assert as_of_len(df, "2007-01-01") == 0
    assert as_of_len(df, "2009-10-26") == 0

    # On the filing date the latest reported period becomes available.
    assert fundamentals.as_of(df, "2009-10-27", "book_equity")["TEST"] == 200.0

    # The 2010 figure only arrives with its own filing.
    assert fundamentals.as_of(df, "2010-10-26", "book_equity")["TEST"] == 200.0
    assert fundamentals.as_of(df, "2010-10-27", "book_equity")["TEST"] == 300.0


def as_of_len(df: pd.DataFrame, date: str) -> int:
    return len(fundamentals.as_of(df, date, "book_equity"))


def test_as_of_prefers_latest_period_not_latest_row():
    """A 10-K carries comparatives; the current period must win, not the oldest."""
    df = _synthetic_facts()
    # Both 2006 and 2009 arrive on 2009-10-27. The answer must be the 2009 one.
    assert fundamentals.as_of(df, "2009-10-27", "book_equity")["TEST"] == 200.0


def test_restatement_of_same_period_supersedes_original():
    df = _synthetic_facts()
    restated = pd.DataFrame({
        "ticker": ["TEST"], "cik": [1], "field": ["book_equity"],
        "tag": ["StockholdersEquity"], "tag_rank": [0], "unit": ["USD"],
        "end": pd.to_datetime(["2010-09-30"]), "filed": pd.to_datetime(["2011-02-01"]),
        "val": [350.0], "form": ["10-K/A"], "accn": ["y"],
    })
    df = pd.concat([df, restated], ignore_index=True)
    assert fundamentals.as_of(df, "2011-01-31", "book_equity")["TEST"] == 300.0
    assert fundamentals.as_of(df, "2011-02-01", "book_equity")["TEST"] == 350.0


def test_point_in_time_series_is_monotone_in_period():
    """The step series never steps backwards to an older fiscal period."""
    df = _synthetic_facts()
    step = marketcap.point_in_time_series(df)
    assert list(step.values) == [200.0, 300.0]
    assert list(step.index) == [pd.Timestamp("2009-10-27"), pd.Timestamp("2010-10-27")]


@needs_data
def test_facts_dated_after_their_filing_are_rare_and_harmless(facts):
    """A handful of EDGAR facts carry a period end after the filing date.

    Mostly cover-page share counts dated a few days forward, plus the odd
    tagging slip. There were 14 of these out of roughly half a million facts
    when this was written, which is a quirk of the source rather than a defect
    in the pipeline.

    They cannot cause look-ahead either way, because :func:`as_of` filters on
    the filing date and never on the period end. The test pins both claims: the
    count stays negligible, and no such fact becomes visible before it was
    filed.
    """
    future = facts[facts["end"] > facts["filed"] + pd.Timedelta(days=5)]
    share = len(future) / max(len(facts), 1)
    assert share < 0.001, (
        f"{len(future)} of {len(facts)} facts ({share:.4%}) are dated after "
        f"their filing date, which is too many to treat as source noise"
    )

    # The property that matters is not that a particular number is absent
    # beforehand. The same share count often appears in two consecutive
    # filings, so a value test gives false alarms. What has to hold is that
    # whatever as_of returns is traceable to a fact filed by then.
    rng = np.random.default_rng(11)
    tickers = rng.choice(facts["ticker"].dropna().unique(), size=40, replace=False)
    dates = pd.date_range("2012-06-29", "2026-06-30", freq="2YE")
    sub = facts[facts["ticker"].isin(tickers)]

    for field in ("book_equity", "shares_outstanding"):
        for when in dates:
            got = fundamentals.as_of(sub, when, field)
            visible = sub[(sub["field"] == field) & (sub["filed"] <= when)]
            for ticker, value in got.dropna().items():
                known = visible.loc[visible["ticker"] == ticker, "val"]
                assert (known == value).any(), (
                    f"{ticker} {field} on {when.date()}: as_of returned {value}, "
                    f"which matches no fact filed by then"
                )


@needs_data
def test_apple_equity_unavailable_before_first_xbrl_filing(facts):
    """The motivating case, checked against the real EDGAR data.

    Apple reports a 2006 balance sheet, but only from 2009 onwards. Asking for
    Apple's book equity in 2007 must return nothing.
    """
    aapl = facts[(facts["ticker"] == "AAPL") & (facts["field"] == "book_equity")]
    if aapl.empty:
        pytest.skip("Apple fundamentals not in the cached panel")

    early_periods = aapl[aapl["end"] < pd.Timestamp("2008-01-01")]
    assert len(early_periods) > 0, "expected Apple to report pre-2008 comparatives"
    assert early_periods["filed"].min() >= pd.Timestamp("2009-01-01"), (
        "a pre-2008 Apple period appears to have been filed before 2009"
    )

    got = fundamentals.as_of(aapl, "2007-06-30", "book_equity")
    assert "AAPL" not in got.index or pd.isna(got.get("AAPL")), (
        "Apple book equity is available in 2007, which is a look-ahead leak"
    )


@needs_data
def test_daily_panel_never_leads_its_filing_date(facts):
    """Spot-check the assembled panel against the facts it was built from."""
    sub = facts[facts["field"] == "book_equity"]
    tickers = sub["ticker"].dropna().unique()[:25]
    dates = pd.date_range("2012-01-31", "2024-12-31", freq="YE")

    panel = marketcap.daily_fact_panel(sub, "book_equity", dates, list(tickers))
    for ticker in tickers:
        first_filed = sub.loc[sub["ticker"] == ticker, "filed"].min()
        if pd.isna(first_filed):
            continue
        before = panel.loc[panel.index < first_filed, ticker]
        assert before.isna().all(), (
            f"{ticker} has a book-equity value before its first filing on "
            f"{first_filed.date()}"
        )
