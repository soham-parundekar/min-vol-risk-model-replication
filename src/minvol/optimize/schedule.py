"""Rebalance calendars for the two index rules.

MSCI reviews its minimum volatility indexes semi-annually, effective at the
close of the last business day of May and November. S&P rebalances the Low
Volatility index quarterly, effective after the close of the third Friday of
February, May, August and November.

The difference is not incidental and it is one of the things the study is
actually comparing. Quarterly rebalancing keeps the portfolio closer to its
target at the cost of trading four times a year; semi-annual trades half as
often and spends the intervening months drifting. MSCI pairs its slower
schedule with a hard 10% turnover cap, so the two design choices reinforce each
other: trade rarely, and when you do, do not trade much.

Both rules resolve to an actual trading day, because a target date can fall on
a holiday and a rebalance that silently lands on no date at all is worse than
one that lands a day early.
"""

from __future__ import annotations

import pandas as pd


def last_business_day_dates(calendar: pd.DatetimeIndex, months: list[int],
                            start=None, end=None) -> pd.DatetimeIndex:
    """Last trading day of each listed month inside the window."""
    days = pd.DatetimeIndex(calendar).sort_values()
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    frame = pd.Series(days, index=days)
    chosen = frame[frame.index.month.isin(months)]
    if chosen.empty:
        return pd.DatetimeIndex([])
    grouped = chosen.groupby([chosen.index.year, chosen.index.month]).max()
    return pd.DatetimeIndex(sorted(grouped.to_numpy()))


def third_friday_dates(calendar: pd.DatetimeIndex, months: list[int],
                       start=None, end=None) -> pd.DatetimeIndex:
    """Third Friday of each listed month, pulled back to a trading day if closed.

    Good Friday is the case that matters here: the third Friday of a month is
    occasionally a market holiday, and the effective rebalance moves to the
    preceding session rather than vanishing.
    """
    days = pd.DatetimeIndex(calendar).sort_values()
    if start is not None:
        days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    if len(days) == 0:
        return pd.DatetimeIndex([])

    out: list[pd.Timestamp] = []
    years = range(days[0].year, days[-1].year + 1)
    for year in years:
        for month in months:
            fridays = pd.date_range(
                start=pd.Timestamp(year=year, month=month, day=1),
                end=pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0),
                freq="W-FRI",
            )
            if len(fridays) < 3:
                continue
            target = fridays[2]
            idx = days.searchsorted(target, side="right") - 1
            if idx < 0:
                continue
            landed = days[idx]
            if landed.year == year and landed.month == month:
                out.append(landed)
    return pd.DatetimeIndex(sorted(set(out)))


def rebalance_dates(calendar: pd.DatetimeIndex, months: list[int], rule: str,
                    start=None, end=None) -> pd.DatetimeIndex:
    """Dispatch to the calendar rule named in config."""
    if rule == "last_business_day":
        return last_business_day_dates(calendar, months, start, end)
    if rule == "third_friday":
        return third_friday_dates(calendar, months, start, end)
    raise ValueError(f"Unknown rebalance rule: {rule!r}")
