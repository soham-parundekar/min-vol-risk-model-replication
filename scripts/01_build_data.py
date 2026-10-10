"""Acquire and clean every dataset the study uses.

Downloads run against public endpoints and are cached under ``data/``, so a
second run costs almost nothing. Expect fifteen to twenty minutes on a cold
cache, most of it spent pulling one company-facts document per registrant from
EDGAR.

    python scripts/01_build_data.py [--force]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from minvol.config import PARAMS, PATHS  # noqa: E402
from minvol.data import (  # noqa: E402
    external_factors, fundamentals, marketcap, prices, sic_industries, splits, universe,
)


def main(force: bool = False) -> None:
    started = time.time()
    PATHS.ensure()
    sample = PARAMS["sample"]

    print("1. Point-in-time S&P 500 membership")
    uni, check = universe.build(force=force)
    print(f"   panel as of {check['panel_as_of']}, commit {check['commit'][:10]}")
    print(f"   agreement with Wikipedia: {check['agreement_pct']}% "
          f"({check['overlap']}/{check['wikipedia_count']} names)")
    print(f"   differences: only in panel {check['in_panel_not_wikipedia']}, "
          f"only in Wikipedia {check['in_wikipedia_not_panel']}")
    pd.Series(
        {k: str(v) for k, v in check.items()}
    ).to_frame("value").to_parquet(PATHS.processed / "universe_reconciliation.parquet")

    print("\n2. Daily prices")
    px, audit = prices.build(uni, force=force)
    counts = audit["status"].value_counts().to_dict()
    print(f"   {counts}")
    print(f"   usable panel: {px['adj_close'].shape[1]} tickers, "
          f"{px['adj_close'].shape[0]} sessions")

    usable = audit.loc[audit["status"] == "ok", "ticker"].tolist()

    print("\n3. Split history")
    split_events = splits.fetch_splits(usable, force=force)
    split_events.to_parquet(PATHS.processed / "splits.parquet", index=False)
    print(f"   {len(split_events)} events across "
          f"{split_events['ticker'].nunique()} tickers")

    print("\n4. SEC fundamentals")
    wiki = universe.load_wikipedia_current()
    client = fundamentals.SecClient()
    candidates = fundamentals.load_cik_candidates(client, extra=wiki)
    facts = fundamentals.build(usable, wiki=wiki, force=force)
    print(f"   {len(facts):,} facts for {facts['ticker'].nunique()} tickers")

    print("\n5. SIC codes and industry groups")
    primary = fundamentals.primary_cik_map(
        {t: c for t, c in candidates.items() if t in set(usable)}
    )
    industries = sic_industries.build(primary, force=force, verbose=False)
    print(f"   {len(industries)} tickers mapped")
    print(industries["industry"].value_counts().to_string())

    print("\n6. Market cap and book-to-price")
    split_factors = splits.cumulative_factor_panel(
        split_events, px["close"].index, list(px["close"].columns)
    )
    caps = marketcap.build(facts, px["close"], split_factors)
    print(f"   share facts removed as outliers: {len(caps['share_outliers'])}")
    print(f"   market-cap cells removed as implausible: "
          f"{len(caps['market_cap_outliers'])}")

    print("\n7. External factor data")
    external_factors.build(force=force)
    print("   Fama-French, AQR BAB, FRED, ETF prices cached")

    print(f"\nFinished in {time.time() - started:.0f}s "
          f"(sample {sample['price_start']} to {sample['price_end']})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ignore caches and refetch")
    main(**vars(ap.parse_args()))
