"""Published factor returns used to judge the replicas from the outside.

These are not the factors the risk model estimates. The risk model builds its
own market, industry and style returns from the cross-section (see
``minvol.factors``). What lives here is third-party data used for attribution:
the Fama-French five factors plus momentum from the Ken French Data Library,
the Frazzini-Pedersen betting-against-beta factor from AQR, and the ten-year
Treasury yield from FRED.

The alpha test in Module G turns on BAB. A low-volatility portfolio reliably
shows a positive intercept against the usual five-factor-plus-momentum model.
Frazzini and Pedersen's argument is that this is not an anomaly needing its own
explanation but compensation for the leverage constraints that push investors
into high-beta stocks. If the intercept collapses once BAB is on the right-hand
side, the low-volatility effect is largely that story. If it survives, it is
not.

One date caveat worth knowing before reading any attribution table: Ken French
rebuilds from a CRSP database with a lag, and AQR reconstructs its full history
on each update. Both end earlier than the price data. Regressions therefore run
to the last date all three series share, and that date is reported with the
results rather than left implicit.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from ..config import PARAMS, PATHS

KF_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
AQR_BAB_DAILY = (
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/"
    "Betting-Against-Beta-Equity-Factors-Daily.xlsx"
)
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}&cosd={start}"

# AQR ships the factor panel under a block of descriptive text; the column
# header sits on this row and the series start immediately below it.
AQR_HEADER_ROW = 18
BROWSER_UA = (
    "Mozilla/5.0 (compatible; SohamParundekar-MFEResearch; parundekarsoham@gmail.com)"
)


def _kf_download(dataset: str, force: bool = False) -> Path:
    """Download and unpack one Ken French CSV archive into data/raw."""
    dest = PATHS.raw / f"{dataset}.csv"
    if dest.is_file() and not force:
        return dest
    url = f"{KF_BASE}/{dataset}_CSV.zip"
    resp = requests.get(
        url, headers={"User-Agent": PARAMS["data_sources"]["sec_user_agent"]}, timeout=120
    )
    resp.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(zf.read(name))
    return dest


def _parse_kf_daily(path: Path) -> pd.DataFrame:
    """Read a daily Ken French CSV, skipping the prose header and the copyright line.

    The files open with a few paragraphs of description, then a comma-separated
    block keyed by an eight-digit date, then a copyright line. Rows are located
    by the date pattern rather than by counting header lines, because the number
    of description paragraphs differs between datasets and changes over time.
    """
    text = path.read_text(encoding="latin-1").splitlines()
    start = next(
        i for i, line in enumerate(text)
        if line.strip() and line.split(",")[0].strip().isdigit()
        and len(line.split(",")[0].strip()) == 8
    )
    header_idx = start - 1
    columns = [c.strip() for c in text[header_idx].split(",")]
    columns[0] = "date"

    rows = []
    for line in text[start:]:
        parts = [p.strip() for p in line.split(",")]
        if not parts[0].isdigit() or len(parts[0]) != 8:
            break
        rows.append(parts)

    df = pd.DataFrame(rows, columns=columns[: len(rows[0])])
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Ken French publishes percentages and flags gaps with -99.99 / -999.
    df = df.set_index("date").sort_index()
    df = df.mask(df <= -99.0)
    return df / 100.0


def load_fama_french(force: bool = False) -> pd.DataFrame:
    """Daily FF5 factors plus the momentum factor and the risk-free rate."""
    ff5 = _parse_kf_daily(_kf_download("F-F_Research_Data_5_Factors_2x3_daily", force))
    mom = _parse_kf_daily(_kf_download("F-F_Momentum_Factor_daily", force))
    mom = mom.rename(columns={mom.columns[0]: "MOM"})
    out = ff5.join(mom[["MOM"]], how="outer")
    return out.rename(columns={"Mkt-RF": "MKT_RF"})


def load_bab(force: bool = False, market: str = "USA") -> pd.Series:
    """Daily betting-against-beta returns for one market from AQR's dataset.

    AQR serves the file only to a browser-shaped request, so the User-Agent
    here differs from the one used for SEC and Ken French. The file is cached
    on first download; it is around 30 MB and is not committed to the
    repository.
    """
    cache = PATHS.raw / "AQR_BAB_Equity_Factors_Daily.xlsx"
    if force or not cache.is_file():
        resp = requests.get(AQR_BAB_DAILY, headers={"User-Agent": BROWSER_UA}, timeout=300)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(resp.content)

    raw = pd.read_excel(cache, sheet_name="BAB Factors", header=AQR_HEADER_ROW)
    date_col = raw.columns[0]
    if market not in raw.columns:
        raise KeyError(f"{market} is not a column in the AQR BAB sheet: "
                       f"{list(raw.columns)[:15]}")
    out = raw.loc[:, [date_col, market]].rename(columns={date_col: "date", market: "BAB"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).set_index("date").sort_index()
    return pd.to_numeric(out["BAB"], errors="coerce").dropna()


def load_fred(series: str = "DGS10", start: str | None = None,
              force: bool = False) -> pd.Series:
    """One FRED series as a daily float series.

    DGS10 is quoted in percent and carries blanks on bank holidays that are not
    market holidays. Blanks are left as gaps rather than filled, so that a
    caller choosing to forward-fill does it knowingly.
    """
    start = start or PARAMS["sample"]["price_start"]
    cache = PATHS.raw / f"FRED_{series}.csv"
    if force or not cache.is_file():
        resp = requests.get(FRED_CSV.format(series=series, start=start), timeout=90)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(resp.content)

    df = pd.read_csv(cache)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    out = df.set_index(date_col)[series]
    return pd.to_numeric(out, errors="coerce").dropna()


def load_etf_prices(tickers: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    """Adjusted closes for the real ETFs and the benchmark, for tracking tests."""
    cfg = PARAMS["evaluation"]
    tickers = tickers or [*cfg["replication_targets"], cfg["benchmark_ticker"]]
    cache = PATHS.interim / "etf_prices.parquet"
    if cache.is_file() and not force:
        return pd.read_parquet(cache)

    import warnings

    import yfinance as yf

    s = PARAMS["sample"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(tickers, start=s["price_start"], end=s["price_end"],
                          auto_adjust=False, progress=False, threads=True, timeout=60)
    adj = raw["Adj Close"].copy()
    adj.index = pd.to_datetime(adj.index).tz_localize(None)
    adj = adj.sort_index()
    adj.to_parquet(cache)
    return adj


def build(force: bool = False) -> dict[str, pd.DataFrame | pd.Series]:
    """Collect every external series and cache a combined daily frame."""
    PATHS.ensure()
    ff = load_fama_french(force=force)
    bab = load_bab(force=force)
    dgs10 = load_fred("DGS10", force=force)
    etfs = load_etf_prices(force=force)

    combined = ff.join(bab, how="outer").join(dgs10.rename("DGS10"), how="outer")
    combined.to_parquet(PATHS.processed / "external_factors.parquet")
    return {"factors": combined, "etf_prices": etfs}
