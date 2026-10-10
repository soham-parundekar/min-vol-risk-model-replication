# Optimiser versus heuristic: replicating USMV and SPLV from free data

Two funds promise the same thing and go about it in completely different ways.

USMV tracks an MSCI index built by feeding a commercial Barra risk model into a
constrained optimiser. SPLV tracks an S&P index built by sorting the S&P 500 on
trailing volatility, keeping the quietest hundred names, and weighting them by
one over sigma. One is a quadratic program with sector bands, style bands, a
turnover cap and a minimum holding size. The other fits in a sentence.

This repository rebuilds both from public data and asks what the machinery buys.
It constructs a Barra-style factor risk model from SEC filings, Yahoo prices and
Ken French's industry definitions, uses it to run MSCI's published constraint
set, and compares the result against the one-line rule and against both live
funds.

The short answer is that the heuristic replicates far better than the optimiser,
and that this says more about the inputs than about the method. See
[`reports/report.md`](reports/report.md) for the full argument.

## What is actually in here

Seven modules, run in order by the scripts in `scripts/`:

| | What it does |
|---|---|
| A | Weekly style exposures on the live cross-section: size, shrunk beta, residual volatility, 12-1 momentum, book-to-price, liquidity |
| B | Weekly factor returns by cross-sectional WLS, with cap-weighted industry returns constrained to sum to zero |
| C | Four covariance estimators: structural factor model, sample, Ledoit-Wolf constant-correlation shrinkage, PCA with a Bai-Ng factor count |
| D | Bias statistics, out of sample, across factor-mimicking, random and minimum-variance portfolios |
| E | The USMV rule as a constrained QP, with a semi-continuous heuristic for the 5bp floor and MSCI's published relaxation ladder |
| F | The SPLV rule |
| G | Replication accuracy, an estimator horse race, risk decomposition, and alpha tests with and without betting-against-beta |

## The parts worth reading the code for

Most of the effort went into four places where a plausible-looking shortcut
produces results that are wrong in a way nothing complains about.

**Point-in-time fundamentals.** Every XBRL fact carries a period-end date and a
filing date, and only the second says when a number became public. Apple's
oldest `StockholdersEquity` fact describes the period ending 30 September 2006
and was filed on 27 October 2009. Key the panel off the period end and the 2006
cross-section receives a figure nobody could read for another three years;
book-to-price sharpens, the optimiser leans on it, and the backtest improves for
a reason that does not exist. `src/minvol/data/fundamentals.py` exposes no lookup
by fiscal period at all, and `tests/test_point_in_time_fundamentals.py` asserts
the property on both synthetic and live data.

**Ticker reuse.** Yahoo answers a request for `ABI` with a company that listed in
2025, although the S&P 500 member of that name left the index in 2008. Joined
naively, a 2025 IPO's returns get stitched onto a 2008 constituent. Thirty-three
tickers in this sample do that. The loader intersects every price series with the
window in which that ticker was actually an index member and treats a series
beginning after membership ended as a different company.

**Split-adjusted prices meeting unadjusted share counts.** Yahoo back-adjusts its
close series, so Apple's close on 28 June 2019 reads 49.48 rather than the 197.92
that printed that day. Share counts from a 2019 filing are in 2019 units.
Multiply them together and the market cap is four times too small, with the error
concentrated in exactly the mega caps that dominate a cap-weighted benchmark.
`src/minvol/data/splits.py` puts both on the same basis.

**The rank-deficient industry block.** Industry dummies sum to the intercept for
every stock, so something has to identify the system. Dropping a column works
numerically and silently redefines the market factor as the omitted industry's
return. On one week in this sample the two parametrisations differ by 0.54
percentage points on the market factor. The code imposes cap-weighted industry
returns summing to zero by reparametrising, which holds to 1.5e-18 and keeps the
intercept interpretable as the cap-weighted market return.

## Where the replication is honest about its limits

The parent universe is the S&P 500. MSCI optimises over the MSCI USA Investable
Market Index, which is roughly five times broader. That substitution is the
single largest known source of replication error and it cannot be fixed with free
data.

Industry groups come from SIC codes mapped to Fama-French 12 industries, because
GICS is licensed and the only free source of it covers companies that are in the
index today. Using that would reintroduce the survivorship bias the study is
trying to measure.

The price history has survivors in it and not much else. Of 899 tickers that
passed through the index between 2008 and 2026, 618 have usable history and 242
return nothing at all, at a steady rate of about thirteen a year. Rebuilding the
cap-weighted parent index from what remains and comparing it with SPY puts a
number on the damage: it tracks at 1.50% annualised error with 0.9962 correlation
from 2010, and overstates the index return by roughly half a point a year.

## Running it

```bash
pip install -r requirements.txt
make all          # data, factors, portfolios, bias tests, evaluation
make test
```

`make data` is the slow stage, fifteen to twenty minutes on a cold cache, most of
it spent pulling one company-facts document per registrant from EDGAR. Everything
is cached, so a second run costs almost nothing. Nothing requires an API key, a
subscription or a login.

Every parameter lives in [`config/params.yaml`](config/params.yaml), with a
comment on each one saying whether it is transcribed from a published methodology
or is our own modelling choice, and why.

## Layout

```
config/params.yaml      every bound, window and half-life, with its provenance
src/minvol/
  data/                 universe, prices, splits, fundamentals, industries, market cap
  factors/              Module A exposures, Module B factor returns
  risk/                 Module C covariance estimators, Module D bias tests
  optimize/             Module E USMV rules, Module F SPLV rules, rebalance calendars
  evaluate/             replication accuracy, risk attribution, alpha tests
  pipeline.py           shared loading and the one exposure matrix everything uses
scripts/                01 data, 02 factors, 03 portfolios, 04 bias, 05 evaluate, 06 figures
tests/                  look-ahead, constraint satisfaction, industry identification
references/             source register, bibliography, preserved filings
reports/                the written report, tables and figures
```

## Sources

Nineteen sources, each recorded in
[`references/source_register.csv`](references/source_register.csv) with its
identifier, access date, what depends on it, and whether it can be redistributed.
The two index methodologies and the journal articles are copyrighted and are not
stored here; the register says how to obtain each one. The two SEC filings are
public records and are kept under `references/sec_filings/`.

One parameter rests on a document that could not be opened. The S&P methodology
PDF returned HTTP 403, so the SPLV quarterly rebalance calendar and its
third-Friday effective date are taken from the published convention rather than
verified against the source. The selection and weighting rules were confirmed
independently from Invesco's own prospectus. That gap is recorded rather than
papered over.
