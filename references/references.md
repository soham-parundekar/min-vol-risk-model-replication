# Sources

Everything the study leans on, what it was used for, and where to get it.
`source_register.csv` holds the same list in machine-readable form with
redistribution status and retrieval notes per source; `references.bib` holds the
academic citations.

Two conventions worth stating up front. Metadata here was checked against the
source rather than recalled, and where a detail could not be verified it is left
out rather than guessed. And the actual document is preserved in this repository
only where redistribution is clearly permitted, which in practice means SEC
filings and one short definitional listing. Everything else carries retrieval
instructions instead.

## The two index rules

**REF-001 — MSCI Minimum Volatility Indices Methodology, January 2012.**
The primary source for the whole of Module E. Every bound in
`config/params.yaml` under `usmv` is transcribed from it: the per-security cap
at the lower of 1.5% or twenty times the parent weight, the 0.05% minimum
holding, the plus or minus 5% sector band, the plus or minus 0.25 standard
deviation style band with the volatility factor exempted, and the 10% one-way
turnover cap. Section 4.1 supplies the semi-annual review on the last business
day of May and November, and also the detail that the covariance matrix used is
the one dated to the end of the preceding month, which is why
`covariance_lag_months` exists. Appendix III supplies the infeasibility ladder,
which turned out to differ from what an intuitive reading would produce: MSCI
relaxes turnover in five-point steps to 30%, then walks the minimum holding down
to 0.01%, then loosens the cap and sector band together, and it never relaxes
the style band at all. When the sequence is exhausted the index is not
rebalanced that review, which the implementation reproduces.

The document is copyrighted and is not included here.

**REF-002 — MSCI Minimum Volatility Advanced Indexes Methodology, February
2025.** Consulted to check whether the constraint set had moved in thirteen
years. It has not: the cap, floor, bands and turnover limit are identical. One
sentence does differ in a way that matters. The 2012 text exempts "the Barra
Volatility risk index factor", singular; the 2025 text exempts "the Barra
Volatility risk factors (BETA and RESVOL)", two of them, because the USE4 model
split what USE3 had combined. The same words therefore describe two different
constraint sets depending on the risk model behind them, so the study runs both
readings and reports which one behaves like the live fund. This document governs
a different index family and is cited only for that comparison.

**REF-004 — Invesco S&P 500 Low Volatility ETF, Form 497K, 20 December 2024.**
The SPLV rule in the issuer's own words, and a public filing, so a copy is kept
under `sec_filings/`. It confirms the selection rule (about 100 S&P 500
securities with the least realised volatility, measured from price return data
over the trailing one year of trading days) and the weighting rule (inverse of
each security's volatility, so the quietest names get the largest weights). It
also reports the fund's own portfolio turnover of 68% for the most recent fiscal
year, which is a useful external check on the replica's turnover.

**REF-005 — S&P Dow Jones Indices, S&P 500 Low Volatility Index methodology.**
One parameter in the study rests on a document that could not be opened: the
quarterly rebalance calendar and the third-Friday effective date. The URL
returned HTTP 403 from the environment the study was run in, so no copy was
obtained and the selection and weighting rules were verified from REF-004
instead. The rebalance timing is flagged in the limitations section of the report
rather than presented as verified.

**REF-003 — iShares MSCI USA Minimum Volatility Index Fund, Form 497K,
7 October 2011.** Filed eleven days before USMV began trading. Used to fix the
fund's inception date and confirm which index it tracks. Public filing, kept
under `sec_filings/`.

## Methods

**REF-006 — Ledoit and Wolf (2004).** The constant-correlation shrinkage target
and the analytic shrinkage intensity. Implemented directly in
`src/minvol/risk/covariance.py` rather than taken from a library, because the
widely available library implementation shrinks toward scaled identity, which
comes from the authors' other 2004 paper and is a different estimator. The
implementation is checked against a simulation where the constant-correlation
target is exactly true and the intensity should go to one.

**REF-007 — Bai and Ng (2002).** The information criteria that set the number of
principal components in the PCA estimator. Without them the factor count is a
free parameter that can be tuned until the results look good. Validated by
recovering known factor counts from simulated panels.

**REF-008 — Frazzini and Pedersen (2014).** The leverage-constraint account of
the low-volatility effect, and the factor used to test it. The citation detail
is as AQR states it in the dataset file itself.

**REF-009 — Fama and French (2015).** Defines the five factors used as the base
attribution model.

**REF-010 — Markowitz (1952).** Cited here because MSCI cites it: the minimum
variance portfolio the index optimises toward is his. Only the journal name and
year are asserted, because that is all the citing footnote gives and the
original was not checked.

### A note on the beta adjustment

The raw 252-day regression beta is shrunk toward one with coefficients 0.67 and
0.33, the conventional two-thirds / one-third adjustment. This is widely
attributed to Marshall Blume's work in the 1970s on the tendency of estimated
betas to regress toward the mean. The specific coefficients used here come from
the project specification, and no page-level citation is given because the
attribution could not be verified against the original article from this
environment. The adjustment is documented as a modelling choice in
`config/params.yaml` and its effect is a parameter a reader can change.

## Data

Seven datasets, all free, none re-hosted here except the SIC definitions:

- **REF-011** Fama/French Data Library, daily five factors and momentum. The
  vintage used was built from the 202608 CRSP database and ends 2026-08-31,
  which is what limits the attribution window.
- **REF-012** AQR betting-against-beta daily factors. Ends 2026-07-31, the
  binding constraint on any regression that includes BAB. AQR rebuilds the full
  history on each update, so two runs against different vintages can differ
  legitimately.
- **REF-013** SEC EDGAR XBRL company-facts, for point-in-time book equity and
  share counts. The study uses company-facts rather than company-concept because
  the per-concept endpoint returns an empty result for some filers while the
  company-facts document for the same registrant has the data.
- **REF-014** SEC EDGAR submissions, for SIC codes.
- **REF-015** Ken French's `Siccodes12`, the SIC-range-to-industry mapping. A
  copy is preserved under `data_documentation/` because the industry assignment
  cannot be reproduced without knowing which ranges were in force.
- **REF-016** FRED DGS10. Downloaded by the acquisition script but not used by
  any reported result; listed for completeness rather than left as an
  undocumented fetch.
- **REF-017** Yahoo Finance prices and splits.
- **REF-018** and **REF-019** The two membership sources, cross-checked against
  each other. The second can only validate the first's endpoint, not its
  history, and the reconciliation output says so.

## What is deliberately absent

No copyrighted methodology document or journal article is stored in this
repository. For each one the register records the citation, the identifier, the
URL, the date accessed, and how another researcher can obtain it legitimately.
Nothing here was obtained by working around a paywall, a login or an access
control; the one document that refused the request (REF-005) is recorded as
not retrieved, and the parameter that depended on it is flagged in the report.
