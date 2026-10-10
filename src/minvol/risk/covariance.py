"""Module C: four covariance estimators, built to be compared.

All four produce a covariance of *weekly* stock returns for a given set of names
on a given date, using only data up to and including that date. Keeping the
frequency and the window common across estimators is the point: the horse race
in Module G is only meaningful if the four differ in method rather than in how
much data they were handed.

1. **Structural factor model.** Sigma = X F X' + Delta. The factor covariance F
   is an exponentially weighted estimate of the Module B factor returns, with a
   shorter half-life for volatilities than for correlations. Delta is diagonal
   specific variance, each name shrunk toward the mean of its market-cap decile.

2. **Sample covariance.** The naive baseline. With roughly 500 names and 156
   weekly observations it is singular, which is the honest result rather than a
   configuration mistake. A minimum-variance optimiser handed a singular matrix
   will find directions it believes carry zero risk and pile into them.

3. **Ledoit-Wolf shrinkage** toward constant correlation, with the shrinkage
   intensity computed from the data as in Ledoit and Wolf (2004).

4. **PCA statistical factors**, with the number of components chosen by the
   Bai-Ng information criterion rather than a round number.

The two-half-life construction in (1) deserves a word. Volatilities move with
regime and need to react; a long window is still reporting calm when the market
has stopped being calm. Correlations are more stable and more noisily
estimated, so they benefit from a longer window. Estimating one matrix with a
single half-life forces one compromise on both. Instead the correlation matrix
comes from a long half-life, the volatilities from a short one, and they are
recombined as Sigma = D C D.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import PARAMS


def ewma_weights(n_obs: int, halflife: float) -> np.ndarray:
    """Normalised exponential weights, oldest first, for ``n_obs`` observations."""
    lam = 0.5 ** (1.0 / float(halflife))
    ages = np.arange(n_obs - 1, -1, -1, dtype="float64")
    w = lam ** ages
    return w / w.sum()


def ewma_cov(data: np.ndarray, halflife: float) -> np.ndarray:
    """Exponentially weighted covariance of a (T, K) array, oldest row first."""
    w = ewma_weights(data.shape[0], halflife)
    mu = w @ data
    centred = data - mu
    return (centred * w[:, None]).T @ centred


def two_halflife_cov(data: np.ndarray, vol_halflife: float,
                     corr_halflife: float) -> np.ndarray:
    """Covariance with volatilities and correlations estimated on different memories.

    The correlation matrix comes from the long half-life, the standard
    deviations from the short one, and they are recombined. Correlations are
    taken from a proper covariance so the result stays positive semi-definite.
    """
    long_cov = ewma_cov(data, corr_halflife)
    long_sd = np.sqrt(np.clip(np.diag(long_cov), 1e-18, None))
    corr = long_cov / np.outer(long_sd, long_sd)
    np.fill_diagonal(corr, 1.0)

    short_cov = ewma_cov(data, vol_halflife)
    short_sd = np.sqrt(np.clip(np.diag(short_cov), 1e-18, None))
    return corr * np.outer(short_sd, short_sd)


def shrink_specific_variance(spec_var: pd.Series, market_cap: pd.Series,
                             n_deciles: int, intensity: float) -> pd.Series:
    """Pull each specific variance toward the mean of its market-cap decile.

    A single stock's idiosyncratic variance estimated from a few years of weekly
    residuals is noisy, and the noise is dangerous in one direction in
    particular: a minimum-variance optimiser seeks out names whose specific risk
    looks lowest, which is to say the names whose estimates are most
    understated. Shrinking toward a peer group built from size, which is the
    strongest observable predictor of idiosyncratic volatility, blunts that.
    """
    valid = spec_var.dropna()
    if valid.empty:
        return spec_var
    caps = market_cap.reindex(valid.index)
    if caps.notna().sum() < n_deciles:
        target = pd.Series(valid.mean(), index=valid.index)
    else:
        try:
            buckets = pd.qcut(caps.rank(method="first"), n_deciles, labels=False)
        except ValueError:
            buckets = pd.Series(0, index=valid.index)
        bucket_mean = valid.groupby(buckets).transform("mean")
        target = bucket_mean.fillna(valid.mean())
    shrunk = (1.0 - intensity) * valid + intensity * target
    return shrunk.reindex(spec_var.index)


@dataclass
class FactorRiskModel:
    """A fitted structural model, kept in parts so risk can be decomposed later."""

    exposures: pd.DataFrame      # names x factors
    factor_cov: pd.DataFrame     # factors x factors
    specific_var: pd.Series      # names

    def covariance(self) -> pd.DataFrame:
        x = self.exposures.to_numpy(dtype="float64")
        f = self.factor_cov.to_numpy(dtype="float64")
        sigma = x @ f @ x.T
        sigma[np.diag_indices_from(sigma)] += self.specific_var.to_numpy(dtype="float64")
        return pd.DataFrame(sigma, index=self.exposures.index,
                            columns=self.exposures.index)


def structural_covariance(exposures: pd.DataFrame, factor_returns: pd.DataFrame,
                          specific_returns: pd.DataFrame, market_cap: pd.Series,
                          asof: pd.Timestamp) -> FactorRiskModel | None:
    """Fit Sigma = X F X' + Delta using only history up to ``asof``."""
    cfg = PARAMS["covariance"]
    hist = factor_returns.loc[factor_returns.index <= asof]
    if len(hist) < PARAMS["sample"]["min_weeks_for_covariance"]:
        return None
    hist = hist.tail(cfg["ewma_window_weeks"])

    factors = [c for c in exposures.columns if c in hist.columns]
    hist = hist.loc[:, factors].dropna(axis=1, how="all")
    factors = list(hist.columns)
    if not factors:
        return None
    hist = hist.fillna(0.0)

    f_cov = two_halflife_cov(
        hist.to_numpy(dtype="float64"),
        cfg["factor_vol_halflife_weeks"],
        cfg["factor_corr_halflife_weeks"],
    )
    factor_cov = pd.DataFrame(f_cov, index=factors, columns=factors)

    spec_hist = specific_returns.loc[specific_returns.index <= asof]
    spec_hist = spec_hist.tail(cfg["ewma_window_weeks"])
    names = [n for n in exposures.index if n in spec_hist.columns]
    if len(names) < 50:
        return None
    spec_hist = spec_hist.loc[:, names]

    w = ewma_weights(len(spec_hist), cfg["specific_vol_halflife_weeks"])
    vals = spec_hist.to_numpy(dtype="float64")
    ok = np.isfinite(vals)
    vals = np.where(ok, vals, 0.0)
    wmat = np.broadcast_to(w[:, None], vals.shape) * ok
    denom = wmat.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (wmat * vals).sum(axis=0) / denom
        var = (wmat * (vals - mean) ** 2).sum(axis=0) / denom
    spec_var = pd.Series(var, index=names)
    # Too few residuals to say anything; fall back to the cross-sectional median.
    spec_var[ok.sum(axis=0) < 20] = np.nan
    spec_var = spec_var.fillna(spec_var.median())

    spec_var = shrink_specific_variance(
        spec_var, market_cap, cfg["specific_cap_deciles"], cfg["specific_shrink_intensity"]
    )
    spec_var = spec_var.clip(lower=1e-10)

    x = exposures.loc[names, factors].fillna(0.0)
    return FactorRiskModel(exposures=x, factor_cov=factor_cov, specific_var=spec_var)


def sample_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Plain sample covariance. Singular whenever there are more names than weeks."""
    return returns.cov()


def ledoit_wolf_constant_correlation(returns: pd.DataFrame,
                                     intensity: float | None = None
                                     ) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf (2004) shrinkage toward a constant-correlation target.

    The target keeps each stock's own sample variance and replaces every
    pairwise correlation with the average correlation across the cross-section.
    That is a better target than scaled identity for equities, which really do
    share a common factor, and it is the version from "Honey, I Shrunk the
    Sample Covariance Matrix" rather than the identity-target paper that most
    library implementations ship.

    The optimal intensity is (pi - rho) / gamma / T, clipped to [0, 1]:
    ``pi`` is the summed variance of the sample covariance entries, ``rho`` the
    summed covariance between the target and the sample estimate, and ``gamma``
    the squared distance between target and sample.
    """
    x = returns.to_numpy(dtype="float64")
    t, n = x.shape
    if t < 3 or n < 2:
        return returns.cov(), 1.0

    mean = x.mean(axis=0)
    xc = x - mean
    s = (xc.T @ xc) / t

    var = np.diag(s)
    sd = np.sqrt(np.clip(var, 1e-20, None))
    outer_sd = np.outer(sd, sd)
    corr = s / outer_sd
    off = ~np.eye(n, dtype=bool)
    rbar = corr[off].mean()

    target = rbar * outer_sd
    np.fill_diagonal(target, var)

    if intensity is None:
        # pi: variance of each entry of the sample covariance.
        y = xc ** 2
        pi_mat = (y.T @ y) / t - s ** 2
        pi = pi_mat.sum()

        # rho: the diagonal part, plus the cross terms the common correlation in
        # the target induces. theta[i, j] below is
        # (1/T) sum_t (xc_ti^2 - s_ii)(xc_ti xc_tj - s_ij), which expands to the
        # third-moment product minus var_i * s_ij; theta.T supplies the mirrored
        # term in j.
        theta = ((xc ** 3).T @ xc) / t - var[:, None] * s
        rho_diag = np.diag(pi_mat).sum()
        sd_ratio = np.outer(1.0 / sd, sd)
        rho_off = (rbar / 2.0) * (sd_ratio * theta + sd_ratio.T * theta.T)
        rho = rho_diag + rho_off[off].sum()

        gamma = ((target - s) ** 2).sum()
        kappa = (pi - rho) / gamma if gamma > 0 else 0.0
        intensity = float(np.clip(kappa / t, 0.0, 1.0))

    shrunk = intensity * target + (1.0 - intensity) * s
    return pd.DataFrame(shrunk, index=returns.columns, columns=returns.columns), intensity


def bai_ng_factor_count(returns: pd.DataFrame, kmax: int = 20,
                        criterion: str = "ICp2") -> int:
    """Number of principal components by the Bai and Ng (2002) criterion.

    Picking k by eye, or by "enough to explain 80% of variance", makes the
    factor count a free parameter that can be tuned until the results look
    good. Bai-Ng penalises additional factors by a term that depends on both
    panel dimensions and has a defensible consistency argument behind it.
    """
    x = returns.to_numpy(dtype="float64")
    t, n = x.shape
    x = x - x.mean(axis=0)
    sd = x.std(axis=0, ddof=0)
    sd[sd <= 0] = 1.0
    x = x / sd

    kmax = int(min(kmax, n - 1, t - 1))
    if kmax < 1:
        return 1

    # Eigen-decompose the smaller Gram matrix.
    if n <= t:
        gram = (x.T @ x) / t
    else:
        gram = (x @ x.T) / n
    eigvals = np.linalg.eigvalsh(gram)[::-1]
    eigvals = np.clip(eigvals, 0.0, None)
    total = (x ** 2).sum() / (n * t)

    scale = eigvals.sum()
    if scale <= 0:
        return 1

    best_k, best_ic = 1, np.inf
    cnt = min(n, t)
    for k in range(1, kmax + 1):
        explained = eigvals[:k].sum() / scale
        v_k = max(total * (1.0 - explained), 1e-12)
        if criterion == "ICp1":
            penalty = k * ((n + t) / (n * t)) * np.log((n * t) / (n + t))
        else:
            penalty = k * ((n + t) / (n * t)) * np.log(cnt)
        ic = np.log(v_k) + penalty
        if ic < best_ic:
            best_ic, best_k = ic, k
    return best_k


def pca_covariance(returns: pd.DataFrame, kmax: int = 20,
                   criterion: str = "ICp2") -> tuple[pd.DataFrame, int]:
    """Statistical factor covariance: k principal components plus diagonal residual."""
    k = bai_ng_factor_count(returns, kmax=kmax, criterion=criterion)
    x = returns.to_numpy(dtype="float64")
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = (xc.T @ xc) / max(t - 1, 1)

    vals, vecs = np.linalg.eigh(s)
    order = np.argsort(vals)[::-1]
    vals, vecs = np.clip(vals[order], 0.0, None), vecs[:, order]

    load = vecs[:, :k] * np.sqrt(vals[:k])
    common = load @ load.T
    resid = np.clip(np.diag(s) - np.diag(common), 1e-12, None)
    sigma = common + np.diag(resid)
    return pd.DataFrame(sigma, index=returns.columns, columns=returns.columns), k


def nearest_psd(matrix: pd.DataFrame, floor: float = 0.0) -> pd.DataFrame:
    """Clip negative eigenvalues so a matrix can be handed to a QP solver.

    Only ever needed for the sample covariance, where tiny negative eigenvalues
    appear from floating-point error on an already rank-deficient matrix. The
    structural, shrunk and PCA estimators are positive definite by construction.
    """
    a = matrix.to_numpy(dtype="float64")
    a = (a + a.T) / 2.0
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, floor, None)
    fixed = (vecs * vals) @ vecs.T
    fixed = (fixed + fixed.T) / 2.0
    return pd.DataFrame(fixed, index=matrix.index, columns=matrix.columns)


def estimate_all(asof: pd.Timestamp, names: list[str], weekly_returns: pd.DataFrame,
                 exposures: pd.DataFrame, factor_returns: pd.DataFrame,
                 specific_returns: pd.DataFrame, market_cap: pd.Series
                 ) -> dict[str, object]:
    """Build all four estimates for one rebalance date on one set of names.

    Every estimator sees the same window and the same names, so differences in
    the Module G results come from method and nothing else.
    """
    cfg = PARAMS["covariance"]
    hist = weekly_returns.loc[weekly_returns.index <= asof, :]
    hist = hist.tail(cfg["sample_window_weeks"])
    hist = hist.loc[:, [n for n in names if n in hist.columns]]
    hist = hist.dropna(axis=1, thresh=int(0.8 * len(hist)))
    hist = hist.fillna(0.0)
    usable = list(hist.columns)

    out: dict[str, object] = {"names": usable, "n_obs": len(hist)}
    # Every estimator gets the same minimum history. The three non-structural
    # ones would happily run on far less, but letting them start earlier would
    # mean the horse race in Module G compared four estimators over four
    # different samples, and whichever began in a calmer year would look best
    # for a reason that has nothing to do with the estimator.
    min_weeks = PARAMS["sample"]["min_weeks_for_covariance"]
    enough_factor_history = (
        len(factor_returns.loc[factor_returns.index <= asof]) >= min_weeks
    )
    if len(usable) < 50 or len(hist) < min_weeks or not enough_factor_history:
        return out

    model = structural_covariance(
        exposures.reindex(usable), factor_returns, specific_returns,
        market_cap.reindex(usable), asof,
    )
    if model is not None:
        out["structural"] = model.covariance()
        out["structural_model"] = model

    out["sample"] = sample_covariance(hist)
    lw, intensity = ledoit_wolf_constant_correlation(
        hist, intensity=cfg["ledoit_wolf_intensity"]
    )
    out["ledoit_wolf"] = lw
    out["ledoit_wolf_intensity"] = intensity
    pca, k = pca_covariance(hist, kmax=cfg["pca_kmax"], criterion=cfg["pca_criterion"])
    out["pca"] = pca
    out["pca_k"] = k
    return out
