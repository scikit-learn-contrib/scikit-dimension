# SPDX-License-Identifier: BSD-3-Clause
import numbers
import warnings

import numpy as np
from scipy.optimize import brentq
from .._commonfuncs import _check_X, get_nn, GlobalEstimator


class GRIDE(GlobalEstimator):
    """Intrinsic dimension estimation with the Generalized Ratios ID Estimator (Gride). [Denti2022]_

    For each point, mu = r_{n2} / r_{n1} is the ratio of its distances to its n2-th and n1-th
    nearest neighbours. Under local homogeneity mu has density
    d (mu^d - 1)^(n2 - n1 - 1) / (B(n2 - n1, n1) mu^((n2 - 1) d + 1)), mu > 1 [Denti2022]_.
    The ID is the root of the score of this likelihood, which is concave in d, and its error
    is 1 / sqrt(observed Fisher information). With n1 = 1, n2 = 2 the density is Pareto(1, d)
    and the estimate is the unbiased TwoNN MLE (N - 1) / sum(log mu), as in ``TwoNN(method="mle")``.
    Raising n1 probes larger scales without decimating the data, which reduces the effect of
    short-scale noise.

    Reimplemented from [Denti2022]_; see DADApy (https://github.com/sissa-data-science/DADApy)
    for the reference implementation, whose conventions are followed: the likelihood uses
    (N - 1) log d in place of N log d (unbiased at n1 = 1), and ratios above
    q50 + 20 (q95 - q50) are discarded as likely overlapping points. Unlike DADApy, tied
    distances (mu = 1) are dropped rather than nudged.
    The tests follow the comparison approach of Jakub Malinowski's Gride port in the
    kmyim/scikit-dimension fork.

    Parameters
    ----------
    n1: int, default=1
        Order of the nearer neighbour.
    n2: int, optional
        Order of the farther neighbour, n2 > n1. Defaults to 2 * n1.
    multiscale: bool, default=False
        If True, also estimate the ID at scales n1 = 1, 2, 4, ..., with n2 = 2 * n1 and
        n2 <= min(range_max, n_samples - 1).
    range_max: int, default=64
        Largest neighbour order used when ``multiscale=True``.

    Attributes
    ----------
    dimension_: float
        ID estimate at (n1, n2).
    error_: float
        Standard error of ``dimension_`` from the observed Fisher information.
    scales_: 1d array
        n1 values of the multiscale estimates (only if ``multiscale=True``); n2 = 2 * n1.
    dimension_scales_, error_scales_: 1d arrays
        ID estimates and their standard errors at each scale (only if ``multiscale=True``).
    radius_scales_: 1d array
        Mean distance to the n2-th neighbour at each scale (only if ``multiscale=True``).
    """

    def __init__(self, n1=1, n2=None, multiscale=False, range_max=64):
        self.n1 = n1
        self.n2 = n2
        self.multiscale = multiscale
        self.range_max = range_max

    def fit(self, X, y=None):
        """Fit Gride on a data set.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A data set for which the intrinsic dimension is estimated.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self : object
            Returns self.
        """
        X = _check_X(self, X, ensure_min_samples=2)
        n2 = 2 * self.n1 if self.n2 is None else self.n2
        if not (_is_int(self.n1) and _is_int(n2) and 1 <= self.n1 < n2):
            raise ValueError("n1 and n2 must be integers with 1 <= n1 < n2")
        if n2 > len(X) - 1:
            raise ValueError(f"n2={n2} needs at least n2 + 1 = {n2 + 1} samples")
        kmax = n2
        if self.multiscale:
            if not (_is_int(self.range_max) and self.range_max >= 2):
                raise ValueError("range_max must be an integer >= 2")
            range_max = min(self.range_max, len(X) - 1)
            kmax = max(kmax, range_max)
        dists, _ = get_nn(X, k=kmax)  # dists[:, k - 1] = distance to k-th neighbour

        self.dimension_, self.error_ = _gride(dists, self.n1, n2)
        if self.multiscale:
            self.scales_ = 2 ** np.arange(int(np.log2(range_max)))
            res = np.array([_gride(dists, n, 2 * n) for n in self.scales_])
            self.dimension_scales_, self.error_scales_ = res[:, 0], res[:, 1]
            self.radius_scales_ = dists[:, 2 * self.scales_ - 1].mean(axis=0)

        self.is_fitted_ = True
        return self


def _is_int(n):
    return isinstance(n, numbers.Integral) and not isinstance(n, bool)


def _gride(dists, n1, n2):
    """Gride MLE and its standard error from the (n1, n2) neighbour distances."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mu = np.log(dists[:, n2 - 1] / dists[:, n1 - 1])
    # r_n1 == 0 (duplicates) gives inf/nan; r_n1 == r_n2 (ties) gives 0, where the score is 0/0
    ok = np.isfinite(log_mu) & (log_mu > 0)
    if not np.all(ok):
        warnings.warn(
            f"Gride (n1={n1}, n2={n2}): dropping {np.sum(~ok)} points with zero or tied "
            "neighbour distances (duplicate points)."
        )
        log_mu = log_mu[ok]
    # DADApy's outlier cut on mu (monotone in log mu), for overlapping or fat-tailed data
    q95, q50 = np.percentile(np.exp(log_mu), [95, 50])
    log_mu = log_mu[np.exp(log_mu) < 20 * (q95 - q50) + q50]
    N = len(log_mu)
    if N < 2:
        raise ValueError("Fewer than 2 points left to estimate the dimension.")

    # d log L / dd; mu^d / (mu^d - 1) = 1 / (1 - exp(-d log mu))
    def score(d):
        return (
            (N - 1) / d
            + (n2 - n1 - 1) * np.sum(log_mu / -np.expm1(-d * log_mu))
            - (n2 - 1) * np.sum(log_mu)
        )

    # score -> +inf as d -> 0 and -> -n1 * sum(log mu) < 0 as d -> inf: one root
    hi = 1.0
    while score(hi) > 0:
        hi *= 2
    d = brentq(score, 1e-8, hi, xtol=1e-12, rtol=1e-12)
    # observed information -d2 log L / dd2; mu^d / (mu^d - 1)^2 = 1 / (4 sinh^2(d log mu / 2))
    info = N / d**2 + (n2 - n1 - 1) * np.sum(
        (log_mu / (2 * np.sinh(d * log_mu / 2))) ** 2
    )
    return d, 1 / np.sqrt(info)
