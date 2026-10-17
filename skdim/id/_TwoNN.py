# SPDX-License-Identifier: BSD-3-Clause AND MIT
# Adapted from https://github.com/fmottes/TWO-NN, Copyright (c) 2019 Francesco Mottes (fmottes),
# MIT License; see LICENSES_bundled.txt

import warnings

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.linear_model import LinearRegression
from .._commonfuncs import _check_X, get_nn, GlobalEstimator


class TwoNN(GlobalEstimator):
    """Intrinsic dimension estimation using the TwoNN algorithm. [Facco2019]_ [IDFacco]_ [IDMottes]_

    The ratios mu = r2/r1 of second to first nearest-neighbour distances follow a
    Pareto(1, d) law. ``method='linear'`` fits the line of Facco et al. 2017 through
    (log mu, -log(1 - F_emp(mu))). ``method='mle'`` uses the closed-form maximum
    likelihood estimate. log mu ~ Exponential(d), so keeping the k = int(N * (1 - discard_fraction))
    smallest of the N ratios is Type-II censoring, and the unbiased estimate is
    (k - 1) / (sum_{i<=k} log mu_(i) + (N - k) log mu_(k)).
    With ``discard_fraction=0`` this is (N - 1) / sum(log mu), the unbiased Pareto-shape MLE
    of [Denti2022]_; see DADApy
    (https://github.com/sissa-data-science/DADApy) for the reference implementation, whose
    'ml' option ignores the discarded fraction and so matches ``discard_fraction=0``.
    Points with r1 == 0 (duplicates) give undefined ratios. They are dropped with a warning.

    Parameters
    ----------  
    discard_fraction: float 
        Fraction (between 0 and 1) of largest ratios mu to discard (heuristic from the paper).
        With ``method='mle'`` the discarded ratios are treated as censored, not ignored.
    dist: bool
        If True, X is not a distance matrix but an (n_samples, 2) array whose columns hold
        each point's distances to its first and second nearest neighbours (r1, r2).
    method: str, 'linear' or 'mle'
        Linear fit of the empirical cumulative distribution (Facco et al. 2017) or
        closed-form maximum likelihood estimate.

    Attributes
    ----------
    x_: 1d array 
        np.array with the -log(mu) values.
    y_: 1d array 
        np.array with the -log(F(mu_{sigma(i)})) values.
    """

    def __init__(
        self, discard_fraction: float = 0.1, dist: bool = False, method: str = "linear"
    ):
        self.discard_fraction = discard_fraction
        self.dist = dist
        self.method = method

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
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
        if self.method not in ("linear", "mle"):
            raise ValueError("method must be 'linear' or 'mle'")
        X = _check_X(self, X, ensure_min_samples=2, ensure_min_features=2)

        self.dimension_, self.x_, self.y_ = self._twonn(X)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _twonn(self, X):
        """
        Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.

        -----------
        Parameters:

        X : 2d array-like
            2d data matrix. Samples on rows and features on columns.
        return_xy : bool (default=False)
            Whether to return also the coordinate vectors used for the linear fit.
        discard_fraction : float between 0 and 1
            Fraction of largest distances to discard (heuristic from the paper)
        dist : bool (default=False)
            Whether X holds precomputed (r1, r2) nearest-neighbour distances
        -----------
        Returns:

        d : float
            Intrinsic dimension of the dataset according to TWO-NN.
        x : 1d np.array (optional)
            Array with the -log(mu) values.
        y : 1d np.array (optional)
            Array with the -log(F(mu_{sigma(i)})) values.

        -----------
        References:

        E. Facco, M. d’Errico, A. Rodriguez & A. Laio
        Estimating the intrinsic dimension of datasets by a minimal neighborhood information (https://doi.org/10.1038/s41598-017-11873-y)
        """

        if self.dist:
            r1, r2 = X[:, 0], X[:, 1]
        # relatively high dimensional data, use distance matrix generator
        elif X.shape[1] > 25:
            r1, r2 = np.zeros(len(X)), np.zeros(len(X))
            i = 0
            for x in pairwise_distances_chunked(X):
                x = np.sort(x, axis=1)
                r1[i : i + len(x)], r2[i : i + len(x)] = x[:, 1], x[:, 2]
                i += len(x)
        else:  # relatively low dimensional data, search nearest neighbors directly
            dists, _ = get_nn(X, k=2)
            r1, r2 = dists[:, 0], dists[:, 1]

        # mu = r2/r1 for each data point; r1 == 0 (duplicate points) gives inf or nan
        with np.errstate(divide="ignore", invalid="ignore"):
            _mu = r2 / r1
        if not np.all(np.isfinite(_mu)):
            warnings.warn(
                f"Dropping {np.sum(~np.isfinite(_mu))} points with zero distance to "
                "their nearest neighbour (duplicate points)."
            )
            _mu = _mu[np.isfinite(_mu)]
        N = len(_mu)
        n_keep = int(N * (1 - self.discard_fraction))
        if n_keep < 2:
            raise ValueError("Fewer than 2 points left to estimate the dimension.")

        # discard the largest ratios
        mu = np.sort(_mu)[:n_keep]

        # Empirical cumulate
        Femp = np.arange(1, 1 + n_keep) / N

        if self.method == "mle":
            # unbiased MLE of the exponential rate of log(mu) under Type-II censoring
            log_mu = np.log(mu)
            d = (n_keep - 1) / (np.sum(log_mu) + (N - n_keep) * log_mu[-1])
        else:
            # Fit line
            lr = LinearRegression(fit_intercept=False)
            lr.fit(np.log(mu).reshape(-1, 1), -np.log(1 - Femp).reshape(-1, 1))
            d = lr.coef_[0][0]  # extract slope

        return (
            d,
            np.log(mu).reshape(-1, 1),
            -np.log(1 - Femp).reshape(-1, 1),
        )
