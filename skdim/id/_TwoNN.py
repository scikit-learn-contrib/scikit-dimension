# SPDX-License-Identifier: BSD-3-Clause AND MIT
# Adapted from https://github.com/fmottes/TWO-NN, Copyright (c) 2019 Francesco Mottes (fmottes),
# MIT License; see LICENSES_bundled.txt

from sklearn.utils.validation import check_array

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.linear_model import LinearRegression
from .._commonfuncs import _check_X, get_nn, GlobalEstimator


class TwoNN(GlobalEstimator):
    """Intrinsic dimension estimation using the TwoNN algorithm. [Facco2019]_ [IDFacco]_ [IDMottes]_

    Parameters
    ----------  
    discard_fraction: float 
        Fraction (between 0 and 1) of largest distances to discard (heuristic from the paper)
    dist: bool
        Whether data is a precomputed distance matrix

    Attributes
    ----------
    x_: 1d array 
        np.array with the -log(mu) values.
    y_: 1d array 
        np.array with the -log(F(mu_{sigma(i)})) values.
    """

    def __init__(self, discard_fraction: float = 0.1, dist: bool = False):
        self.discard_fraction = discard_fraction
        self.dist = dist

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
            Whether data is a precomputed distance matrix
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

        N = len(X)

        if self.dist:
            r1, r2 = X[:, 0], X[:, 1]
            _mu = r2 / r1
            # discard the largest distances
            mu = _mu[np.argsort(_mu)[: int(N * (1 - self.discard_fraction))]]

        else:
            # mu = r2/r1 for each data point
            # relatively high dimensional data, use distance matrix generator
            if X.shape[1] > 25:
                distmat_chunks = pairwise_distances_chunked(X)
                _mu = np.zeros((len(X)))
                i = 0
                for x in distmat_chunks:
                    x = np.sort(x, axis=1)
                    r1, r2 = x[:, 1], x[:, 2]
                    _mu[i : i + len(x)] = r2 / r1
                    i += len(x)

                # discard the largest distances
                mu = _mu[np.argsort(_mu)[: int(N * (1 - self.discard_fraction))]]

            else:  # relatively low dimensional data, search nearest neighbors directly
                dists, _ = get_nn(X, k=2)
                r1, r2 = dists[:, 0], dists[:, 1]
                _mu = r2 / r1
                # discard the largest distances
                mu = _mu[np.argsort(_mu)[: int(N * (1 - self.discard_fraction))]]

        # Empirical cumulate
        Femp = np.arange(1, 1 + int(N * (1 - self.discard_fraction))) / N

        # Fit line
        lr = LinearRegression(fit_intercept=False)
        lr.fit(np.log(mu).reshape(-1, 1), -np.log(1 - Femp).reshape(-1, 1))

        d = lr.coef_[0][0]  # extract slope

        return (
            d,
            np.log(mu).reshape(-1, 1),
            -np.log(1 - Femp).reshape(-1, 1),
        )
