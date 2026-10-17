# SPDX-License-Identifier: BSD-3-Clause
# Adapted from R package ider (https://cran.r-project.org/package=ider) by Hideitsu Hino,
# originally released under GPL-2; see LICENSES_bundled.txt
import warnings
import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from .._commonfuncs import _check_X, get_nn, GlobalEstimator
from sklearn.utils.validation import check_array, check_symmetric


class CorrInt(GlobalEstimator):
    """Intrinsic dimension estimation using the Correlation Dimension. [Grassberger1983]_ [IDHino]_

    Parameters
    ----------
    k1: int
        First neighborhood size considered
    k2: int
        Last neighborhood size considered
    DM: bool, default=False
        Is the input a precomputed distance matrix (dense)
    """

    def __init__(self, k1=10, k2=20, DM=False):
        self.k1 = k1
        self.k2 = k2
        self.DM = DM

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self : object
            Returns self.
        """
        X = _check_X(self, X, ensure_min_samples=2, ensure_min_features=2)

        self._k1, self._k2 = self.k1, self.k2
        if self._k2 >= len(X):
            warnings.warn("k2 larger or equal to len(X), using len(X)-1")
            self._k2 = len(X) - 1

        if self._k1 >= len(X) or self._k1 > self._k2:
            warnings.warn("k1 larger than k2 or len(X), using k2-1")
            self._k1 = self._k2 - 1

        self.dimension_ = self._corrint(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _corrint(self, X):

        n_elements = len(X) ** 2  # number of elements

        if self.DM is False:
            dists, _ = get_nn(X, self._k2)
            chunked_distmat = pairwise_distances_chunked(X)
        else:
            dists = np.sort(X, axis=1)[:, 1 : self._k2 + 1]
            chunked_distmat = X

        r1 = np.median(dists[:, self._k1 - 1])
        r2 = np.median(dists[:, self._k2 - 1])

        n_diagonal_entries = len(X)  # remove diagonal from sum count
        s1 = -n_diagonal_entries
        s2 = -n_diagonal_entries
        for chunk in chunked_distmat:
            s1 += (chunk < r1).sum()
            s2 += (chunk < r2).sum()

        Cr = np.array([s1 / n_elements, s2 / n_elements])
        estq = np.diff(np.log(Cr)) / np.log(r2 / r1)
        return estq[0]
