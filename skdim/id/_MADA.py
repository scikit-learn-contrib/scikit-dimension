# SPDX-License-Identifier: BSD-3-Clause
# Adapted from R package ider (https://cran.r-project.org/package=ider) by Hideitsu Hino,
# originally released under GPL-2; see LICENSES_bundled.txt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from .._commonfuncs import LocalEstimator


class MADA(LocalEstimator):
    """Intrinsic dimension estimation using the Manifold-Adaptive Dimension Estimation algorithm. [Farahmand2007]_, [IDHino]_

    MADA uses a variant of fractal dimension called the local information dimension.
    MADA considers the first order expansion of the probability mass around the inspection point,
    and it estimates the local information dimension by using two different radii from the inspection point. 

    Parameters
    ----------
    DM: bool
        Whether input is a precomputed distance matrix
    """

    _N_NEIGHBORS = 20

    def __init__(self, DM=False):
        self.DM = DM

    def _fit(self, **kwargs):
        self.dimension_pw_ = self._mada(kwargs["X"])

    def _mada(self, X):

        if self.DM is False:
            distmat = squareform(pdist(X))
        else:
            distmat = X

        distmat[distmat == 0] = np.max(distmat)

        sortedD = np.sort(distmat, axis=0, kind="mergesort")
        RK = sortedD[self.n_neighbors_ - 1, :]
        RK2 = sortedD[int(np.floor(self.n_neighbors_ / 2) - 1), :]
        ests = np.log(2) / np.log(RK / RK2)

        return ests
