#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.utils.validation import check_array
from .._commonfuncs import GlobalEstimator


class KNN(GlobalEstimator):
    # SPDX-License-Identifier: MIT, 2017 Kerstin Johnsson [IDJohnsson]_
    """Intrinsic dimension estimation using the kNN algorithm. [Carter2010]_ [IDJohnsson]_

    This is a simplified version of the kNN dimension estimation method described by Carter et al. (2010), 
    the difference being that block bootstrapping is not used.

    Parameters
    ----------
    X: 2D numeric array
        A 2D data set with each row describing a data point.
    k: int
        Number of distances to neighbors used at a time.
    ps: 1D numeric array
        Vector with sample sizes; each sample size has to be larger than k and smaller than nrow(data).
    M: int, default=1
        Number of bootstrap samples for each sample size.
    gamma: int, default=2
        Weighting constant.
    """

    def __init__(self, k=None, ps=None, M=1, gamma=2):
        self.k = k
        self.ps = ps
        self.M = M
        self.gamma = gamma

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self: object
            Returns self.
        self.dimension_: float
            The estimated intrinsic dimension
        self.residual_: float
            Residuals
        """

        self._k = 2 if self.k is None else self.k
        self._ps = np.arange(self._k + 1, self._k + 5) if self.ps is None else self.ps

        X = check_array(X, ensure_min_samples=self._k + 1, ensure_min_features=2)

        self.dimension_, self.residual_ = self._knnDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _knnDimEst(self, X):
        n = len(X)
        Q = len(self._ps)

        if min(self._ps) <= self._k or max(self._ps) > n:
            raise ValueError("ps must satisfy k<ps<len(X)")
        # Compute the distance between any two points in the X set
        dist = squareform(pdist(X))

        # Compute weighted graph length for each sample
        L = np.zeros((Q, self.M))

        for i in range(Q):
            for j in range(self.M):
                samp_ind = np.random.randint(0, n, self._ps[i])
                for l in samp_ind:
                    L[i, j] += np.sum(
                        np.sort(dist[l, samp_ind])[1 : (self._k + 1)] ** self.gamma
                    )
                    # Add the weighted sum of the distances to the k nearest neighbors.
                    # We should not include the sample itself, to which the distance is
                    # zero.

        # Least squares solution for m
        d = X.shape[1]
        epsilon = np.repeat(np.nan, d)
        for m0, m in enumerate(np.arange(1, d + 1)):
            alpha = (m - self.gamma) / m
            ps_alpha = self._ps ** alpha
            hat_c = np.sum(ps_alpha * np.sum(L, axis=1)) / (
                np.sum(ps_alpha ** 2) * self.M
            )
            epsilon[m0] = np.sum(
                (L - np.tile((hat_c * ps_alpha)[:, None], self.M)) ** 2
            )
            # matrix(vec, nrow = length(vec), ncol = b) is a matrix with b
            # identical columns equal to vec
            # sum(matr) is the sum of all elements in the matrix matr

        de = np.argmin(epsilon) + 1  # Missing values are discarded
        return de, epsilon[de - 1]
