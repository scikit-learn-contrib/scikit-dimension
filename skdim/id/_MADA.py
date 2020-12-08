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
import warnings
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class MADA(BaseEstimator):

    """ Local intrinsic dimension estimation using the Manifold-Adaptive Dimension Estimation algorithm.
    A variant of fractal dimension called the local information  dimension i  considered. The local information  dimension is estimated by using the  probability mass function. Mada considers first order expansion of the probability mass around the inspection point, and it estimates the local information dimension by using two different radii from the inspection point. 
 
    Attributes
    ----------
    k : int, default=20
        Number of neighbors to consider
    comb : str, default="average"
        How to combine local estimates if local=False. Possible values : "average", "median"
    local : bool, default=False
        Whether to return local estimates
        
    Returns
    -------
    
    self.dimension_ : float
        The estimated intrinsic dimension
        
    References
    ----------
    Code translated and description taken from the ider R package by Hideitsu Hino.

    A. M. Farahmand, C. Szepesvari and J-Y. Audibert.  Manifold-adaptive dimension estimation.  International Conference on Machine Learning, 2007.
    """

    def __init__(self, k=20, comb="average", DM=False, local=False):
        self.k = k
        self.comb = comb
        self.DM = DM
        self.local = local

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
        X = check_array(X, accept_sparse=False)
        if len(X) == 1:
            raise ValueError("Can't fit with 1 sample")
        if X.shape[1] == 1:
            raise ValueError("Can't fit with n_features = 1")
        if not np.isfinite(X).all():
            raise ValueError("X contains inf or NaN")
        if self.k >= len(X):
            warnings.warn("k larger or equal to len(X), using len(X)-1")

        self._k = len(X) - 1 if self.k >= len(X) else self.k

        self.dimension_ = self._mada(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def fit_transform(self, X, y=None):
        if not self.is_fitted_:
            self.fit(X)
        return self.dimension_

    def _mada(self, X):

        if self.DM == False:
            distmat = squareform(pdist(X))

        else:
            distmat = X

        n = len(distmat)

        if self.local == False and n > 10000:
            ID = np.random.choice(n, size=int(np.round(n / 2)), replace=False)
            tmpD = distmat[ID, :]
            tmpD[tmpD == 0] = np.max(tmpD)

        else:
            tmpD = distmat
            tmpD[tmpD == 0] = np.max(tmpD)

        sortedD = np.sort(tmpD, axis=0, kind="mergesort")
        RK = sortedD[self._k - 1, :]
        RK2 = sortedD[int(np.floor(self._k / 2) - 1), :]
        ests = np.log(2) / np.log(RK / RK2)

        if self.local == True:
            return ests

        if self.comb == "average":
            return np.mean(ests)
        elif self.comb == "median":
            return np.median(ests)
        else:
            raise ValueError(
                "Invalid comb parameter. It has to be 'average' or 'median'"
            )
