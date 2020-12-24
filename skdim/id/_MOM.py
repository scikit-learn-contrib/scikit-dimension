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
import warnings
from .._commonfuncs import get_nn, GlobalEstimator, PointwiseEstimator
from sklearn.utils.validation import check_array


class MOM(GlobalEstimator, PointwiseEstimator):
    """
    Intrinsic dimension estimation using the Method Of Moments algorithm.

    Attributes
    ----------
    k : int
        Number of nearest neighbors considered
    
    References
    ----------
    Code translated from the original implementation by Miloš Radovanović (https://perun.pmf.uns.ac.rs/radovanovic/tle/).

     L.  Amsaleg,  O.  Chelly,  T.  Furon,  S.  Girard,  M.  E.Houle,  K.  Kawarabayashi,  and  M.  Nett. Extreme-value-theoretic estimation of local intrinsic dimensionality.DAMI, 32(6):1768–1805, 2018.
    """

    def __init__(self, k=20):
        self.k = k

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
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

        if self.k >= len(X):
            warnings.warn("k >= len(X), using k = len(X)-1")

        dists, inds = get_nn(X, min(self.k, len(X) - 1))

        self.dimension_ = self._idmom(dists)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _idmom(self, dists):
        # dists - nearest-neighbor distances (k x 1, or k x n), sorted
        w = dists[:, -1]
        m1 = np.sum(dists, axis=1) / dists.shape[1]
        ID = -m1 / (m1 - w)
        return ID
