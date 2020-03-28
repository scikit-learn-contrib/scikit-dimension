#
# MIT License
#
# Copyright (c) 2020 Jonathan Bac
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import numpy as np
import warnings
from .._commonfuncs import get_nn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class MOM(BaseEstimator):
    """
    Intrinsic dimension estimation using the Method Of Moments algorithm.

    -----------
    Attributes

    k : int
        Number of nearest neighbors considered

    -----------
    Returns

    dimension_ : int
        Intrinsic dimension of the dataset

    -----------
    References
    
     L.  Amsaleg,  O.  Chelly,  T.  Furon,  S.  Girard,  M.  E.Houle,  K.  Kawarabayashi,  and  M.  Nett.    Extreme-value-theoretic  estimation  of  local  intrinsic  dimensio-nality.DAMI, 32(6):1768–1805, 2018.
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
        X = check_array(X, accept_sparse=False)
        if len(X) == 1:
            raise ValueError("Can't fit with 1 sample")
        if X.shape[1] == 1:
            raise ValueError("Can't fit with n_features = 1")
        if not np.isfinite(X).all():
            raise ValueError("X contains inf or NaN")
        if self.k >= len(X):
            warnings.warn('k >= len(X), using k = len(X)-1')

        dists, inds = get_nn(X, min(self.k, len(X)-1))

        self.dimension_ = self._idmom(dists)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _idmom(self, dists):
        # dists - nearest-neighbor distances (k x 1, or k x n), sorted
        w = dists[:, -1]
        m1 = np.sum(dists, axis=1)/dists.shape[1]
        ID = -m1 / (m1-w)
        return ID
