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
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class KNN(BaseEstimator):
    """ This is a somewhat simplified version of the kNN dimension estimation method described by Carter et al. (2010), the difference being that block bootstrapping is not used.

    Attributes
    ----------

    X : 2D numeric array
        A 2D data set with each row describing a data point.
    k : int
        Number of distances to neighbors used at a time.
    ps : 1D numeric array
        Vector with sample sizes; each sample size has to be larger than k and smaller than nrow(data).
    M : int, default=1
        Number of bootstrap samples for each sample size.
    gamma : int, default=2
        Weighting constant.

    References
    ----------

    Carter, K.M., Raich, R. and Hero, A.O. (2010) On local intrinsic dimension estimation and its applications. IEEE Trans. on Sig. Proc., 58(2), 650-663. 
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

        self._k = 2 if self.k is None else self.k
        self._ps = np.arange(self._k+1, self._k +
                             5) if self.ps is None else self.ps

        self.dimension_, self.residual_ = self._knnDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _knnDimEst(self, X):
        n = len(X)
        Q = len(self._ps)

        if min(self._ps) <= self._k or max(self._ps) > n:
            raise ValueError('ps must satisfy k<ps<len(X)')
        # Compute the distance between any two points in the X set
        dist = squareform(pdist(X))

        # Compute weighted graph length for each sample
        L = np.zeros((Q, self.M))

        for i in range(Q):
            for j in range(self.M):
                samp_ind = np.random.randint(0, n, self._ps[i])
                for l in samp_ind:
                    L[i, j] += np.sum(np.sort(dist[l, samp_ind])
                                      [1:(self._k + 1)]**self.gamma)
                    # Add the weighted sum of the distances to the k nearest neighbors.
                    # We should not include the sample itself, to which the distance is
                    # zero.

        # Least squares solution for m
        d = X.shape[1]
        epsilon = np.repeat(np.nan, d)
        for m0, m in enumerate(np.arange(1, d+1)):
            alpha = (m - self.gamma)/m
            ps_alpha = self._ps**alpha
            hat_c = np.sum(ps_alpha * np.sum(L, axis=1)) / \
                (np.sum(ps_alpha**2)*self.M)
            epsilon[m0] = np.sum(
                (L - np.tile((hat_c*ps_alpha)[:, None], self.M))**2)
            # matrix(vec, nrow = length(vec), ncol = b) is a matrix with b
            # identical columns equal to vec
            # sum(matr) is the sum of all elements in the matrix matr

        de = np.argmin(epsilon)+1  # Missing values are discarded
        return np.array(de), epsilon[de-1]
