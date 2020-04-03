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
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class MiND_ML(BaseEstimator):
    """
    Intrinsic dimension estimation with the MiND_MLx algorithms.

    -----------
    Attributes

    k : int, default=20
        Neighborhood parameter for ver='MLk' or ver='MLi'.
    ver : str
        'MLk' or 'MLi'. See the reference paper

    -----------
    Returns

    dimension_ : int
        Intrinsic dimension of the dataset

    -----------
    References
    
    Code translated and description taken from the intrinsicDimension R package by Kerstin Johnsson (https://cran.r-project.org/web/packages/intrinsicDimension/index.html).

    Rozza, A., Lombardi, G., Ceruti, C., Casiraghi, E., & Campadelli, P. (2012). Novel high intrinsic dimensionality estimators. 
    Machine Learning, 89(1-2), 37–65. doi:10.1007/s10994-012-5294-7 
    """

    def __init__(self, k=20, D=10, ver='MLk'):
        self.k = k
        self.D = D
        self.ver = ver

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
        if self.k+1 >= len(X):
            warnings.warn('k+1 >= len(X), using k+1 = len(X)-1')

        self.dimension_ = self._MiND_MLx(X)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _MiND_MLx(self, X):
        nbh_data, idx = get_nn(X, min(self.k+1, len(X)-1))

#         if (self.ver == 'ML1'):
#             return self._MiND_ML1(nbh_data)

        rhos = nbh_data[:, 0]/nbh_data[:, -1]

        d_MIND_MLi = self._MiND_MLi(rhos)
        if (self.ver == 'MLi'):
            return(d_MIND_MLi)

        d_MIND_MLk = self._MiND_MLk(rhos, d_MIND_MLi)
        if (self.ver == 'MLk'):
            return(d_MIND_MLk)
        else:
            raise ValueError("Unknown version: ", self.ver)

#     @staticmethod
#     def _MiND_ML1(nbh_data):
#         n = len(nbh_data)
#         #need only squared dists to first 2 neighbors
#         dists2 = nbh_data[:, :2]**2
#         s = np.sum(np.log(dists2[:, 0]/dists2[:, 1]))
#         ID = -2/(s/n)
#         return ID

    def _MiND_MLi(self, rhos):
        #MiND MLi MLk REVERSED COMPARED TO R TO CORRESPOND TO PAPER
        N = len(rhos)
        d_lik = np.array([np.nan]*self.D)
        for d in range(self.D):
            d_lik[d] = self._lld(d+1, rhos, N)
        return np.argmax(d_lik)+1

    def _MiND_MLk(self, rhos, dinit):
        #MiND MLi MLk REVERSED COMPARED TO R TO CORRESPOND TO PAPER
        res = minimize(fun=self._nlld,
                       x0=np.array([dinit]),
                       jac=self._nlld_gr,
                       args=(rhos, len(rhos)),
                       method='L-BFGS-B',
                       bounds=[(0, self.D)])

        return(res['x'][0])

    def _nlld(self, d, rhos, N):
        return(-self._lld(d, rhos, N))

    def _lld(self, d, rhos, N):
        if (d == 0):
            return(np.array([-1e30]))
        else:
            return N*np.log(self.k*d) + (d-1)*np.sum(np.log(rhos)) + (self.k-1)*np.sum(np.log(1-rhos**d))

    def _nlld_gr(self, d, rhos, N):
        if (d == 0):
            return(np.array([-1e30]))
        else:
            return -(N/d + np.sum(np.log(rhos) - (self.k-1)*(rhos**d)*np.log(rhos)/(1 - rhos**d)))
