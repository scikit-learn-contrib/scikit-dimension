# SPDX-License-Identifier: BSD-3-Clause AND MIT
# Adapted from R package intrinsicDimension (https://cran.r-project.org/package=intrinsicDimension),
# Copyright (c) 2017 Kerstin Johnsson, MIT License; see LICENSES_bundled.txt
import numpy as np
import warnings
from .._commonfuncs import _check_X, get_nn, GlobalEstimator
from scipy.optimize import minimize
from sklearn.utils.validation import check_array


class MiND_ML(GlobalEstimator):
    """Intrinsic dimension estimation using the MiND_MLk and MiND_MLi algorithms. [Rozza2012]_ [IDJohnsson]_
    
    Parameters
    ----------
    k: int, default=20
        Neighborhood parameter for ver='MLk' or ver='MLi'.
    ver: str
        'MLk' or 'MLi'. See the reference paper
    """

    def __init__(self, k=20, D=10, ver="MLk"):
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
        X = _check_X(self, X, ensure_min_samples=2, ensure_min_features=2)

        if self.k + 1 >= len(X):
            warnings.warn("k+1 >= len(X), using k+1 = len(X)-1")

        self.dimension_ = self._MiND_MLx(X)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _MiND_MLx(self, X):
        nbh_data, idx = get_nn(X, min(self.k + 1, len(X) - 1))

        #         if (self.ver == 'ML1'):
        #             return self._MiND_ML1(nbh_data)

        rhos = nbh_data[:, 0] / nbh_data[:, -1]

        d_MIND_MLi = self._MiND_MLi(rhos)
        if self.ver == "MLi":
            return d_MIND_MLi

        d_MIND_MLk = self._MiND_MLk(rhos, d_MIND_MLi)
        if self.ver == "MLk":
            return d_MIND_MLk
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
        # MiND MLi MLk REVERSED COMPARED TO R TO CORRESPOND TO PAPER
        N = len(rhos)
        d_lik = np.array([np.nan] * self.D)
        for d in range(self.D):
            d_lik[d] = self._lld(d + 1, rhos, N)
        return np.argmax(d_lik) + 1

    def _MiND_MLk(self, rhos, dinit):
        # MiND MLi MLk REVERSED COMPARED TO R TO CORRESPOND TO PAPER
        res = minimize(
            fun=self._nlld,
            x0=np.array([dinit]),
            jac=self._nlld_gr,
            args=(rhos, len(rhos)),
            method="L-BFGS-B",
            bounds=[(0, self.D)],
        )

        return res["x"][0]

    def _nlld(self, d, rhos, N):
        return -self._lld(d, rhos, N)

    def _lld(self, d, rhos, N):
        if d == 0:
            return np.array([-1e30])
        else:
            return (
                N * np.log(self.k * d)
                + (d - 1) * np.sum(np.log(rhos))
                + (self.k - 1) * np.sum(np.log(1 - rhos ** d))
            )

    def _nlld_gr(self, d, rhos, N):
        if d == 0:
            return np.array([-1e30])
        else:
            return -(
                N / d
                + np.sum(
                    np.log(rhos)
                    - (self.k - 1) * (rhos ** d) * np.log(rhos) / (1 - rhos ** d)
                )
            )
