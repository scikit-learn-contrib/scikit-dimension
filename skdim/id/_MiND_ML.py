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
from .._commonfuncs import get_nn, GlobalEstimator
from scipy.optimize import minimize
from sklearn.utils.validation import check_array


class MiND_ML(GlobalEstimator):
    # SPDX-License-Identifier: MIT, 2017 Kerstin Johnsson [IDJohnsson]_
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
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

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
