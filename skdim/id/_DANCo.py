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


from sklearn.utils.validation import check_array, check_random_state

import sys
import warnings
import numpy as np
from scipy.optimize import minimize
from scipy.special import i0, i1, digamma
from scipy.interpolate import interp1d
from ..datasets import hyperBall
from .._commonfuncs import (
    binom_coeff,
    get_nn,
    lens,
    indnComb,
    GlobalEstimator,
)


class DANCo(GlobalEstimator):
    # SPDX-License-Identifier: MIT, 2017 Kerstin Johnsson [IDJohnsson]_
    """Intrinsic dimension estimation using the Dimensionality from Angle and Norm Concentration algorithm. [Ceruti2012]_ [IDLombardi]_ [IDJohnsson]_ 

    Parameters
    ----------
    k: int, default=10
        Neighborhood parameter.
    D: int, default=None
        Maximal dimension
    ver: str, default='DANCo'
        Version to use. possible values: 'DANCo', 'MIND_MLi', 'MIND_MLk'.
    calibration_data: dict, default=None
        Precomputed calibration data. 
    fractal: bool, default=True
        Whether to return fractal rather than integer dimension
    verbose: bool, default=False
    """

    def __init__(
        self,
        k=10,
        D=None,
        calibration_data=None,
        ver="DANCo",
        fractal=True,
        verbose=False,
        random_state=None,
    ):
        self.k = k
        self.D = D
        self.calibration_data = calibration_data
        self.ver = ver
        self.verbose = verbose
        self.fractal = fractal
        self.random_state = random_state

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A data set for which the intrinsic dimension is estimated.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self : object
            Returns self.
        self.dimension_ : int (or float if fractal is True)
            The estimated intrinsic dimension
        self.kl_divergence : float
            The KL divergence between data and reference data for the estimated dimension (if ver == 'DANCo').
        self.calibration_data : dict
            Calibration data that can be reused when applying DANCo to data sets of the same size with the same neighborhood parameter k.
        """
        X = check_array(X, ensure_min_samples=self.k + 1, ensure_min_features=2)

        if self.k >= len(X):
            warnings.warn("k larger or equal to len(X), using len(X)-2")
            self._k = len(X) - 2
        else:
            self._k = self.k

        self._D = X.shape[1] if self.D is None else self.D

        self.random_state_ = check_random_state(self.random_state)

        if self.ver not in ["DANCo", "DANCoFit"]:
            self.dimension_ = self._dancoDimEst(X)
        else:
            (
                self.dimension_,
                self.kl_divergence_,
                self.calibration_data_,
            ) = self._dancoDimEst(X)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _KL(self, nocal, caldat):
        kld = self._KLd(nocal["dhat"], caldat["dhat"])
        klnutau = self._KLnutau(
            nocal["mu_nu"], caldat["mu_nu"], nocal["mu_tau"], caldat["mu_tau"]
        )
        # print(klnutau)
        return kld + klnutau

    def _KLd(self, dhat, dcal):
        H_k = np.sum(1 / np.arange(1, self._k + 1))
        quo = dcal / dhat
        a = (
            np.power(-1, np.arange(self._k + 1))
            * np.array(list(binom_coeff(self._k, i) for i in range(self._k + 1)))
            * digamma(1 + np.arange(self._k + 1) / quo)
        )
        return H_k * quo - np.log(quo) - (self._k - 1) * np.sum(a)

    @staticmethod
    def _KLnutau(nu1, nu2, tau1, tau2):
        return np.log(
            min(sys.float_info.max, i0(tau2)) / min(sys.float_info.max, i0(tau1))
        ) + min(sys.float_info.max, i1(tau1)) / min(sys.float_info.max, i0(tau1)) * (
            tau1 - tau2 * np.cos(nu1 - nu2)
        )

    def _nlld(self, d, rhos, N):
        return -self._lld(d, rhos, N)

    def _lld(self, d, rhos, N):
        if d == 0:
            return np.array([-1e30])
        else:
            return (
                N * np.log(self._k * d)
                + (d - 1) * np.sum(np.log(rhos))
                + (self._k - 1) * np.sum(np.log(1 - rhos ** d))
            )

    def _nlld_gr(self, d, rhos, N):
        if d == 0:
            return np.array([-1e30])
        else:
            return -(
                N / d
                + np.sum(
                    np.log(rhos)
                    - (self._k - 1) * (rhos ** d) * np.log(rhos) / (1 - rhos ** d)
                )
            )

    def _MIND_MLi(self, rhos, D):
        N = len(rhos)
        d_lik = np.array([np.nan] * D)
        for d in range(D):
            d_lik[d] = self._lld(d + 1, rhos, N)
        return np.argmax(d_lik) + 1

    def _MIND_MLk(self, rhos, D, dinit):
        res = minimize(
            fun=self._nlld,
            x0=np.array([dinit]),
            jac=self._nlld_gr,
            args=(rhos, len(rhos)),
            method="L-BFGS-B",
            bounds=[(0, D)],
        )
        return res["x"][0]

    def _MIND_MLx(self, X, D):
        nbh_data, idx = get_nn(X, self._k + 1)
        rhos = nbh_data[:, 0] / nbh_data[:, -1]

        d_MIND_MLi = self._MIND_MLi(rhos, D)
        if self.ver == "MIND_MLi":
            return d_MIND_MLi

        d_MIND_MLk = self._MIND_MLk(rhos, D, d_MIND_MLi)
        if self.ver == "MIND_MLk":
            return d_MIND_MLk
        else:
            raise ValueError("Unknown version: ", self.ver)

    @staticmethod
    def _Ainv(eta):
        if eta < 0.53:
            return 2 * eta + eta ** 3 + 5 * (eta ** 5) / 6
        elif eta < 0.85:
            return -0.4 + 1.39 * eta + 0.43 / (1 - eta)
        else:
            return 1 / ((eta ** 3) - 4 * (eta ** 2) + 3 * eta)

    def _loc_angles(self, pt, nbs):
        vec = nbs - pt
        # if(len(pt) == 1):
        #     vec = vec.T
        vec_len = lens(vec)
        combs = indnComb(len(nbs), 2).T
        sc_prod = np.sum(vec[combs[0, :]] * vec[combs[1, :]], axis=1)
        # if (length(pt) == 1) {
        # print(sc.prod)
        # print((vec.len[combs[1, ]]*vec.len[combs[2, ]]))
        # }
        cos_th = sc_prod / (vec_len[combs[0, :]] * vec_len[combs[1, :]])

        if np.any(np.abs(cos_th) > 1):
            np.clip(cos_th, a_min=None, a_max=1, out=cos_th)
            if not hasattr(self, "_warned"):
                self._warned = True
                print(
                    "Warning: your data might contain duplicate rows, which can affect results"
                )
        return np.arccos(cos_th)

    def _angles(self, X, nbs):
        N = len(X)

        thetas = np.zeros((N, binom_coeff(self._k, 2)))
        for i in range(N):
            nb_data = X[
                nbs[i,],
            ]
            thetas[i,] = self._loc_angles(X[i,], nb_data)
        return thetas

    def _ML_VM(self, thetas):
        sinth = np.sin(thetas)
        costh = np.cos(thetas)
        nu = np.arctan(np.sum(sinth) / np.sum(costh))
        eta = np.sqrt(np.mean(costh) ** 2 + np.mean(sinth) ** 2)
        tau = self._Ainv(eta)
        return dict(nu=nu, tau=tau)

    def _dancoDimEstNoCalibration(self, X, D, n_jobs=1):
        nbh_data, idx = get_nn(X, self._k + 1, n_jobs=n_jobs)
        rhos = nbh_data[:, 0] / nbh_data[:, -1]
        d_MIND_MLi = self._MIND_MLi(rhos, D)
        d_MIND_MLk = self._MIND_MLk(rhos, D, d_MIND_MLi)

        thetas = self._angles(X, idx[:, : self._k])
        ml_vm = list(map(self._ML_VM, thetas))
        mu_nu = np.mean([i["nu"] for i in ml_vm])
        mu_tau = np.mean([i["tau"] for i in ml_vm])
        if X.shape[1] == 1:
            mu_tau = 1

        return dict(dhat=d_MIND_MLk, mu_nu=mu_nu, mu_tau=mu_tau)

    def _DancoCalibrationData(self, N):
        me = dict(k=self._k, N=N, calibration_data=list(), maxdim=0)
        return me

    def _increaseMaxDimByOne(self, dancoCalDat):
        newdim = dancoCalDat["maxdim"] + 1
        MIND_MLx_maxdim = newdim * 2 + 5
        dancoCalDat["calibration_data"].append(
            self._dancoDimEstNoCalibration(
                hyperBall(
                    dancoCalDat["N"],
                    newdim,
                    1,
                    center=[0] * newdim,
                    random_state=self.random_state_,
                ),
                dancoCalDat["k"],
                MIND_MLx_maxdim,
            )
        )
        dancoCalDat["maxdim"] = newdim
        return dancoCalDat

    #    @staticmethod
    #    def increaseMaxDimByOne_precomputedSpline(dancoCalDat,DANCo_splines):
    #        newdim = dancoCalDat['maxdim'] + 1
    #        dancoCalDat['calibration_data'].append({'dhat':DANCo_splines['spline_dhat'](newdim,dancoCalDat['N']),
    #                                                'mu_nu':DANCo_splines['spline_mu'](newdim,dancoCalDat['N']),
    #                                                'mu_tau':DANCo_splines['spline_tau'](newdim,dancoCalDat['N'])})
    #        dancoCalDat['maxdim'] = newdim
    #        return(dancoCalDat)

    def _computeDANCoCalibrationData(self, N):
        print("Computing calibration X...\nCurrent dimension: ", end=" ")
        cal = self._DancoCalibrationData(self._k, N)
        while cal["maxdim"] < self._D:
            if cal["maxdim"] % 10 == 0:
                print(cal["maxdim"], end=" ")
            cal = self._increaseMaxDimByOne(cal)
        return cal

    def _dancoDimEst(self, X):

        cal = self.calibration_data
        N = len(X)

        if cal is not None:
            if cal["k"] != self._k:
                raise ValueError(
                    "Neighborhood parameter k = %s does not agree with neighborhood parameter of calibration data, cal$k = %s",
                    self._k,
                    cal["k"],
                )
            if cal["N"] != N:
                raise ValueError(
                    "Number of data points N = %s does not agree with number of data points of calibration data, cal$N = %s",
                    N,
                    cal["N"],
                )

        if self.ver not in ["DANCo", "DANCoFit"]:
            return self._MIND_MLx(X, self._D)

        nocal = self._dancoDimEstNoCalibration(X, self._D)
        if any(np.isnan(val) for val in nocal.values()):
            return np.nan, np.nan, cal

        if cal is None:
            cal = self._DancoCalibrationData(N)

        if cal["maxdim"] < self._D:

            if self.ver == "DANCoFit":
                if self.verbose:
                    print(
                        "Generating DANCo calibration data from precomputed spline interpolation for cardinality 50 to 5000, k = 10, dimensions 1 to 100"
                    )
                raise ValueError("DANCoFit not yet implemented")
                # load precomputed splines as a function of dimension and dataset cardinality
                # DANCo_splines = {}
                # for spl in ['spline_dhat','spline_mu','spline_tau']:
                #    with open('DANCoFit/DANCo_'+spl+'.pkl', 'rb') as f:
                #        DANCo_splines[spl]=pickle.load(f)
                ##compute interpolated statistics
                # while (cal['maxdim'] < self._D):
                #    cal = self.increaseMaxDimByOne_precomputedSpline(cal,DANCo_splines)

            else:
                if self.verbose:
                    print(
                        "Computing DANCo calibration data for N = {}, k = {} for dimensions {} to {}".format(
                            N, self._k, cal["maxdim"] + 1, self._D
                        )
                    )

                # compute statistics
                while cal["maxdim"] < self._D:
                    cal = self._increaseMaxDimByOne(cal)

        kl = np.array([np.nan] * self._D)
        for d in range(self._D):
            kl[d] = self._KL(nocal, cal["calibration_data"][d])

        de = np.argmin(kl) + 1

        if self.fractal:
            # Fitting with a cubic smoothing spline:
            if X.shape[1] > 3:
                kind = "cubic"
            elif X.shape[1] == 3:
                kind = "quadratic"
            elif X.shape[1] == 2:
                kind = "linear"

            f = interp1d(
                np.arange(1, self._D + 1),
                kl,
                kind=kind,
                bounds_error=False,
                fill_value=(1, self._D + 1),
            )
            # Locating the minima:
            de_fractal = minimize(f, de, bounds=[(1, self._D + 1)], tol=1e-3)["x"]
            return de_fractal[0], kl[de - 1], cal
        else:
            return de, kl[de - 1], cal
