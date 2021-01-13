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
import inspect
import scipy.integrate
import numpy as np
import warnings
from .._commonfuncs import lens, get_nn, LocalEstimator
from sklearn.utils.validation import check_array


class MLE(LocalEstimator):
    # SPDX-License-Identifier: MIT, 2017 Kerstin Johnsson [IDJohnsson]_
    """Intrinsic dimension estimation using the Maximum Likelihood algorithm. [Haro2008]_ [Hill1975]_ [Levina2005]_ [IDJohnsson]_

    The estimators are based on the referenced paper by Haro et al. (2008), using the assumption that there is a single manifold. 
    The estimator in the paper is obtained using default parameters and dnoise = 'dnoiseGaussH'.\n
    With integral.approximation = 'Haro' the Taylor expansion approximation of r^(m-1) that Haro et al. (2008) used are employed. \n
    With integral.approximation = 'guaranteed.convergence', r is factored out and kept and r^(m-2) is approximated with the corresponding Taylor expansion.\n 
    This guarantees convergence of the integrals. Divergence might be an issue when the noise is not sufficiently small in comparison to the smallest distances. 
    With integral.approximation = 'iteration', five iterations is used to determine m.\n

    Parameters
    ----------
    dnoise: None or 'dnoiseGaussH'
        Vector valued function giving the transition density. 'dnoiseGaussH' is the one used in Haro
    sigma: float, default=0
        Estimated standard deviation for the noise.
    n: int, default='None'
        Dimension of the noise (at least data.shape[1])
    integral.approximation: str, default='Haro'
        Can take values 'Haro', 'guaranteed.convergence', 'iteration'
    unbiased: bool, default=False
        Whether to correct bias or not
    neighborhood.based: bool, default=True
        Means that estimation is made for each neighborhood, otherwise the estimation is based on distances in the entire data set.
    comb: str, default='mle'
        How to aggregate the pointwise estimates. Possible values 'mle', 'mean', 'median'
    K: int, default=5
        Number of neighbors per data point that is considered, only used for neighborhood.based = FALSE
    """

    _N_NEIGHBORS = 20

    def __init__(
        self,
        dnoise=None,
        sigma=0,
        n=None,
        integral_approximation="Haro",
        unbiased=False,
        neighborhood_based=True,
        K=5,
    ):

        _, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def fit(
        self,
        X,
        y=None,
        precomputed_knn_arrays=None,
        smooth=False,
        n_neighbors=None,
        comb="mle",
        n_jobs=1,
    ):
        """Fitting method for local ID estimators
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API
        precomputed_knn_arrays: tuple[ np.array (n_samples x n_dims), np.array (n_samples x n_dims) ]
            Provide two precomputed arrays: (sorted nearest neighbor distances, sorted nearest neighbor indices)
        n_neighbors: int, default=self._N_NEIGHBORS
            Number of nearest neighbors to use (ignored when using precomputed_knn)
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by 
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
            smooth_ 

        Returns
        -------
        self : object
            Returns self.
        """
        # check inputs and define internal parameters
        if n_neighbors is None:
            n_neighbors = self._N_NEIGHBORS
        if n_neighbors >= len(X):
            warnings.warn("n_neighbors >= len(X), setting n_neighbors = len(X)-1")
            n_neighbors = len(X) - 1
        if self.K >= len(X):
            warnings.warn("self.K >= len(X), setting n_neighbors = len(X)-1")
            self.K = len(X) - 1
        self.n_neighbors = n_neighbors
        self.comb = comb

        X = check_array(
            X, ensure_min_samples=self.n_neighbors + 1, ensure_min_features=2
        )

        if precomputed_knn_arrays is not None:
            dists, knnidx = precomputed_knn_arrays
        else:
            if self.neighborhood_based:
                dists, knnidx = get_nn(X, k=self.n_neighbors, n_jobs=n_jobs)
            else:
                dists, knnidx = get_nn(X, k=self.K, n_jobs=n_jobs)

        if self.neighborhood_based:
            self.dimension_pw_ = self._maxLikPointwiseDimEst(dists)
            # combine local estimates
            if self.comb == "mean":
                self.dimension_ = np.mean(self.dimension_pw_)
            elif self.comb == "median":
                self.dimension_ = np.median(self.dimension_pw_)
            elif self.comb == "mle":
                self.dimension_ = 1 / np.mean(1 / self.dimension_pw_)
            else:
                raise ValueError(
                    "Invalid comb parameter. It has to be 'mean' or 'median'"
                )

            # compute smoothed local estimates
            if smooth:
                self.dimension_pw_smooth_ = np.zeros(len(knnidx))
                for i, point_nn in enumerate(knnidx):
                    self.dimension_pw_smooth_[i] = np.mean(
                        np.append(self.dimension_pw_[i], self.dimension_pw_[point_nn])
                    )
                self.is_fitted_pw_smooth_ = True
            self.is_fitted_pw_ = True

        else:
            Rs = np.sort(np.array(list(set(dists.flatten(order="F")))))[
                : self.n_neighbors
            ]
            # Since distances between points are used, noise is
            self.dimension_ = self._fit(Rs, np.sqrt(2) * self.sigma)
            # added at both ends, i.e. variance is doubled.
            # likelihood = np.nan

        self.is_fitted_ = True
        return self

    def fit_predict(
        self,
        X,
        y=None,
        precomputed_knn_arrays=None,
        smooth=False,
        n_neighbors=None,
        comb="mle",
        n_jobs=1,
    ):
        """Fit-predict method for local ID estimators
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API
        precomputed_knn_arrays: tuple[ np.array (n_samples x n_dims), np.array (n_samples x n_dims) ]
            Provide two precomputed arrays: (sorted nearest neighbor distances, sorted nearest neighbor indices)
        n_neighbors: int, default=self._N_NEIGHBORS
            Number of nearest neighbors to use (ignored when using precomputed_knn)
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by 
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
            smooth_ 

        Returns
        -------
        dimension_ : {int, float}
            The estimated intrinsic dimension
        """

        return self.fit(
            X,
            precomputed_knn_arrays=precomputed_knn_arrays,
            smooth=smooth,
            n_neighbors=n_neighbors,
            comb=comb,
            n_jobs=n_jobs,
        ).dimension_

    def _maxLikPointwiseDimEst(self, dists):
        # estimates dimension around each point in data[indices, ]
        #
        # 'indices' give the indexes for which local dimension estimation should
        # be performed.
        # 'k' is the number of neighbors used for each local dimension estimation.
        # 'dnoise' is a vector valued function giving the transition density.
        # 'sigma' is the estimated standard deviation for the noise.
        # 'n' is the dimension of the noise (at least dim(data)[2])

        # This vector will hold local dimension estimates
        de = np.repeat(np.nan, len(dists))

        for i in range(len(dists)):
            Rs = dists[i, :]
            de[i] = self._fit(Rs, self.sigma)

        return de

    def fit_once(self, X):
        # assuming data set is local
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
        center = np.mean(X, axis=0)
        cent_X = X - center
        Rs = np.sort(lens(cent_X))
        de = self._fit(Rs, self.sigma)
        return de

    def _fit(self, Rs, sigma):
        """ fit maxLikDimEstFromR """
        if self.integral_approximation not in [
            "Haro",
            "guaranteed.convergence",
            "iteration",
        ]:
            raise ValueError("Unknown integral_approximation parameter")

        if self.dnoise == "dnoiseGaussH":
            self.dnoise = self._dnoiseGaussH

        if not self.integral_approximation == "Haro" and self.dnoise is not None:
            self.dnoise = lambda r, s, sigma, k: r * self.dnoise(r, s, sigma, k)

        de = self._maxLikDimEstFromR_haro_approx(Rs, self.sigma)
        if self.integral_approximation == "iteration":
            raise ValueError(
                "integral_approximation='iteration' not implemented yet. See R intrinsicDimension package"
            )
            # de = maxLikDimEstFromRIterative(Rs, dnoise_orig, sigma, n, de, unbiased)

        return de

    def _maxLikDimEstFromR_haro_approx(self, Rs, sigma):
        # if dnoise is the noise function this is the approximation used in Haro.
        # for 'guaranteed.convergence' dnoise should be r times the noise function
        # with 'unbiased' option, estimator is unbiased if no noise or boundary

        k = len(Rs)
        kfac = k - 2 if self.unbiased else k - 1

        Rk = np.max(Rs)
        if self.dnoise is None:
            return kfac / (np.sum(np.log(Rk / Rs)))

        Rpr = Rk + 100 * sigma

        numerator = np.repeat(np.nan, k - 1)
        denominator = np.repeat(np.nan, k - 1)

        def numInt(x):
            return self.dnoise(x, Rj, sigma, self.n) * np.log(Rk / x)

        def denomInt(x):
            return self.dnoise(x, Rj, sigma, self.n)

        for j in range(k - 1):
            Rj = Rs[j]
            numerator[j] = scipy.integrate.quad(
                numInt, 0, Rpr, epsrel=1e-2, epsabs=1e-2
            )[0]
            denominator[j] = scipy.integrate.quad(
                denomInt, 0, Rpr, epsrel=1e-2, epsabs=1e-2
            )[0]

        return kfac / np.sum(numerator / denominator)

    @staticmethod
    def _dnoiseGaussH(r, s, sigma, k=None):
        return np.exp(-0.5 * ((s - r) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        # f(s|r) in Haro et al. (2008) w/ Gaussian
        # transition density
        # 'k' is not used, but is input
        # for compatibility

    # def maxLikDimEstFromRIterative(Rs, dnoise, sigma, n, init = 5,
    #                  unbiased = False, iterations = 5, verbose = False):
    #    m = init
    #    if verbose:
    #        print("Start iteration, intial value:", m, "\n")
    #    for i in range(iterations):
    #        m = maxLikDimEstFromRIterative_inner(Rs, dnoise, sigma, n, m, unbiased)
    #        if verbose:
    #            print("Iteration", i, ":", m, "\n")
    #        if verbose:
    #            print("\n")
    #    return(m)
    #
    # def maxLikDimEstFromRIterative_inner(Rs, dnoise, sigma, n, m, unbiased):
    #
    #    k = len(Rs)
    #    kfac = k-2 if unbiased else k-1
    #
    #    Rk = np.max(Rs)
    #    if dnoise is None:
    #        return(kfac/(np.sum(np.log(Rk/Rs))))
    #    Rpr = Rk + 100*sigma
    #
    #    numerator = np.repeat(np.nan, k - 1)
    #    denominator = np.repeat(np.nan, k - 1)
    #
    #    numInt = lambda x: x**(m-1)*dnoise(x, Rj, sigma, n) * np.log(Rk/x)
    #    denomInt = lambda x: x**(m-1)*dnoise(x, Rj, sigma, n)
    #
    #    for j in range(k-1):
    #        Rj = Rs[j]
    #        m = np.maximum(m, 1)
    #        numerator[j] = scipy.integrate.quad(numInt, 0, Rpr, epsrel = 1e-2,epsabs = 1e-2)[0]
    #        denominator[j] = scipy.integrate.quad(denomInt, 0, Rpr, epsrel = 1e-2,epsabs = 1e-2)[0]
    #
    #    return(kfac/sum(numerator/denominator))
