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
import itertools
import numbers
import multiprocessing as mp
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_array, check_is_fitted


def indComb(NN):
    pt1 = np.tile(range(NN), NN)
    pt2 = np.repeat(range(NN), NN)

    un = pt1 > pt2

    pt1 = pt1[un]
    pt2 = pt2[un]

    return pt1, pt2, np.hstack((pt2[:, None], pt1[:, None]))


def indnComb(NN, n):
    if n == 1:
        return np.arange(NN).reshape((-1, 1))
    prev = indnComb(NN, n - 1)
    lastind = prev[:, -1]
    ind_cf1 = np.repeat(lastind, NN)
    ind_cf2 = np.tile(np.arange(NN), len(lastind))
    # ind_cf2 = np.arange(NN)
    # for i in range(len(lastind)-1):
    #    ind_cf2 = np.concatenate((ind_cf2,np.arange(NN)))
    new_ind = np.where(ind_cf1 < ind_cf2)[0]
    new_ind1 = (new_ind - 1) // NN
    new_ind2 = new_ind % NN
    new_ind2[new_ind2 == 0] = NN
    return np.hstack((prev[new_ind1, :], np.arange(NN)[new_ind2].reshape((-1, 1))))


def efficient_indnComb(n, k, random_generator_):
    """
    memory-efficient indnComb:
    uniformly takes 5000 samples from itertools.combinations(n,k)
    """
    ncomb = binom_coeff(n, k)
    pop = itertools.combinations(range(n), k)
    targets = set(random_generator_.choice(ncomb, min(ncomb, 5000), replace=False))
    return np.array(
        list(itertools.compress(pop, map(targets.__contains__, itertools.count())))
    )


def lens(vectors):
    return np.sqrt(np.sum(vectors ** 2, axis=1))


def proxy(tup):
    function, X, Dict = tup
    return function(X, **Dict)


def get_nn(X, k, n_jobs=1):
    """Compute the k-nearest neighbors of a dataset np.array (n_samples x n_dims)"""
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
    neigh.fit(X)
    dists, inds = neigh.kneighbors(return_distance=True)
    return dists, inds


def asPointwise(data, class_instance, precomputed_knn=None, n_neighbors=100, n_jobs=1):
    """Use a global estimator as a pointwise one by creating kNN neighborhoods"""
    if precomputed_knn is not None:
        knn = precomputed_knn
    else:
        _, knn = get_nn(data, k=n_neighbors, n_jobs=n_jobs)

    if n_jobs > 1:
        pool = mp.Pool(n_jobs)
        results = pool.map(class_instance.fit, [data[i, :] for i in knn])
        pool.close()
        return np.array([r.dimension_ for r in results])
    else:
        return np.array([class_instance.fit(data[i, :]).dimension_ for i in knn])


class GlobalEstimator:
    """ Superclass: predict, fit_predict """

    def predict(self, X=None):
        """ Predict dimension after a previous call to self.fit

        Parameters
        ----------
        X : Dummy parameter

        Returns
        -------
        dimension_ : {int, float}
            The estimated intrinsic dimension
        """
        if X is not None:
            X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
        check_is_fitted(self, "is_fitted_")
        return self.dimension_

    def fit_predict(self, X, y=None):
        """Fit estimator and return dimension

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        dimension_ : {int, float}
            The estimated intrinsic dimension
        """
        return self.fit(X).dimension_


class PointwiseEstimator:
    """ Superclass: fit_pw, predict_pw, fit_predict_pw """

    def fit_pw(self, X, precomputed_knn=None, smooth=False, n_neighbors=100, n_jobs=1):
        """
        Creates an array of pointwise ID estimates (self.dimension_pw_) by fitting the estimator in kNN of each point.

        Parameters
        ----------
        X: np.array (n_samples x n_neighbors)
            Dataset to fit
        precomputed_knn: bool
            An array of precomputed (sorted) nearest neighbor indices
        n_neighbors:
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
            Returns self
        """
        X = check_array(X, ensure_min_samples=n_neighbors + 1, ensure_min_features=2)

        if precomputed_knn is not None:
            knnidx = precomputed_knn
        else:
            _, knnidx = get_nn(X, k=n_neighbors, n_jobs=n_jobs)

        if n_jobs > 1:
            pool = mp.Pool(n_jobs)
            results = pool.map(self.fit, [X[i, :] for i in knnidx])
            pool.close()
            self.dimension_pw_ = np.array([r.dimension_ for r in results])
        else:
            self.dimension_pw_ = np.array(
                [self.fit(X[i, :]).dimension_ for i in knnidx]
            )

        if smooth:
            self.dimension_pw_smooth_ = np.zeros(len(knnidx))
            for i, point_nn in enumerate(knnidx):
                self.dimension_pw_smooth_[i] = np.mean(
                    np.append(self.dimension_pw_[i], self.dimension_pw_[point_nn])
                )
        return self

    def predict_pw(self, X=None):
        """ Predict dimension after a previous call to self.fit

        Parameters
        ----------
        X : Dummy parameter

        Returns
        -------
        dimension_pw : np.array 
            Pointwise ID estimates
        dimension_pw_smooth : np.array 
            If self.fit_pw(smooth=True), additionally returns smoothed pointwise ID estimates
        """

        check_is_fitted(
            self,
            "dimension_pw_",
            msg=(
                "This class instance is not fitted yet. Call 'fit_pw' with "
                "appropriate arguments before using this method."
            ),
        )

        if hasattr(self, "dimension_pw_smooth_"):
            return self.dimension_pw_, self.dimension_pw_smooth_
        else:
            return self.dimension_pw_

    def fit_predict_pw(
        self, X, precomputed_knn=None, smooth=False, n_neighbors=100, n_jobs=1
    ):
        """
        Creates an array of pointwise ID estimates (self.dimension_pw_) by fitting the estimator in kNN of each point.

        Parameters
        ----------
        X: np.array (n_samples x n_neighbors)
            Dataset to fit
        precomputed_knn: bool
            An array of precomputed (sorted) nearest neighbor indices
        n_neighbors:
            Number of nearest neighbors to use (ignored when using precomputed_knn)
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by 
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
           smooth_ 

        Returns
        -------
        dimension_pw : np.array 
            Pointwise ID estimates
        dimension_pw_smooth : np.array 
            If smooth is True, additionally returns smoothed pointwise ID estimates
        """

        X = check_array(X, ensure_min_samples=n_neighbors + 1, ensure_min_features=2)

        if precomputed_knn is not None:
            knnidx = precomputed_knn
        else:
            _, knnidx = get_nn(X, k=n_neighbors, n_jobs=n_jobs)

        if n_jobs > 1:
            pool = mp.Pool(n_jobs)
            results = pool.map(self.fit, [X[i, :] for i in knnidx])
            pool.close()
            dimension_pw_ = np.array([r.dimension_ for r in results])
        else:
            dimension_pw_ = np.array([self.fit(X[i, :]).dimension_ for i in knnidx])

        if smooth:
            dimension_pw_smooth_ = np.zeros(len(knnidx))
            for i, point_nn in enumerate(knnidx):
                dimension_pw_smooth_[i] = np.mean(
                    np.append(dimension_pw_[i], dimension_pw_[point_nn])
                )
            return dimension_pw_, dimension_pw_smooth_
        else:
            return dimension_pw_


def mean_local_id(local_id, knnidx):
    """
    Compute the mean ID of all neighborhoods in which a point appears

    Parameters
    ----------
    local_id : list or np.array
        list of local ID for each point
    knnidx : np.array 
        indices of kNN for each point returned by function get_nn

    Results
    -------
    dimension_pw_smooth_ : np.array
        list of mean local ID for each point

    """
    dimension_pw_smooth_ = np.zeros(len(local_id))
    for point_i in range(len(local_id)):
        # get all points which have this point in their neighbourhoods
        all_neighborhoods_with_point_i = np.append(
            np.where(knnidx == point_i)[0], point_i
        )
        # get the mean local ID of these points
        dimension_pw_smooth_[point_i] = local_id[all_neighborhoods_with_point_i].mean()
    return dimension_pw_smooth_


def binom_coeff(n, k):
    """
    Taken from : https://stackoverflow.com/questions/26560726/python-binomial-coefficient
    Compute the number of ways to choose $k$ elements out of a pile of $n.$

    Use an iterative approach with the multiplicative formula:
    $$\frac{n!}{k!(n - k)!} =
    \frac{n(n - 1)\dots(n - k + 1)}{k(k-1)\dots(1)} =
    \prod_{i = 1}^{k}\frac{n + 1 - i}{i}$$

    Also rely on the symmetry: $C_n^k = C_n^{n - k},$ so the product can
    be calculated up to $\min(k, n - k).$

    :param n: the size of the pile of elements
    :param k: the number of elements to take from the pile
    :return: the number of ways to choose k elements out of a pile of n
    """

    # When k out of sensible range, should probably throw an exception.
    # For compatibility with scipy.special.{comb, binom} returns 0 instead.
    if k < 0 or k > n:
        return 0

    if k == 0 or k == n:
        return 1

    total_ways = 1
    for i in range(min(k, n - k)):
        total_ways = total_ways * (n - i) // (i + 1)

    return total_ways


def check_random_generator(seed):
    """Turn seed into a numpy.random._generator.Generator' instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random._generator.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random._generator.Generator"
        " instance" % seed
    )
