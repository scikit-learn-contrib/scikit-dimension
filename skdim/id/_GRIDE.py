# Copyright 2021-2023 The DADApy Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# BSD 3-Clause License
#
# Copyright (c) 2024, Jakub Malinowski
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


from sklearn.utils.validation import check_array, check_is_fitted

import math
import numpy as np
from scipy.spatial import distance
from functools import partial
from sklearn.metrics.pairwise import pairwise_distances_chunked
from .._commonfuncs import  GlobalEstimator


class Gride(GlobalEstimator):
    """Intrinsic dimension estimation using the GRIDE algorithm
        References:
        F. Denti, D. Doimo, A. Laio, A. Mira, Distributional results for model-based intrinsic dimension
            estimators, arXiv preprint arXiv:2104.13832 (2021).
    """
    
    def __init__(self, n1=1, n2=None, d0=0.001, d1=1000, eps=1e-7, range_max=None, metric="euclidean", n_jobs=1):
        """Initialize the GRIDE object.
        Parameters
        ----------
        n1 : int, default=1
            The oreder of nearest neighbors to consider for the first scale.
        n2 : int, default=None
            The order of nearest neighbors to consider for the second scale. If None, n2 = 2 * n1
        d0 : float, default=0.001
            The minimum dimension to consider in the search.
        d1 : float, default=1000
            The maximum dimension to consider in the search.
        eps : float, default=1e-7
            The precision of the approximate id calculation.
        range_max : int, default=None
            The maximum range of the neighbors to consider in the multi-scale estimation.
        metric : str, default="euclidean"
            The metric to use when calculating distances between points.
        """
        self.n_jobs = n_jobs
        self.n1 = n1
        if n2 == None:
            self.n2 = 2 * n1
        else:
            self.n2 = n2
        self.d0 = d0
        self.d1 = d1
        self.eps = eps
        self.range_max = range_max
        self.metric = metric
        # Object validation
        if n1 >= self.n2:
            raise ValueError("n1 must be smaller than n2")
        if type(n1) != int or type(self.n2) != int:
            raise ValueError("n1 and n2 must be integers")
        if n1 < 1 or self.n2 < 1:
            raise ValueError("n1 and n2 must be greater than 0")
        if d0 > d1:
            raise ValueError("Interval is not proper. d0 must be smaller than d1")
        if d0 < 0 or d1 < 0:
            raise ValueError("Dimenstions d0 and d1 must be greater than 0")
    
    def fit(self, X, y=None):
        """Implementation of single and multi scale intrinsic dimension estimation using the GRIDE algorithm.
        Mutli-scale estimation is performed when self.range_max is not None.
        Single scale estimation is performed always
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A data set for which the intrinsic dimension is estimated.
        y : dummy parameter to respect the sklearn API

        Returns
        self : object
            Returns self.
        """
        X = check_array(X, ensure_min_samples=2)
        if not self.range_max is None:
            self.dimension_array_ = self.__return_id_scaling_gride(X)

        self.dimension_ = self.__compute_id_gride_single_scale(None, self.n1, self.n2, self.eps, X)
        self.is_fitted_ = True
        return self

    def transform_multiscale(self, X=None):
        """Return the intrinsic dimension at different scales using the Gride algorithm.
        Parameters
        ----------
        X : 
            Ignored. Present for API consistency with sklearn."""
        check_is_fitted(self, "dimension_array_")
        return self.dimension_array_

    def __return_id_scaling_gride(self, X):
        """Compute the id at different scales using the Gride algorithm.

        Args:
            X (np.ndarray(float)): data matrix of size (n_samples, n_features)
        Returns:
            ids_scaling (np.ndarray(float)): array of intrinsic dimensions of length log2(range_max);
        """

        max_step = int(math.log(self.range_max, 2))
        mus = self.__return_mus_scaling(
            X,
            range_scaling=self.range_max
        )

        # compute IDs via maximum likelihood for all the scales up to range_max
        ids_scaling = self.__compute_id_gride_multiscale(
            mus, self.d0, self.d1, self.eps, X
        )

        return ids_scaling

    def __compute_id_gride_multiscale(self, mus, d0, d1, eps, X):
        """Compute the id using the gride algorithm.

        Helper of return return_id_gride.

        Args:
            mus (np.ndarray(float)): ratio of the distances of nth and 2nth nearest neighbours (Ndata x log2(range_max))
            d0 (float): minimum intrinsic dimension considered in the search;
            d1 (float): maximum intrinsic dimension considered in the search;
            eps (float): precision of the approximate id calculation.

        Returns:
            intrinsic_dim (np.ndarray(float): array of id estimates
        """
        # array of ids (as a function of the average distance to a point)
        ids_scaling = np.zeros(mus.shape[1])

        for i in range(mus.shape[1]):
            n1 = 2**i

            intrinsic_dim  = self.__compute_id_gride_single_scale(
               mus[:, i], n1, 2 * n1, eps, X
            )

            ids_scaling[i] = intrinsic_dim

        return ids_scaling

    def __compute_id_gride_single_scale(self, mus, n1, n2, eps, X):
        if mus is None:
            mus = _k_n_distances_ratios(X, n1, n2)

        id_ = _argmax_loglik(
            "float64", self.d0, self.d1, mus, n1, n2, eps=eps
        )  # eps=precision id calculation

        return id_

    def __mus_scaling_reduce_func(self, dist, start, range_scaling):
        """Help to compute the "mus" needed to compute the id.

        Applied at the end of pairwise_distance_chunked see:
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/metrics/pairwise.py#L1474

        Once a chunk of the distance matrix is computed _mus_scaling_reduce_func
        1) extracts the distances of the  neighbors of order 2**i up to the maximum
        neighbor range given by range_scaling
        2) computes the mus[i] (ratios of the neighbor distance of order 2**(i+1)
        and 2**i (see return id scaling gride)
        3) returns the chunked distances up to maxk, the mus, and rs, the distances
        of the neighbors involved in the estimate

        Args:
            dist: chunk of distance matrix passed internally by pairwise_distance_chunked
            start: dummy variable neede for compatibility with sklearn, not used
            range_scaling (int): maximum neighbor rank

        Returns:
            mus: ratios of the neighbor distances of order 2**(i+1) and 2**i
        """
        # argsort may be faster than argpartition when gride is applied on the full dataset (for the moment not used)
        max_step = int(math.log(range_scaling, 2))
        steps = np.array([2**i for i in range(max_step + 1)])

        sample_range = np.arange(dist.shape[0])[:, None]
        neigh_ind = np.argpartition(dist, steps[-1], axis=1)
        neigh_ind = neigh_ind[:, : steps[-1] + 1]

        # argpartition doesn't guarantee sorted order, so we sort again
        neigh_ind = neigh_ind[sample_range, np.argsort(dist[sample_range, neigh_ind])]

        dist = np.sqrt(dist[sample_range, neigh_ind])
        dist = _remove_zero_dists(dist)
        mus = dist[:, steps[1:]] / dist[:, steps[:-1]]

        return mus

    def __return_mus_scaling(self, X, range_scaling):
        """Return the "mus" needed to compute the id.

        Adapted from kneighbors function of sklearn
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/neighbors/_base.py#L596
        It allows to keep a nearest neighbor matrix up to rank 'maxk' (few tens of points)
        instead of 'range_scaling' (few thousands), while computing the ratios between neighbors' distances
        up to neighbors' rank 'range scaling'.
        For big datasets it avoids out of memory errors

        Args:
            range_scaling (int): maximum neighbor rank considered in the computation of the mu ratios

        Returns:
            mus np.ndarray(float)): the FULL matrix of the ratios of the neighbor distances of order 2**(i+1) and 2**i
        """
        reduce_func = partial(
            self.__mus_scaling_reduce_func, range_scaling=range_scaling
        )

        kwds = {"squared": True}
        chunked_results = np.array(list(
            pairwise_distances_chunked(
                X,
                X,
                reduce_func=reduce_func,
                metric=self.metric,
                n_jobs=self.n_jobs,
                working_memory=1024,
                **kwds,
            ))
        )

        return np.vstack(chunked_results)

def _neg_dloglik_did(d, mus, n1, n2, N, eps):
    """Compute the negative derivative of the log likelihood with respect to the id."""
    one_m_mus_d = 1.0 - mus ** (-d)
    "regularize small numbers"
    one_m_mus_d[one_m_mus_d < 2 * eps] = 2 * eps

    summation = np.sum(((1 - n2 + n1) / one_m_mus_d + n2 - 1.0) * np.log(mus))
    return summation - (N - 1) / d

def _filter_mus(dtype, mus, n1, n2):
    indx = np.nonzero(mus == 1)
    mus[indx] += 10 * np.finfo(dtype).eps

    q3, q2 = np.percentile(mus, [95, 50])
    mu_max = (
        20 * (q3 - q2) + q2
    )  # very generous threshold: to accomodate fat tailed distributions
    select = (
        mus < mu_max
    )  # remove high mu values related very likely to overlapping datapoints
    mus = mus[select]

    if isinstance(n1, np.ndarray):
        n1 = n1[select]

    if isinstance(n2, np.ndarray):
        n2 = n2[select]

    return mus, n1, n2

def _argmax_loglik(dtype, d0, d1, mus, n1, n2, eps=1.0e-7):
    mus, n1, n2 = _filter_mus(dtype, mus, n1, n2)

    N = len(mus)
    l1 = _neg_dloglik_did(d1, mus, n1, n2, N, eps)
    while abs(d0 - d1) > eps:
        d2 = (d0 + d1) / 2.0
        l2 = _neg_dloglik_did(d2, mus, n1, n2, N, eps)
        if l2 * l1 > 0:
            d1 = d2
        else:
            d0 = d2
    d = (d0 + d1) / 2.0

    return d

def _k_n_distances_ratios(dataset, k, n):
    """
    Return distances to the k-th and n-th nearest neighbors for all points in the dataset.

    Parameters:
    - dataset: numpy.ndarray of shape (num_points, num_dimensions), the dataset.
    - k: int, the index of the k-th nearest neighbor.
    - n: int, the index of the n-th nearest neighbor.

    Returns:
    - kth_distances: numpy.ndarray of shape (num_points,), distances to the k-th nearest neighbors.
    - nth_distances: numpy.ndarray of shape (num_points,), distances to the n-th nearest neighbors.
    """
    num_points = dataset.shape[0]
    
    # Initialize arrays to store distances
    kth_distances = np.zeros(num_points)
    nth_distances = np.zeros(num_points)
    
    # Compute pairwise distances for the entire dataset
    pairwise_distances = distance.cdist(dataset, dataset, 'euclidean')
    
    # Iterate over each point in the dataset
    for i in range(num_points):
        # Get distances from the current point to all other points
        distances = pairwise_distances[i]
        
        # Sort distances
        sorted_distances = np.sort(distances)
        
        kth_distances[i] = sorted_distances[k]
        nth_distances[i] = sorted_distances[n]

    for i in range(num_points):
        nth_distances[i] = nth_distances[i] / kth_distances[i]
    
    return nth_distances

def _remove_zero_dists(distances):
    """Find zero neighbour distances and substitute the numerical zeros with a very small number.
    This method is mostly useful to regularize the computation of certain id estimators.
    """
    # find all distances which are 0
    indices = np.nonzero(distances[:, 1:] < np.finfo("float64").eps)
    # set distance to epsilon
    distances[:, 1:][indices] = np.finfo("float64").eps

    return distances
