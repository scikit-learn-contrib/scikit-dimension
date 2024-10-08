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
from scipy.spatial import distance
import numpy as np
import statistics
import math

from .._commonfuncs import GlobalEstimator


class PackingNumbers(GlobalEstimator):
    """Intrinsic dimension estimation using the Packing Numbers method.
        References: Intrinsic Dimension Estimation Using Packing Numbers
        by Balázs Kégl
        https://proceedings.neurips.cc/paper/2002/hash/1177967c7957072da3dc1db4ceb30e7a-Abstract.html
    """
    
    def __init__(self, r1=1, radius_multiplier=2, accuracy=0.9, metric="euclidean", iter_number=10):
        """Initialize the Packing Numbers estimator.
        Parameters
        r1 : float, default=1
            The first radius used for calculating the r-packing number.
        radius_multiplier : float, default=2
            Radius of second ball used for calculating the r-packing number is r1 * radius_multiplier
        accuracy : float, default=0.9
            The accuracy of the estimation.
        metric : string, default="euclidean"
            The metric used for calculating the distance between points.
        iter_number : int, default=10
            The number of iterations used for the estimation. For each iteration data is shuffled.
        ----------
        """
        self.r1 = r1
        self.r2 = r1 * radius_multiplier
        self.accuracy = accuracy
        self.metric = metric
        self.iter_number = iter_number
    
    def fit(self, X, y=None):
        """Fit Packing Numbers estimator.
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
        self._check_params()
        self.dimension_ = self._aproximate_dim_with_packing_numbers(X)
        self.is_fitted_ = True
        return self
    
    def _check_params(self):
        if self.r1 == self.r2:
            raise ValueError("r1 and r2 must be different")
        if self.r2 <= 0 or self.r1 <= 0:
            raise ValueError("r1 and r2 must be positive")
        if self.accuracy <= 0:
            raise ValueError("accuracy must be positive")
        if self.iter_number <= 0:
            raise ValueError("iter_number must be positive")
        
    def _aproximate_dim_with_packing_numbers(self, X):
        iter_counter = 0
        log_packing_numbers = [[], []]
        LOG_RS_DIFF = np.log(self.r2) - np.log(self.r1)
        RADIUSES = [self.r1, self.r2]
        MAX_ITER = 10 * self.iter_number
        while True:
            iter_counter += 1
            perm_set = np.random.permutation(X)
            for k in range(2):
                centers = []
                for i in range(len(perm_set)):
                    is_not_covered = True
                    for j in range(len(centers)):
                        if distance.cdist([perm_set[i]], [centers[j]], metric=self.metric)[0][0] < RADIUSES[k]:
                            is_not_covered = False
                            break
                    if is_not_covered:
                        centers.append(perm_set[i])
                log_packing_numbers[k].append(np.log(len(centers)))
            d_pack = (statistics.mean(log_packing_numbers[0]) - statistics.mean(log_packing_numbers[1])) / LOG_RS_DIFF
            if iter_counter > self.iter_number:
                if (math.sqrt(statistics.variance(log_packing_numbers[0]) + statistics.variance(log_packing_numbers[1]))
                    / (math.sqrt(iter_counter) * LOG_RS_DIFF ))  <= d_pack * (1.0 -self.accuracy) / 2 or iter_counter >= MAX_ITER:
                    return d_pack
