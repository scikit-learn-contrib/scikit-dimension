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
import warnings
import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from .._commonfuncs import get_nn, GlobalEstimator
from sklearn.utils.validation import check_array


class CorrInt(GlobalEstimator):
    """Intrinsic dimension estimation using the Correlation Dimension. [Grassberger1983]_ [IDHino]_

    Parameters
    ----------
    k1: int
        First neighborhood size considered
    k2: int
        Last neighborhood size considered
    DM: bool, default=False
        Is the input a precomputed distance matrix (dense)
    """

    def __init__(self, k1=10, k2=20, DM=False):
        self.k1 = k1
        self.k2 = k2
        self.DM = DM

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

        if self.k2 >= len(X):
            warnings.warn("k2 larger or equal to len(X), using len(X)-1")
            self.k2 = len(X) - 1

        if self.k1 >= len(X) or self.k1 > self.k2:
            warnings.warn("k1 larger than k2 or len(X), using k2-1")
            self.k1 = self.k2 - 1

        self.dimension_ = self._corrint(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _corrint(self, X):

        n_elements = len(X) ** 2  # number of elements

        dists, _ = get_nn(X, self.k2)

        if self.DM is False:
            chunked_distmat = pairwise_distances_chunked(X)
        else:
            chunked_distmat = X

        r1 = np.median(dists[:, self.k1 - 1])
        r2 = np.median(dists[:, self.k2 - 1])

        n_diagonal_entries = len(X)  # remove diagonal from sum count
        s1 = -n_diagonal_entries
        s2 = -n_diagonal_entries
        for chunk in chunked_distmat:
            s1 += (chunk < r1).sum()
            s2 += (chunk < r2).sum()

        Cr = np.array([s1 / n_elements, s2 / n_elements])
        estq = np.diff(np.log(Cr)) / np.log(r2 / r1)
        return estq[0]
