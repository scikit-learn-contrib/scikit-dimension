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
from scipy.spatial.distance import pdist, squareform
from .._commonfuncs import LocalEstimator


class MADA(LocalEstimator):
    """Intrinsic dimension estimation using the Manifold-Adaptive Dimension Estimation algorithm. [Farahmand2007]_, [IDHino]_

    MADA uses a variant of fractal dimension called the local information dimension.
    MADA considers the first order expansion of the probability mass around the inspection point,
    and it estimates the local information dimension by using two different radii from the inspection point. 

    Parameters
    ----------
    DM: bool
        Whether input is a precomputed distance matrix
    """

    _N_NEIGHBORS = 20

    def __init__(self, DM=False):
        self.DM = DM

    def _fit(self, **kwargs):
        self.dimension_pw_ = self._mada(kwargs["X"])

    def _mada(self, X):

        if self.DM is False:
            distmat = squareform(pdist(X))
        else:
            distmat = X

        distmat[distmat == 0] = np.max(distmat)

        sortedD = np.sort(distmat, axis=0, kind="mergesort")
        RK = sortedD[self.n_neighbors - 1, :]
        RK2 = sortedD[int(np.floor(self.n_neighbors / 2) - 1), :]
        ests = np.log(2) / np.log(RK / RK2)

        return ests
