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
# MIT License
#
# Copyright (c) 2019 fmottes
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

from sklearn.utils.validation import check_array

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.linear_model import LinearRegression
from .._commonfuncs import get_nn, GlobalEstimator


class TwoNN(GlobalEstimator):
    # SPDX-License-Identifier: MIT, 2019 Francesco Mottes [IDMottes]_
    """Intrinsic dimension estimation using the TwoNN algorithm. [Facco2019]_ [IDFacco]_ [IDMottes]_

    Parameters
    ----------  
    discard_fraction: float 
        Fraction (between 0 and 1) of largest distances to discard (heuristic from the paper)
    dist: bool
        Whether data is a precomputed distance matrix

    Attributes
    ----------
    x_: 1d array 
        np.array with the -log(mu) values.
    y_: 1d array 
        np.array with the -log(F(mu_{sigma(i)})) values.
    """

    def __init__(self, discard_fraction: float = 0.1, dist: bool = False):
        self.discard_fraction = discard_fraction
        self.dist = dist

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
        """
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

        self.dimension_, self.x_, self.y_ = self._twonn(X)

        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _twonn(self, X):
        """
        Calculates intrinsic dimension of the provided data points with the TWO-NN algorithm.

        -----------
        Parameters:

        X : 2d array-like
            2d data matrix. Samples on rows and features on columns.
        return_xy : bool (default=False)
            Whether to return also the coordinate vectors used for the linear fit.
        discard_fraction : float between 0 and 1
            Fraction of largest distances to discard (heuristic from the paper)
        dist : bool (default=False)
            Whether data is a precomputed distance matrix
        -----------
        Returns:

        d : float
            Intrinsic dimension of the dataset according to TWO-NN.
        x : 1d np.array (optional)
            Array with the -log(mu) values.
        y : 1d np.array (optional)
            Array with the -log(F(mu_{sigma(i)})) values.

        -----------
        References:

        E. Facco, M. d’Errico, A. Rodriguez & A. Laio
        Estimating the intrinsic dimension of datasets by a minimal neighborhood information (https://doi.org/10.1038/s41598-017-11873-y)
        """

        N = len(X)

        if self.dist:
            r1, r2 = X[:, 0], X[:, 1]
            _mu = r2 / r1
            # discard the largest distances
            mu = _mu[np.argsort(_mu)[: int(N * (1 - self.discard_fraction))]]

        else:
            # mu = r2/r1 for each data point
            # relatively high dimensional data, use distance matrix generator
            if X.shape[1] > 25:
                distmat_chunks = pairwise_distances_chunked(X)
                _mu = np.zeros((len(X)))
                i = 0
                for x in distmat_chunks:
                    x = np.sort(x, axis=1)
                    r1, r2 = x[:, 1], x[:, 2]
                    _mu[i : i + len(x)] = r2 / r1
                    i += len(x)

                # discard the largest distances
                mu = _mu[np.argsort(_mu)[: int(N * (1 - self.discard_fraction))]]

            else:  # relatively low dimensional data, search nearest neighbors directly
                dists, _ = get_nn(X, k=2)
                r1, r2 = dists[:, 0], dists[:, 1]
                _mu = r2 / r1
                # discard the largest distances
                mu = _mu[np.argsort(_mu)[: int(N * (1 - self.discard_fraction))]]

        # Empirical cumulate
        Femp = np.arange(1, 1 + int(N * (1 - self.discard_fraction))) / N

        # Fit line
        lr = LinearRegression(fit_intercept=False)
        lr.fit(np.log(mu).reshape(-1, 1), -np.log(1 - Femp).reshape(-1, 1))

        d = lr.coef_[0][0]  # extract slope

        return (
            d,
            np.log(mu).reshape(-1, 1),
            -np.log(1 - Femp).reshape(-1, 1),
        )
