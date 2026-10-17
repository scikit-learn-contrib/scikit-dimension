# SPDX-License-Identifier: BSD-3-Clause
# Adapted from the MATLAB code of Amsaleg, Chelly, Houle, Kawarabayashi, Radovanovic and Treeratanajaru
# (https://github.com/radacha/tle; doi:10.1137/1.9781611975673.21, arXiv:2209.14475),
# originally released under CC BY-SA 4.0; see LICENSES_bundled.txt
import numpy as np
from .._commonfuncs import LocalEstimator
from scipy.spatial.distance import pdist, squareform
from joblib import Parallel, delayed


class TLE(LocalEstimator):
    """Intrinsic dimension estimation using the Tight Local intrinsic dimensionality Estimator algorithm. [Amsaleg2019]_ [IDRadovanović]_

    This is the variant including central measurements and reflections (``idtlec.m`` in [IDRadovanović]_).

    Parameters
    ----------
    epsilon: float
    """

    _N_NEIGHBORS = 20

    def __init__(
        self, epsilon=1e-4,
    ):
        self.epsilon = epsilon

    def _fit(self, X, dists, knnidx, n_jobs=1):
        self.dimension_pw_ = np.zeros(len(X))
        if n_jobs > 1:
            with Parallel(n_jobs=n_jobs) as parallel:
                # Asynchronously apply the `fit` function to each data point and collect the results
                results = parallel(
                    delayed(self._idtle)(
                        X[knnidx[i, :]], dists[[i], :]
                    ) for i in range(len(X))
                )
            self.dimension_pw_ = np.array(results)
        else:
            self.dimension_pw_ = np.zeros(len(X))
            for i in range(len(X)):
                self.dimension_pw_[i] = self._idtle(X[knnidx[i, :]], dists[[i], :])

    def _idtle(self, nn, dists):
        # nn - matrix of nearest neighbors (n_neighbors x d), sorted by distance
        # dists - nearest-neighbor distances (1 x n_neighbors), sorted
        r = dists[0, -1]  # distance to n_neighbors-th neighbor

        # Boundary case 1: If $r = 0$, this is fatal, since the neighborhood would be degenerate.
        if r == 0:
            raise ValueError("All k-NN distances are zero!")
        # Main computation
        n_neighbors = dists.shape[1]
        V = squareform(pdist(nn))
        Di = np.tile(dists.T, (1, n_neighbors))
        Dj = Di.T
        Z2 = 2 * Di ** 2 + 2 * Dj ** 2 - V ** 2
        S = (
            r
            * (
                ((Di ** 2 + V ** 2 - Dj ** 2) ** 2 + 4 * V ** 2 * (r ** 2 - Di ** 2))
                ** 0.5
                - (Di ** 2 + V ** 2 - Dj ** 2)
            )
            / (2 * (r ** 2 - Di ** 2))
        )
        T = (
            r
            * (
                ((Di ** 2 + Z2 - Dj ** 2) ** 2 + 4 * Z2 * (r ** 2 - Di ** 2)) ** 0.5
                - (Di ** 2 + Z2 - Dj ** 2)
            )
            / (2 * (r ** 2 - Di ** 2))
        )
        # handle case of repeating k-NN distances
        Dr = (dists == r).squeeze()
        S[Dr, :] = r * V[Dr, :] ** 2 / (r ** 2 + V[Dr, :] ** 2 - Dj[Dr, :] ** 2)
        T[Dr, :] = r * Z2[Dr, :] / (r ** 2 + Z2[Dr, :] - Dj[Dr, :] ** 2)
        # Boundary case 2: If $u_i = 0$, then for all $1\leq j\leq n_neighbors$ the measurements $s_{ij}$ and $t_{ij}$ reduce to $u_j$.
        Di0 = (Di == 0).squeeze()
        T[Di0] = Dj[Di0]
        S[Di0] = Dj[Di0]
        # Boundary case 3: If $u_j = 0$, then for all $1\leq j\leq n_neighbors$ the measurements $s_{ij}$ and $t_{ij}$ reduce to $\frac{r v_{ij}}{r + v_{ij}}$.
        Dj0 = (Dj == 0).squeeze()
        T[Dj0] = r * V[Dj0] / (r + V[Dj0])
        S[Dj0] = r * V[Dj0] / (r + V[Dj0])
        # Boundary case 4: If $v_{ij} = 0$, then the measurement $s_{ij}$ is zero and must be dropped. The measurement $t_{ij}$ should be dropped as well.
        V0 = (V == 0).squeeze()
        np.fill_diagonal(V0, False)
        # by setting to r, $t_{ij}$ will not contribute to the sum s1t
        T[V0] = r
        # by setting to r, $s_{ij}$ will not contribute to the sum s1s
        S[V0] = r
        # will subtract twice this number during ID computation below
        nV0 = np.sum(V0)
        # Drop T & S measurements below epsilon (V4: If $s_{ij}$ is thrown out, then for the sake of balance, $t_{ij}$ should be thrown out as well (or vice versa).)
        # also drop NaN measurements, obtained as 0/0 with duplicate points
        TSeps = (T < self.epsilon) | (S < self.epsilon) | np.isnan(T) | np.isnan(S)
        np.fill_diagonal(TSeps, 0)
        nTSeps = np.sum(TSeps)
        T[TSeps] = r
        T = np.log(T / r)
        S[TSeps] = r
        S = np.log(S / r)
        np.fill_diagonal(T, 0)  # delete diagonal elements
        np.fill_diagonal(S, 0)
        # Sum over the whole matrices
        s1t = np.sum(T)
        s1s = np.sum(S)
        # Drop distances below epsilon and compute sum
        Deps = dists < self.epsilon
        nDeps = np.sum(Deps, dtype=int)
        dists = dists[:, nDeps:]
        s2 = np.sum(np.log(dists / r))
        # Boundary case 5: (almost) all kNN distances are equal
        if s1t + s1s > -self.epsilon:
            return 0.0
        # Compute ID, subtracting numbers of dropped measurements
        ID = -2 * (n_neighbors ** 2 - nTSeps - nDeps - nV0) / (s1t + s1s + 2 * s2)
        return ID
