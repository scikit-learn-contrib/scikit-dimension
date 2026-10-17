# SPDX-License-Identifier: BSD-3-Clause
from numbers import Integral

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist, squareform
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state

from .._commonfuncs import _check_X, GlobalEstimator


class PH(GlobalEstimator):
    """Intrinsic dimension estimation from 0-dimensional persistent homology (PH0 dimension). [Schweinhart2020]_ [Jaquette2020]_ [Binnie2025]_

    For random subsamples of increasing size n, the total alpha-weighted length
    E = sum(l**alpha) of the Euclidean minimum spanning tree (the PH0 total persistence)
    grows as n**((d - alpha) / d). The dimension is estimated as alpha / (1 - slope),
    where slope is fitted by least squares to log E against log n.
    Check the fit by plotting ``x_`` against ``y_``.

    Cost: one MST per subsample on a dense distance matrix, i.e. nsteps * subsamples
    MSTs of up to nmax_ points, and O(n_samples**2) memory for the distance matrix.

    Parameters
    ----------
    alpha: float or sequence of float
        Persistence power(s), > 0. A sequence gives one estimate per value.
    n_range_min: float or int
        Smallest subsample size, as a fraction of n_samples if range_type='fraction',
        else as a number of points.
    n_range_max: float or int
        Largest subsample size, same convention as n_range_min.
    range_type: str
        'fraction' or 'num'.
    nsteps: int
        Number of subsample sizes, evenly spaced between n_range_min and n_range_max.
    subsamples: int
        Number of random subsamples per size.
    metric: str or callable
        Distance metric, passed to scipy.spatial.distance.pdist.
    random_state: int, RandomState instance or None
        Seed for subsampling.
    n_jobs: int
        Number of joblib workers for the MST computations.
        Results do not depend on n_jobs.

    Attributes
    ----------
    nmin_: int
        Smallest subsample size used.
    nmax_: int
        Largest subsample size used.
    x_: 2d array, shape (nsteps * subsamples, 1)
        log subsample sizes.
    y_: array, shape (nsteps * subsamples,) or (nsteps * subsamples, len(alpha))
        log total persistence of each subsample.
    reg_: sklearn.linear_model.LinearRegression
        Regression of y_ on x_.

    References
    ----------
    .. [Schweinhart2020] Schweinhart, B. (2020). Fractal dimension and the persistent homology
       of random geometric complexes. Advances in Mathematics, 372, 107291. arXiv:1808.01079
    .. [Jaquette2020] Jaquette, J. & Schweinhart, B. (2020). Fractal dimension estimation with
       persistent homology: a comparative study. Communications in Nonlinear Science and
       Numerical Simulation, 84, 105163. doi:10.1016/j.cnsns.2019.105163
    .. [Binnie2025] Binnie, J., Dłotko, P., Harvey, J., Malinowski, J. & Yim, K. M. (2025).
       A survey of dimension estimation methods. arXiv:2507.13887

    Implementation adapted from the kmyim/scikit-dimension fork (Ka Man Yim) [IDkmyim]_.
    """

    def __init__(
        self,
        alpha=1.0,
        n_range_min=0.5,
        n_range_max=1.0,
        range_type="fraction",
        nsteps=100,
        subsamples=10,
        metric="euclidean",
        random_state=None,
        n_jobs=1,
    ):
        self.alpha = alpha
        self.n_range_min = n_range_min
        self.n_range_max = n_range_max
        self.range_type = range_type
        self.nsteps = nsteps
        self.subsamples = subsamples
        self.metric = metric
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self: object
            Returns self.
        """
        X = _check_X(self, X, ensure_min_samples=3)
        n = X.shape[0]
        alpha = np.asarray(self.alpha, dtype=float)
        if np.any(alpha <= 0):
            raise ValueError("alpha must be strictly positive.")
        if not isinstance(self.nsteps, Integral) or self.nsteps < 2:
            raise ValueError("nsteps must be an integer >= 2.")
        if not isinstance(self.subsamples, Integral) or self.subsamples < 1:
            raise ValueError("subsamples must be an integer >= 1.")
        if self.range_type == "fraction":
            if not 0 < self.n_range_min < self.n_range_max <= 1:
                raise ValueError("Need 0 < n_range_min < n_range_max <= 1 for range_type='fraction'.")
            self.nmin_ = int(np.ceil(self.n_range_min * n))
            self.nmax_ = int(np.ceil(self.n_range_max * n))
        elif self.range_type == "num":
            if not all(isinstance(v, Integral) for v in (self.n_range_min, self.n_range_max)):
                raise ValueError("n_range_min and n_range_max must be integers for range_type='num'.")
            self.nmin_, self.nmax_ = int(self.n_range_min), int(self.n_range_max)
        else:
            raise ValueError("range_type should either be 'fraction', or 'num'.")
        if not 2 <= self.nmin_ < self.nmax_ <= n:
            raise ValueError(
                f"Need 2 <= min subsample size < max subsample size <= n_samples, got {self.nmin_}, {self.nmax_}, {n}."
            )

        D = squareform(pdist(X, metric=self.metric))
        sizes = np.repeat(np.ceil(np.linspace(self.nmin_, self.nmax_, self.nsteps)).astype(int), self.subsamples)
        # draw all subsamples here so the result does not depend on n_jobs
        rng = check_random_state(self.random_state)
        idx = [rng.choice(n, size=s, replace=False) for s in sizes]
        E = Parallel(n_jobs=self.n_jobs)(delayed(_total_persistence)(D[np.ix_(i, i)], alpha) for i in idx)

        self.x_ = np.log(sizes).reshape(-1, 1)
        self.y_ = np.log(np.array(E))
        self.reg_ = LinearRegression().fit(self.x_, self.y_)
        dim = alpha / (1 - np.ravel(self.reg_.coef_))
        self.dimension_ = dim[0] if alpha.ndim == 0 else dim
        self.is_fitted_ = True
        return self


def _total_persistence(D, alpha):
    """Sum of MST edge lengths to the power alpha (PH0 total persistence)"""
    # dense csgraph input treats |d| <= 1e-8 as a missing edge (duplicate points would be
    # joined by a longer edge); sparse input keeps explicit zeros as edges
    i, j = np.triu_indices(len(D), 1)
    G = csr_matrix((D[i, j], (i.astype(np.int32), j.astype(np.int32))), shape=D.shape)
    w = minimum_spanning_tree(G).data
    return np.sum(w[:, None] ** np.atleast_1d(alpha), axis=0) if alpha.ndim else np.sum(w**alpha)
