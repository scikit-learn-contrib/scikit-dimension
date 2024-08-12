import numpy as np
import random
from functools import reduce

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_array
from scipy.sparse.csgraph import minimum_spanning_tree

from sklearn.utils.validation import check_array
from sklearn.linear_model import LinearRegression

from .._commonfuncs import GlobalEstimator


class MST(GlobalEstimator):

    def __init__(self, nmin = 2, step = 1, alpha = 1.0, k = 10, metric = 'euclidean', seed =12345):
        self.alpha = alpha
        self.nmin = nmin
        self.nstep = step
        self.k = k
        self.metric = metric 
        self.seed = seed

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self: object
            Returns self.
        self.dimension_: float
            The estimated intrinsic dimension
        self.score_: float
            Regression score
        """
        X = check_array(X, ensure_min_samples=self.nmin + self.nstep + 1, ensure_min_features=2)

        self.dimension_, self.score_ = self._MSTEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _MSTEst(self, X):

        NUMPOINTS = X.shape[0]
        random.seed(self.seed)
        
        D = squareform(pdist(X))
        D = np.triu(D)

        nrange = range(self.nmin, NUMPOINTS, self.nstep)

        E = [self._mst(D, n) for n in nrange]
        E = np.array(reduce(lambda xs, ys: xs + ys, E)) #flatten

        x = np.repeat(nrange, self.k).reshape([-1,1])

        self.x_ = np.log(x)
        self.y_ = np.log(E)

        reg = LinearRegression(fit_intercept = True).fit(self.x_, self.y_)
        score = reg.score(self.x_, self.y_)
        dim = np.divide(self.alpha,(1-reg.coef_))
        return dim, score
    
    def _mst(self, D, n):

        NUMPOINTS = D.shape[0]
        y = []
        for _ in range(self.k):
            idx = random.sample(range(NUMPOINTS),n)
            D_sample = csr_array(D[idx,:][:,idx])
            T = minimum_spanning_tree(D_sample)
            y.append(np.sum(np.power(T.data, self.alpha)))

        return y


