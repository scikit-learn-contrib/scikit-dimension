import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

class KNN(BaseEstimator):    
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, k, ps, M, gamma = 2):
        self.k = k
        self.ps = ps
        self.M = M
        self.gamma = gamma
        
    def fit(self, X):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        self : object
            Returns self.
        """
        X = check_array(X, accept_sparse=False)
        self.dimension_, self.residual_ = self._knnDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self    

    def _knnDimEst(self, X):
        n = len(X)
        Q = len(self.ps)

        if min(self.ps) <= self.k or max(self.ps) > n:
            raise ValueError('ps must satisfy k<ps<len(X)')
        # Compute the distance between any two points in the X set
        dist = squareform(pdist(X))

        # Compute weighted graph length for each sample
        L = np.zeros((Q, self.M))

        for i in range(Q):
            for j in range(self.M):
                samp_ind = np.random.randint(0, n, self.ps[i])
                for l in samp_ind:
                    L[i, j] += np.sum(np.sort(dist[l, samp_ind])[1:(self.k + 1)]**self.gamma)
                    # Add the weighted sum of the distances to the k nearest neighbors.
                    # We should not include the sample itself, to which the distance is 
                    # zero. 


        # Least squares solution for m
        d = X.shape[1]
        epsilon = np.repeat(np.nan, d)
        for m0,m in enumerate(np.arange(1,d+1)):
            alpha = (m - self.gamma)/m
            ps_alpha = self.ps**alpha
            hat_c = np.sum(ps_alpha * np.sum(L, axis=1)) / (np.sum(ps_alpha**2)*self.M)
            epsilon[m0] = np.sum((L - np.tile((hat_c*ps_alpha)[:,None],self.M))**2)
            # matrix(vec, nrow = length(vec), ncol = b) is a matrix with b
            # identical columns equal to vec
            # sum(matr) is the sum of all elements in the matrix matr

        de = np.argmin(epsilon)+1 # Missing values are discarded
        return de, epsilon[de-1]