import numpy as np
import warnings
from .._commonfuncs import get_nn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class MOM(BaseEstimator):
    """
    Class to calculate the intrinsic dimension of the provided data points with the Tight Locality Estimator algorithm.
    
    -----------
    Attributes
    
   epsilon : float
   
    -----------
    Returns
    
    dimension_ : int
        Intrinsic dimension of the dataset

    -----------
    References:
    """
    
    
    def __init__(self, k = 20):
        self.k = k
        
    def fit(self,X,y=None):
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
        X = check_array(X, accept_sparse=False)
        if len(X) == 1:
            raise ValueError("Can't fit with 1 sample")
        if X.shape[1]==1:
            raise ValueError("Can't fit with n_features = 1")
        if not np.isfinite(X).all():
            raise ValueError("X contains inf or NaN")
        if self.k >= len(X):
            warnings.warn('k >= len(X), using k = len(X)-1')
        
        
        dists, inds = get_nn(X,min(self.k,len(X)-1))
        
        self.dimension_ = self._idmom(dists)
                    
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self  

    def _idmom(self, dists):
        # dists - nearest-neighbor distances (k x 1, or k x n), sorted
        w = dists[:,-1]
        m1 = np.sum(dists, axis=1)/dists.shape[1]
        ID = -m1 / (m1-w)
        return ID
