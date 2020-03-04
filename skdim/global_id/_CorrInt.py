import sys
sys.path.append('..')

import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from _commonfuncs import get_nn

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class CorrInt(BaseEstimator):
    
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    
    Attributes
    ----------
    k1 : int
        First neighborhood size considered
    k2 : int
        Last neighborhood size considered
    DM : bool, default=False
        Is the input a precomputed distance matrix (dense)
        
    Returns
    ----------
    
    dimension_ : float
        The estimated intrinsic dimension
    
    """
    def __init__(self,k1 = 10, k2 = 20, DM = False):
        self.k1 = k1
        self.k2 = k2
        self.DM = DM
        
    def fit(self,X):
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
        self.dimension_ = self._corrint(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self        

    def _corrint(self,X):

        n_elements = len(X)**2 #number of elements

        dists, _ = get_nn(X,self.k2)

        if self.DM is False:
            chunked_distmat = pairwise_distances_chunked(X)
        else:
            chunked_distmat = X

        r1 = np.median(dists[:, self.k1-1])
        r2 = np.median(dists[:, -1])


        n_diagonal_entries = len(X) #remove diagonal from sum count
        s1=-n_diagonal_entries
        s2=-n_diagonal_entries
        for chunk in chunked_distmat:
            s1+=(chunk<r1).sum()
            s2+=(chunk<r2).sum()


        Cr = np.array([s1/n_elements, s2/n_elements])
        estq = np.diff(np.log(Cr))/np.log(r2/r1)
        return(estq)
