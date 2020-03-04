import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class Mada(BaseEstimator):
    
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, k = 20, comb = "average", DM = False, local = False):
        self.k = k
        self.comb = comb
        self.DM = DM
        self.local = local
        
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
        self.dimension_ = self._mada(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self        

    def _mada(self,X):

        if (self.DM == False):
            distmat = squareform(pdist(X))

        else:
            distmat = X

        n = len(distmat)

        if (self.local == False):  
            ID = np.random.choice(n, size=int(np.round(n/2)), replace = False)
            tmpD = distmat[ID,:]
            tmpD[tmpD == 0] = np.max(tmpD)

        else:
            tmpD = distmat
            tmpD[tmpD == 0] = np.max(tmpD)

        sortedD = np.sort(tmpD,axis=0,kind='mergesort')
        RK = sortedD[self.k-1,:]
        RK2 = sortedD[int(np.floor(self.k/2)-1), ]
        ests = np.log(2)/np.log(RK/RK2)
        
        if (self.local == True):  
            return(ests)

        if (self.comb == "average"):
            return np.mean(ests)
        elif (self.comb == "median"):
            return np.median(ests)