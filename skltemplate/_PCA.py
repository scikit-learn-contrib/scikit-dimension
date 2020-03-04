import numpy as np
from sklearn.decomposition import PCA

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class localPCA(BaseEstimator):    
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, ver = 'FO', alphaRatio = .05, alphaFO = .05, alphaFan = 10, betaFan = .8,
                 PFan = .95, maxdim = None,
                 verbose = True):
        
        self.ver = ver
        self.alphaRatio = alphaRatio
        self.alphaFO = alphaFO
        self.alphaFan = alphaFan
        self.betaFan = betaFan
        self.PFan = PFan
        self.maxdim = maxdim
        self.verbose = verbose
        
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
        self.dimension_, self.gap_ = self._pcaLocalDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self        


    def _pcaLocalDimEst(self, X):
        if self.maxdim is None:
            self.maxdim = X.shape[1]

        pca = PCA().fit(X)
        explained_var = pca.explained_variance_    

        if (self.ver == 'FO'): return(self._FO(explained_var))
        elif (self.ver == 'fan'): return(self._fan(explained_var))
        elif (self.ver == 'maxgap'): return(self._maxgap(explained_var))

    def _FO(self,explained_var):
        de = sum(explained_var>(self.alphaFO*explained_var[0]))
        gaps = explained_var[:-1]/explained_var[1:]
        try: return de, gaps[de-1]
        except: return de, gaps[-1]
    
    @staticmethod
    def _maxgap(explained_var):
        gaps = explained_var[:-1]/explained_var[1:]
        de = np.argmax(gaps)+1
        try: return de, gaps[de-1]
        except: return de, gaps[-1]
        
        
    def _ratio(self, explained_var):
        #X - data set (n x d)
        #theta - ratio of variance to preserve (theta \in [0,1])
        sumexp = np.cumsum(explained_var)
        de = np.where(sumexp>(1-self.alphaRatio))[0].min() + 1 
        return de
        
    def _fan(self,explained_var):
        r = np.where(np.cumsum(explained_var)/sum(explained_var) > self.PFan)[0][0]
        sigma = np.mean(explained_var[r:])
        explained_var -= sigma
        gaps = explained_var[:-1]/explained_var[1:]
        de = 1 + np.min(np.concatenate((np.where(gaps>self.alphaFan)[0],
                                    np.where((np.cumsum(explained_var)/sum(explained_var))>self.betaFan)[0])))
        try: return de, gaps[de-1]
        except: return de, gaps[-1]