import numpy as np
from sklearn.decomposition import PCA

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class lPCA(BaseEstimator):    
    """ Intrinsic dimension estimation using local PCA.
    Version 'FO' is the method by Fukunaga-Olsen, version 'fan' is the method by Fan et al..
    Version 'maxgap' returns the position of the largest relative gap in the sequence of singular values.
    All versions assume that the data is local, i.e. that it is a neighborhood taken from a larger data set, such that the curvature and the noise within the neighborhood is relatively small. In the ideal case (no noise, no curvature) this is equivalent to the data being uniformly distributed over a hyper ball.

    Parameters
    ----------
    ver : str 	
        Version. Possible values: 'FO', 'fan', 'maxgap'.
    alphaRatio : float
        Only for ver = 'ratio'. Intrinsic dimension is estimated to be the number of principal components needed to retain (1-alphaRatio) percent of the variance.
    alphaFO: float
        Only for ver = 'FO'. An eigenvalue is considered significant if it is larger than alpha times the largest eigenvalue.
    alphaFan : float
        Only for ver = 'Fan'. The alpha parameter (large gap threshold).
    betaFan : float
        Only for ver = 'Fan'. The beta parameter (total covariance threshold).
    PFan : float
        Only for ver = 'Fan'. Total covariance in non-noise.
    verbose : bool, default=False
    
    References
    ----------
    Fukunaga, K. and Olsen, D. R. (1971). An algorithm for finding intrinsic dimensionality of data. IEEE Trans. Comput., c-20(2):176-183.
    
    Fan, M. et al. (2010). Intrinsic dimension estimation of data by principal component analysis. arXiv preprint 1002.2050. 
    """
    def __init__(self, ver = 'FO', alphaRatio = .05, alphaFO = .05, alphaFan = 10, betaFan = .8,
                 PFan = .95, verbose = True):
        
        self.ver = ver
        self.alphaRatio = alphaRatio
        self.alphaFO = alphaFO
        self.alphaFan = alphaFan
        self.betaFan = betaFan
        self.PFan = PFan
        self.verbose = verbose
        
    def fit(self,X):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A local dataset of training input samples.

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