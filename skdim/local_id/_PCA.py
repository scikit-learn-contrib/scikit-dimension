import numpy as np
from sklearn.decomposition import PCA

import numpy as np
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
        Version. Possible values: 'FO', 'fan', 'maxgap','ratio'.
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
        
    def fit(self,X,y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A local dataset of training input samples.
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

        if self.ver == 'ratio':
            self.dimension_ = self._pcaLocalDimEst(X)
        else:
            self.dimension_, self.gap_ = self._pcaLocalDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self        


    def _pcaLocalDimEst(self, X):
        pca = PCA().fit(X)
        explained_var = pca.explained_variance_    

        if (self.ver == 'FO'): return(self._FO(explained_var))
        elif (self.ver == 'fan'): return(self._fan(explained_var))
        elif (self.ver == 'maxgap'): return(self._maxgap(explained_var))
        elif (self.ver == 'ratio'): return(self._ratio(explained_var))


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
        idx_threshold = np.where(sumexp>(1-self.alphaRatio))[0]
        if len(idx_threshold) == 0:
            de = len(explained_var)
        else:
            de = idx_threshold.min() + 1 
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
        
        
##### dev in progress
#from sklearn.cluster import KMeans
#def pcaOtpmPointwiseDimEst(data, N, alpha = 0.05):
#    km = KMeans(n_clusters=N)
#    km.fit(data)
#    pt = km.cluster_centers_
#    pt_bm = km.labels_
#    pt_sm = np.repeat(np.nan, len(pt_bm))
#
#    for k in range(len(data)):
#        pt_sm[k] = np.argmin(lens(pt[[i for i in range(N) if i!=pt_bm[k]],:] - data[k,:]))
#        if (pt_sm[k] >= pt_bm[k]):
#            pt_sm[k] += 1
#
#    de_c = np.repeat(np.nan, N)
#    nbr_nb_c = np.repeat(np.nan, N)
#    for k in range(N):
#        nb = np.unique(np.concatenate((pt_sm[pt_bm == k], pt_bm[pt_sm == k]))).astype(int)
#        nbr_nb_c[k] = len(nb)
#        loc_dat = pt[nb,:] - pt[k,:]
#        if len(loc_dat) == 1:
#            continue
#        de_c[k] = lPCA().fit(loc_dat).dimension_ #pcaLocalDimEst(loc_dat, ver = "FO", alphaFO = alpha)
#
#    de = de_c[pt_bm]
#    nbr_nb = nbr_nb_c[pt_bm]
#    return de, nbr_nb
