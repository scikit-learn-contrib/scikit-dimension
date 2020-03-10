import numpy as np
import warnings
from _commonfuncs import get_nn
from scipy.spatial.distance import pdist, squareform
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_random_state


class MiND_ML(BaseEstimator):
    """
    Class to calculate the intrinsic dimension of the provided data points with the MiND_MLx algorithms.
    
    -----------
    Attributes
    
    k : int, default=20
        Neighborhood parameter for ver='MLk' or ver='MLi'. ver='ML1' uses the first two neighbor distances.
    ver : str
        'MLk' or 'MLi' or 'ML1'. See the reference paper
    -----------
    Returns
    
    dimension_ : int
        Intrinsic dimension of the dataset

    -----------
    References
    
    Rozza, A., Lombardi, G., Ceruti, C., Casiraghi, E., & Campadelli, P. (2012). Novel high intrinsic dimensionality estimators. 
    Machine Learning, 89(1-2), 37–65. doi:10.1007/s10994-012-5294-7 
    """
    
    
    def __init__(self, k = 20, D = 10, ver = 'MLk'):
        self.k = k
        self.D = D
        self.ver = ver
        
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
        if self.k+1 >= len(X):
            warnings.warn('k+1 >= len(X), using k+1 = len(X)-1')
                
        self.dimension_ = self._MiND_MLx(X)
        
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self  

    def _MiND_MLx(self, X):
        nbh_data,idx = get_nn(X, min(self.k+1,len(X)-1))

        if (self.ver == 'ML1'):
            return self._MiND_ML1(nbh_data)

        rhos = nbh_data[:,0]/nbh_data[:,-1]

        d_MIND_MLi = self._MiND_MLi(rhos)
        if (self.ver == 'MLi'):
            return(d_MIND_MLi)

        d_MIND_MLk = self._MiND_MLk(rhos, d_MIND_MLi)
        if (self.ver == 'MLk'):
            return(d_MIND_MLk)
        else:
            raise ValueError("Unknown version: ", self.ver)
                              
    @staticmethod
    def _MiND_ML1(nbh_data):
        n = len(nbh_data)
        dists2 = nbh_data[:,:2]**2  # need only squared dists to first 2 neighbors
        s = np.sum(np.log(dists2[:,0]/dists2[:,1]))
        ID = -2/(s/n)
        return ID

    def _MiND_MLi(self, rhos):
        #MIND MLI MLK REVERSED COMPARED TO R TO CORRESPOND TO PAPER
        N = len(rhos)
        d_lik = np.array([np.nan]*self.D)
        for d in range(self.D):
            d_lik[d] = lld(d, rhos, self.k, N)
        return(np.argmax(d_lik))

    def _MiND_MLk(self, rhos, dinit):
        #MIND MLI MLK REVERSED COMPARED TO R TO CORRESPOND TO PAPER
        res = minimize(fun=nlld,
                x0=np.array([dinit]),
                jac=nlld_gr,
                args=(rhos, self.k, len(rhos)),
                method = 'L-BFGS-B',
                bounds=[(0,self.D)])

        return(res['x'])  