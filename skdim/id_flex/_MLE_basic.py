import inspect
import numpy as np
import scipy
from .._commonfuncs import FlexNbhdEstimator
from sklearn.metrics import DistanceMetric

class MLE_basic(FlexNbhdEstimator):
    '''
    Basic Levina-Bickel where we always have either knn or epsilon neighbourhood

    Parameters
    ----------
    average_steps : Number of different values of k (number of neighbours) used 
    '''
        

    def __init__(self, average_steps = 0):
        self.average_steps = average_steps #to do: integrate averaging over k neighbourhoods

    def _fit(self, X, nbhd_indices, nbhd_type, metric, radial_dists, **kwargs):

        if nbhd_type == 'eps':  
            if 'eps' in kwargs:
                eps = kwargs['eps']
            else:
                eps = 1.0 #default sklearn
        else: #knn
            eps = None
        
        self.dimension_pw_ = np.array([self._mle_formula(dlist, nbhd_type, eps) for dlist in radial_dists])
        
    @staticmethod
    def _mle_formula(dlist, nbhd_type, eps):

        N = len(dlist)
        
        if nbhd_type == 'knn':
            eps = np.max(dlist)
        
        minv = N*np.log(eps) - np.sum(np.log(dlist))

        if minv < 1e-9:
            return np.nan
        else:
            if nbhd_type == 'knn': 
                return np.divide(N-1,minv)
            else: #eps
                return np.divide(N, minv)


        
    
