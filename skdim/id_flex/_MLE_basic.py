import inspect
import numpy as np
import scipy
from .._commonfuncs import FlexNbhdEstimator
from sklearn.metrics import DistanceMetric

class MLE_basic(FlexNbhdEstimator):

    def __init__(self):
        pass
    def _fit(self, **kwargs):
        self.dimension_pw_ = self._MLE(**kwargs)
        self.is_fitted_pw_ = True
    
    def _MLE(self, **kwargs):

        pw_dims = dict()

        if kwargs['nbhd_type'] == 'eps':
            if 'radius' in kwargs:
                R = kwargs['radius']
            else:
                R = 1.0 #sklearn default
        else: #defaults to knn
            R = None
        
        if kwargs['metric'] == 'precomputed': 
            D = kwargs['X']
        else:
            dist = DistanceMetric.get_metric(kwargs['metric'])
            X = kwargs['X']

        nbhd_dict =  kwargs['nbhd_dict']
        for pt in nbhd_dict: #need to change dlist processing to receive dlist from above
            if kwargs['metric'] == 'precomputed':
                dlist = D[pt][nbhd_dict[[pt]]]
            else:
                dlist = dist.pairwise(X[pt], X[nbhd_dict[[pt]]])
            pw_dims[pt] = self._mle_formula(dlist, R= R)
        
        return pw_dims


    @staticmethod
    def _mle_formula(dlist, R = None):
        N = len(dlist)
        if R is None:  #knn
            R = np.max(dlist)
        minv = N*np.log(R) - np.sum(np.log(dlist))
        if minv < 1e-9:
            return np.nan
        else:
            if R is None: #knn
                return np.divide(N-1,minv)
            else: #eps
                return np.divide(N, minv)


        
    
