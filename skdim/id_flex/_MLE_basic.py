import numpy as np
from .._commonfuncs import FlexNbhdEstimator
from ..errors import EstimatorFailure


class MLE_basic(FlexNbhdEstimator):
    '''
    Basic Levina-Bickel where we always have either knn or epsilon neighbourhood

    Parameters
    ----------
    average_steps : Number of different values of k (number of neighbours) used 
    '''
        

    def __init__(self, nbhd_type = 'knn', metric = 'euclidean', comb = 'mean', smooth = False, n_jobs = 1, radius = 1.0, n_neighbors = 5):
        super().__init__(nbhd_type = nbhd_type, pt_nbhd_incl_pt = False, metric = metric, comb = comb, smooth = smooth, n_jobs = n_jobs, radius = radius, n_neighbors = n_neighbors)
        
        if self.nbhd_type not in ['knn', 'eps']:
            raise ValueError(
                    "Invalid nbhd_type parameter. It has to be 'knn' or 'eps' for Levina-Bickel."
                )
        
        #self.average_steps = average_steps #to do: integrate averaging over k neighbourhoods

    def _fit(self, X,  nbhd_indices, radial_dists):

        self.dimension_pw_ = np.array([self._mle_formula(dlist) for dlist in radial_dists])
    
    def _mle_formula(self, dlist):

        N = len(dlist)
        
        if N > 0:
            if self.nbhd_type != 'eps': # defaults to knn
                radius = np.max(dlist)
            else:
                radius = self.radius
            
            minv = N*np.log(radius)-np.sum(np.log(dlist))

            if minv < 1e-9:
                raise EstimatorFailure("MLE estimation diverges.")
            else:
                if self.nbhd_type == 'eps':
                    return np.divide(N, minv)
                else: #knn
                    return np.divide(N-1,minv)
        else:
            return np.nan
                


        
    
