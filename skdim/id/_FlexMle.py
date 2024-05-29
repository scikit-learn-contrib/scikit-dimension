import numpy as np

from .._commonfuncs import lens, get_nn, LocalEstimator

class FlexMLE(FlexNbhdEstimator):
    """
    Classic Maximum Likelihood proposed by Levina and Bickell
    """
    def _fit(self, X, nbhd_dict, metric):
        """
        Custom method to each local ID estimator, called in fit
        
        """
        self.dimension_pw_ = np.empty(len(X))
        self.__distance_metric = DistanceMetric.get_metric(metric)
        # TODO: Add aggregation over different number of k
        for poind_index in range(len(X)):
            self.dimension_pw[poind_index] = self.__estimate_at_point(X, nbhd_dict, metric)
        self.is_fitted_pw_ = True
        return self

    def __get_distances(self, X, point_id, nbhd_dict, metric):
        if metric == "precomputed":
            return [X[point_id, neighbor_id] for neighbor_id in nbhd_dict[point_id]]
        else:
            return [self.__distance_metric(X[point_id] , X[nbhd_dict]) for neighbor_id in nbhd_dict[point_id]]
    
    def __estimate_at_point(self, X, point_id, nbhd_dict, metric):
        distances = self.__get_distances(X, point_id, nbhd_dict, metric)
        max_dist = max(distances)
        return (len(distances) - 1) / sum(np.log(distances[i] / max_dist))