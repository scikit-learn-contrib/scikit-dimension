from .._commonfuncs import FlexNbhdEstimator
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import DistanceMetric
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np

class GeoMle(FlexNbhdEstimator):
    def __init__(self, average_steps = 2, bootstrap_num = 20, alpha = 5e-3, interpolation_degree = 2):
        """
        Parameters
        ----------
        average_steps : int, optional
            Number of average steps. The default is 2. Estimator considers the k (number of neighbors) between k1 and k1 + average_steps - 1 (including).
        bootstrap_num : int, optional
            Number of bootstrap sets. The default is 20.
        alpha : float, optional
            Regularization parameter for Ridge regression. The default is 5e-3.
        interpolation_degree : int, optional
            Degree of interpolation polynomial. The default is 2.
        """
        self.average_steps = average_steps
        self.alpha = alpha
        self.max_degree = interpolation_degree
        self.bootstrap_num = bootstrap_num

    def _fit(self, X, nbhd_indices, nbhd_type, metric, radial_dists, radius = 1.0, n_neighbors = 5):
        # Check if the parameters are valid
        if self.average_steps < 0:
            raise ValueError("Number of average steps can not be negative")
        if self.bootstrap_num <= 0:
            raise ValueError("Number of bootstrap sets has to  be positive")
        if self.max_degree <= 0:
            raise ValueError("Degree of interpolation polynomial has to be positive")
        if nbhd_type not in ['knn']:
            raise ValueError('Neighbourhood type should be knn')
        
        k2 = n_neighbors + self.average_steps - 1
        self.dimension_pw_ = self.geomle(X, n_neighbors, k2, metric=metric)
    
    def geomle(self, X, k1, k2, metric='euclidean'):
        """
        Returns range of Levina-Bickel dimensionality estimation for k = k1..k2 (k1 < k2) averaged over bootstrap samples
    
        Input parameters:
        X            - data
        k1           - minimal number of nearest neighbours
        k2           - maximal number of nearest neighbours
        max_degree   - maximal degree of polynomial regression (Default = 2)
        metric       - metric for the distance calculation (Default = 'euclidean')
    
        Returns: 
        array of shape (nb_iter1,) of regression dimensionality estimation for k = k1..k2 averaged over bootstrap samples
        """
        if metric == 'precomputed':
            dist = X
            DIM_SPACE = None
        else:
            DIM_SPACE = X.shape[1]
            dist = pairwise_distances(X, X, metric=metric)
        NUM_OF_POINTS = X.shape[0]

        result = []
        data_reg = []
        avg_distances = np.zeros((NUM_OF_POINTS , self.average_steps))
        mles = np.zeros((NUM_OF_POINTS, self.bootstrap_num, self.average_steps))
        final_mles = np.zeros(NUM_OF_POINTS)
        for j in range(self.bootstrap_num):
            bootstrap_ids = GeoMle.__gen_bootstrap_ids(X)
            # Calculate distances to nearest neighbours in bootsrap dataset
            for x_id in range(X.shape[0]):
                bootstrap_distances_nn = np.sort(GeoMle._get_nearest_distances_in_bootstrap(x_id, dist, bootstrap_ids, k2))
                # update average distances
                avg_distances[x_id, :] += bootstrap_distances_nn[k1 - 1:]
                for k_iter in range(self.average_steps):
                    # Calculate MLE for bootstrap sample
                    mles[x_id, j, k_iter] = GeoMle._calc_mle(bootstrap_distances_nn[:k_iter + k1])
                    #print("Estims", mles[x_id, j, k_iter])

        for x_id in range(X.shape[0]):
            avg_distances[x_id, :] /= self.bootstrap_num # average distances over bootstrap samples
            mle_means = [mles[x_id,:,k].mean() for k in range(self.average_steps)] # mean MLE over bootstrap samples
            mle_std_variations = [mles[x_id,:,k].std(ddof=1) for k in range(self.average_steps)] # std MLE for specifik k over bootstrap samples
            final_mles[x_id] = max(self._calc_estimate_from_regression(mle_means, mle_std_variations, avg_distances[x_id, :]),
                                    0)
            if DIM_SPACE is not None:
                final_mles[x_id] = min(final_mles[x_id], DIM_SPACE)
        return final_mles
    
    @staticmethod
    def __gen_bootstrap_ids(data):
        return np.unique(np.random.randint(0, len(data) - 1, size=len(data)))
    
    def _calc_estimate_from_regression(self, mle_means, mle_std_variations, avg_distances):
        X = np.zeros((self.average_steps, self.max_degree))
        weights = [st ** -1 for st in mle_std_variations]
        for k_iter in range(self.average_steps):
            X[k_iter] = [avg_distances[k_iter]**i for i in range(1, self.max_degree + 1)]
        ridge_reg = Ridge(alpha=self.alpha, fit_intercept=True)
        ridge_reg.fit(X, mle_means, weights)
        return ridge_reg.intercept_

    @staticmethod
    def _get_nearest_distances_in_bootstrap(point_id, original_distance_matrix, bootstrap_ids, k):
        """
        Returns k nearest distances to the point_id in the bootstrap sample. The biggest distance is on last position.
        ------
        point_id: int
            index of the point in the original data
        original_distance_matrix: np.array
            distance matrix of the original data
        bootstrap_ids: np.array
            indices of the points in the bootstrap sample
        k: int
            number of nearest distances to return
        """
        if point_id in bootstrap_ids:
            bootstrap_distances = np.zeros(len(bootstrap_ids) - 1)
            iterate = 0
            for neighbor_id in bootstrap_ids:
                if not neighbor_id == point_id:
                    bootstrap_distances[iterate] =  original_distance_matrix[point_id][neighbor_id]
                    iterate = iterate + 1
        else:
            bootstrap_distances = np.zeros(len(bootstrap_ids))
            for i in range(len(bootstrap_ids)):
                bootstrap_distances[i] =  original_distance_matrix[point_id][bootstrap_ids[i]]

            
        return bootstrap_distances[np.argpartition(bootstrap_distances, k)[:k]]
    
    @staticmethod
    def _calc_mle(dlist):
        N = len(dlist)
        if N > 0:
            radius = max(dlist)
            minv = N*np.log(radius)-np.sum(np.log(dlist))

            if minv < 1e-9:
                raise Exception("MLE estimation diverges.")
            else:   
                return np.divide(N-2,minv) # We use N-2 (instead N-1) to get rid of the asymptotic bias
        else:
            return np.nan
