import numpy as np
import random
from functools import reduce
from itertools import combinations  
from collections.abc import Iterable


from joblib import effective_n_jobs, Parallel, delayed
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_array, coo_array, dok_array
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.cluster.hierarchy import DisjointSet

from sklearn.utils.validation import check_array
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import kneighbors_graph

from .._commonfuncs import GlobalEstimator

class PH(GlobalEstimator):
    """Intrinsic dimension estimation using the PHdim algorithm. 
        The final step of the algorithm involves fitting a straight line to data. 
        User should plot self.x_ against self.y_ to verify goodness of fit, and if needs be fit the straight line to a subset of x_ vs y_ to improve the inferrence. 
        
    Parameters
    ----------  
    alpha: float
        Persistence power
    n_range: 2-tuple
        Min and Max sizes of subsamples. If range_type = 'fraction', then n_range is the min and max fractions of the number of points; if 'num', then n_range is the min and max number
    range_type: str
        Specifies whether n_range describes fraction
    nsteps: int
        number of regression subsample sizes
    subsamples: int
        Number of random subsamples per size of subsample
    metric: str
        scipy.spatial.distance metric parameter
    seed: int
        random seed for subsampling

    Attributes
    ----------
    x_: 1d array 
        np.array with the log(n) values. 
    y_: 1d array 
        np.array with the log(E) values. 
    reg_: sklearn.linear_model.LinearRegression
        regression object used to fit line to log E vs log n
    """
    def __init__(self,  alpha = 1.0, n_range_min = 0.5, n_range_max = 1, range_type = 'fraction', nsteps = 100, subsamples = 10, metric = 'euclidean', random_state =12345,  n_jobs = 1):
        self.alpha = alpha
        self.n_range_min = n_range_min
        self.n_range_max = n_range_max
        self.range_type = range_type
        self.nsteps = nsteps
        self.subsamples = subsamples
        self.metric = metric 
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self: object
            Returns self.
        self.dimension_: float
            The estimated intrinsic dimension
        self.reg_: object
            sklearn LinearRegression object
        """
        if self.range_type == 'fraction':
            self.nmin = int(np.ceil(self.n_range_min * X.shape[0]))
            self.nmax = int(np.ceil(self.n_range_max * X.shape[0]))
        elif self.range_type == 'num':
            self.nmin, self.nmax = self.n_range_min, self.n_range_max
        else:
            raise ValueError("range_type should either be 'fraction', or 'num'.")
        
        
        self.subsamplerange = np.ceil(np.linspace(self.nmin,self.nmax, self.nsteps)).astype(int)
        self._check_params(X)

        X = check_array(X, ensure_min_samples=self.subsamplerange[-1], ensure_min_features=2)

        self.dimension_ = self._phEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self


    def _phEst(self, X):

        random.seed(self.random_state)
        n = X.shape[0]
        edges, sort_idx = self._sort_distances(X)

        E = self._ph(n, edges, sort_idx)

        x = np.array([nss for nss in self.subsamplerange for _ in range(self.subsamples)]).reshape([-1,1])
        #x = np.repeat(self.subsamplerange, self.subsamples).reshape([-1,1])

        self.x_ = np.log(x)
        self.y_ = np.log(E)

        ###check the array propagation in this bit!
        if isinstance(self.alpha, float) or isinstance(self.alpha, int):
            reg = LinearRegression(fit_intercept = True).fit(self.x_, self.y_.reshape([-1]))
            dim = np.divide(self.alpha,(1-reg.coef_[0]))
        elif isinstance(self.alpha, Iterable):
            dim = []
            for k in range(len(self.alpha)):
                reg = LinearRegression(fit_intercept = True).fit(self.x_, self.y_[:,k])
                dim.append(np.divide(self.alpha[k],(1-reg.coef_[0])))
        
        self.reg_ = reg

        return dim
    
    def _ph(self, num_points, distances, sort_idx):
        flatten_double_loop = [nss for nss in self.subsamplerange for _ in range(self.subsamples)]
        if isinstance(self.n_jobs, int) and self.n_jobs != 0:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                total_persistence = parallel(
                    delayed(self._ph_subsample)(num_points, nss, distances, sort_idx)
                    for nss in flatten_double_loop
                )
        else:
            total_persistence = [self._ph_subsample(num_points, nss, distances, sort_idx)for nss in self.subsamplerange for _ in range(self.subsamples)]


        return np.array(total_persistence)


    def _check_params(self, X):
        if isinstance(self.alpha, list) or isinstance(self.alpha, tuple):
            if np.any(np.array(self.alpha) <= 0):
                raise ValueError("Alpha power parameter must be a strictly positive.")
        elif isinstance(self.alpha, float) or isinstance(self.alpha, int):
            if self.alpha <= 0:
                raise ValueError("Alpha power parameter must be a strictly positive.")
        if self.range_type == 'num':
            if self.n_range_min <= 1  or not isinstance(self.nmin, int):
                raise ValueError("Min subsample population size must be an integer > 1.")
            if self.n_range_max < self.n_range_min or not isinstance(self.nmax, int):
                raise ValueError("Max subsample population size must be an integer greater than than the min subsample population size.")
        elif self.range_type =='fraction':
            if self.n_range_max < self.n_range_min :
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
            if (self.n_range_max > 1) or (self.n_range_max <= 0):
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
            if (self.n_range_min > 1) or (self.n_range_min <= 0):
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
        if self.nsteps < 2 or not isinstance(self.nsteps, int):
            raise ValueError("Nsteps must be an integer >= 2.")
        if self.subsamples < 1  or not isinstance(self.subsamples, int):
            raise ValueError("Min number of subsamples must be an integer >= 1.")
        
        if self.nmin > X.shape[0]:
            raise ValueError("Minimum subsample population size greater than number of points.")
        if self.nmin <= 1:
            raise ValueError("Minimum subsample population should be greater than one.")
        if self.nmax > X.shape[0]:
            raise ValueError("Maximum subsample population size greater than number of points.")
        if self.nmax <= 1:
            raise ValueError("Maximum subsample population should be greater than one.")
        if self.nmin >= self.nmax:
            raise ValueError("Maximum subsample population should be greater than minimum subsample population.")
        if len(self.subsamplerange) < 2:
            raise ValueError("Subsample population range has fewer than two points, modify range of N or nstep to ensure there is a line to be fitted!")
    

    def _ph_subsample(self,num_points, num_subsamples, distances, sort_idx):
        subsample_indices = random.sample(range(num_points),num_subsamples) #randomly choose subsamples
        edge_filt = self._subsample_filter_edges(num_points, subsample_indices, sort_idx)
        filtered_edges = [distances[e] for e in edge_filt]
        return self._Krukskal(subsample_indices, filtered_edges, self.alpha)
    
    @staticmethod
    def _sort_distances(X):
        Ds = pdist(X)
        filt_idx = np.triu_indices(n = X.shape[0], k = 1)
        sort_idx = np.argsort(Ds)
        return list(zip(filt_idx[0], filt_idx[1], Ds)), sort_idx
    
    @staticmethod
    def _subsample_filter_edges(n, subsample_indices, edge_order):
        filt =[False for _ in range(n*(n-1)//2)] # records whether edge of a certian index is included
        for u,v in combinations(subsample_indices, 2): #unique pairs:
            i,j = sorted([u,v]) # make sure u < v
            k = j-1 + i*(n-1) - i*(i+1) //2 #index in edge list
            filt[k] = True
        return [k for k in edge_order if filt[k]]

    @staticmethod 
    def _Krukskal(vertices, edges, alpha = 1.0):
        #assume edges sorted
        total_persistence = np.zeros_like(alpha).astype(float)
        djs = DisjointSet(vertices)
        for u,v, weight in edges:
            if djs[u] != djs[v]: # if roots distinct
                total_persistence += np.power(weight, alpha)
                djs.merge(djs[u],  djs[v]) # repoint roots
        return total_persistence
    


class PH_knn(GlobalEstimator):
    """Intrinsic dimension estimation using the PHdim algorithm. 
        The final step of the algorithm involves fitting a straight line to data. 
        User should plot self.x_ against self.y_ to verify goodness of fit, and if needs be fit the straight line to a subset of x_ vs y_ to improve the inferrence. 
        
    Parameters
    ----------  
    alpha: float
        Persistence power
    n_range: 2-tuple
        Min and Max sizes of subsamples. If range_type = 'fraction', then n_range is the min and max fractions of the number of points; if 'num', then n_range is the min and max number
    range_type: str
        Specifies whether n_range describes fraction
    nsteps: int
        number of regression subsample sizes
    subsamples: int
        Number of random subsamples per size of subsample
    metric: str
        scipy.spatial.distance metric parameter
    seed: int
        random seed for subsampling

    Attributes
    ----------
    x_: 1d array 
        np.array with the log(n) values. 
    y_: 1d array 
        np.array with the log(E) values. 
    reg_: sklearn.linear_model.LinearRegression
        regression object used to fit line to log E vs log n
    """
    def __init__(self,  alpha = 1.0, n_range_min = 0.5, n_range_max = 1, range_type = 'fraction', nsteps = 10, subsamples = 10, metric = 'euclidean',  n_neighbors = 10, random_state =12345,  n_jobs = 1):
        self.alpha = alpha
        self.n_range_min = n_range_min
        self.n_range_max = n_range_max
        self.range_type = range_type
        self.nsteps = nsteps
        self.subsamples = subsamples
        self.metric = metric 
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self: object
            Returns self.
        self.dimension_: float
            The estimated intrinsic dimension
        self.reg_: object
            sklearn LinearRegression object
        """
        if self.range_type == 'fraction':
            self.nmin = int(np.ceil(self.n_range_min * X.shape[0]))
            self.nmax = int(np.ceil(self.n_range_max * X.shape[0]))
        elif self.range_type == 'num':
            self.nmin, self.nmax = self.n_range_min, self.n_range_max
        else:
            raise ValueError("range_type should either be 'fraction', or 'num'.")
        
        
        self.subsamplerange = np.ceil(np.linspace(self.nmin,self.nmax, self.nsteps)).astype(int)
        self._check_params(X)

        X = check_array(X, ensure_min_samples=self.subsamplerange[-1], ensure_min_features=2)

        self.dimension_ = self._phEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self


    def _phEst(self, X):

        random.seed(self.random_state)
        n = X.shape[0]
        edges = self._sort_distances(X, n_neighbors=self.n_neighbors, metric = self.metric, n_jobs = self.n_jobs)

        self.E_ = self._ph(n, edges)  # subsamples x subsample range x alphas
        self.x_ = np.log(self.subsamplerange).reshape(-1, 1)
        self.y_ = np.log(np.mean(self.E_, axis = 1)) # subsample range x alphas

        ###check the array propagation in this bit!
        if isinstance(self.alpha, float) or isinstance(self.alpha, int):
            reg = LinearRegression(fit_intercept = True).fit(self.x_, self.y_.reshape([-1,1]))
            dim = np.divide(self.alpha,(1-reg.coef_[0]))
        elif isinstance(self.alpha, Iterable):
            dim = []
            for k,a in enumerate(self.alpha):
                reg = LinearRegression(fit_intercept = True).fit(self.x_, self.y_[:,k].reshape([-1,1]))
                dim.append(np.divide(a,(1-reg.coef_[0])))
        
        self.reg_ = reg

        return dim
    
    def _ph(self, num_points, edges):

        flatten_double_loop = [nss for nss in self.subsamplerange for _ in range(self.subsamples)]
        if isinstance(self.n_jobs, int) and self.n_jobs not in [0,1]:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                total_persistence = parallel(
                    delayed(self._ph_subsample)(num_points, nss, edges)
                    for nss in flatten_double_loop
                )
            return np.array(total_persistence).reshape([len(self.subsamplerange), self.subsamples,  -1]) #subsample range x subsamples x  alpha
        else:
            return np.array([self._ph_subsample(num_points, nss, edges)for nss in self.subsamplerange for _ in range(self.subsamples)]).reshape([len(self.subsamplerange), self.subsamples,  -1]) #subsample range x subsamples x  alpha




    def _check_params(self, X):
        if isinstance(self.alpha, list) or isinstance(self.alpha, tuple):
            if np.any(np.array(self.alpha) <= 0):
                raise ValueError("Alpha power parameter must be a strictly positive.")
        elif isinstance(self.alpha, float) or isinstance(self.alpha, int):
            if self.alpha <= 0:
                raise ValueError("Alpha power parameter must be a strictly positive.")
        if self.range_type == 'num':
            if self.n_range_min <= 1  or not isinstance(self.nmin, int):
                raise ValueError("Min subsample population size must be an integer > 1.")
            if self.n_range_max < self.n_range_min or not isinstance(self.nmax, int):
                raise ValueError("Max subsample population size must be an integer greater than than the min subsample population size.")
        elif self.range_type =='fraction':
            if self.n_range_max < self.n_range_min :
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
            if (self.n_range_max > 1) or (self.n_range_max <= 0):
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
            if (self.n_range_min > 1) or (self.n_range_min <= 0):
                raise ValueError("Max subsample population fraction must be in (0,1] greater than than the min subsample fraction.")
        if self.nsteps < 2 or not isinstance(self.nsteps, int):
            raise ValueError("Nsteps must be an integer >= 2.")
        if self.subsamples < 1  or not isinstance(self.subsamples, int):
            raise ValueError("Min number of subsamples must be an integer >= 1.")
        
        if self.nmin > X.shape[0]:
            raise ValueError("Minimum subsample population size greater than number of points.")
        if self.nmin <= 1:
            raise ValueError("Minimum subsample population should be greater than one.")
        if self.nmax > X.shape[0]:
            raise ValueError("Maximum subsample population size greater than number of points.")
        if self.nmax <= 1:
            raise ValueError("Maximum subsample population should be greater than one.")
        if self.nmin >= self.nmax:
            raise ValueError("Maximum subsample population should be greater than minimum subsample population.")
        if len(self.subsamplerange) < 2:
            raise ValueError("Subsample population range has fewer than two points, modify range of N or nstep to ensure there is a line to be fitted!")
    
        if self.n_neighbors <= 1  or not isinstance(self.n_neighbors, int):
            raise ValueError("Number of neighbors for knn graph needs to be an integer greater than 1. ")

    def _ph_subsample(self,num_points, num_subsamples, edges):
        row, col, dat = edges
        boo = np.array([True] * num_subsamples + [False] *(num_points - num_subsamples))
        boo = boo[np.random.permutation(num_points)]
        ss_vert = np.arange(num_points)[boo]
        boo_edge = [boo[r] and boo[col[j]] for j,r in enumerate(row)]
        ss_edge = zip(row[boo_edge], col[boo_edge], dat[boo_edge])

        return self._mstweight(ss_vert, ss_edge)
    
    def _mstweight(self, vertices, edges):
        #assume edges sorted
        total_persistence = np.zeros_like(self.alpha).astype(float)
        djs = DisjointSet(vertices)
        for u,v, weight in edges:
            if djs[u] != djs[v]: # if roots distinct
                total_persistence += np.power(weight, self.alpha)
                djs.merge(djs[u],  djs[v]) # repoint roots
        return total_persistence #shape = alpha
    
    @staticmethod
    def _sort_distances(X, n_neighbors = 10, metric = 'euclidean', n_jobs = 1):
        G = dok_array(kneighbors_graph(X,n_neighbors=n_neighbors, mode = 'distance', metric = metric, n_jobs = n_jobs))  #NB Not symmetric 
        n = X.shape[0]
        #flip all entries of asymmetric matrix above diagonal
        G_upper = dok_array((n,n))
        for i,j in G.keys():
            u,v = min(i,j), max(i,j)
            G_upper[(u,v)] = G[(i,j)] 
        #obtain edges indexed over u < v
        Gc = coo_array(G_upper)
        sort_ind = np.argsort(Gc.data)
        row, col, dat = Gc.row[sort_ind], Gc.col[sort_ind], Gc.data[sort_ind]
        return (row, col, dat)
    
