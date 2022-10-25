import numpy as np
import skdim
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from . import grassmann_distances, embedding, moments

class Subspaces:
    ''' Class to store and compute distances between subspaces (e.g., principal axes)
        
    Parameters
    ----------
    n: int
        embedding dimension
        
    Attributes
    ----------
    n_points_: int
        number of points stored in the class instance
    ranks_: list
        rank of each subspace
    unique_ranks_: np.ndarray
        unique ranks of subspaces
    subspaces_: list[np.ndarray]
        list of ordered orthonormal bases stored as column vectors of shape (n x ranks[i]). 
    sigmas_: list[np.ndarray]
        eigenvalues of each subspace (if obtained from PCA or provided)
    means_: list[np.ndarray]
        means of subspaces (if obtained from PCA or provided)

    '''
    
    def __init__(self,n):
        self.n = n
        self.n_points_ = 0
        self.unique_ranks_ = np.array([])
        self.ranks_, self.subspaces_, self.sigmas_, self.means_ = [], [], [], []
        
    def __repr__(self):
        return (f"Subspaces(n={self.n}, n_points_={self.n_points_}, unique_ranks_={self.unique_ranks_})")
    
    @classmethod
    def from_projection(cls,data,ranks='auto'):
        '''Initialize by projection (centering + SVD of each array in data)
        
        Parameters
        ----------
        data: list[np.ndarray]
            Arrays to project (n_points[i] x self.n)
        ranks: str or int or iterable[int] 
            Method to determine the retained rank:
                if 'auto': automatically determine the rank
                if an integer: all arrays are projected to rank=ranks
                if an iterable: data[i] is projected to rank=ranks[i]
        '''
        return cls(data[0].shape[1]).add(data,project=True,ranks=ranks)
    
    @classmethod
    def from_subspaces(cls,data,sigmas=None,rtol=1e-05, atol=1e-07):
        '''Initialize by projection (centering + SVD of each array in data)
        
        Parameters
        ----------
        data: list[np.ndarray]
            Arrays with orthonormal columns to add 
        sigmas: list[np.ndarray] 
            Variance explained by each component of the array 
        rtol: float
            Relative tolerance on the orthonormal requirement (see np.allclose doc)
        atol: float
            Absolute tolerance on the orthonormal requirement (see np.allclose doc)
        '''
        return cls(data[0].shape[0]).add(data,project=False,sigmas=sigmas,rtol=rtol, atol=atol)
    
    def add(self,data,project=True,ranks='auto',sigmas=None,rtol=1e-05, atol=1e-07):
        '''Add new arrays to project or new subspaces
        
        data: list[np.ndarray]
            Arrays with orthonormal columns to add
        project: bool
            If True, each array is projected to a subspace by centering + SVD
            If False, arrays are assumed to be subspaces already
        ranks: str or int or iterable[int] 
            Method to determine the retained rank:
                if 'auto': automatically determine the rank
                if an integer: all arrays are projected to rank=ranks
                if an iterable: data[i] is projected to rank=ranks[i]
        sigmas: list[np.ndarray] 
            Variance explained by each component of the array     
        '''
        self._check_shape(data,project)
        
        if project:
            self._add_projections(data,ranks=ranks)
        else:
            self._add_subspaces(data,sigmas=sigmas,rtol=rtol, atol=atol)
            
        self._store_attributes()
        return self
    
    
    def embed_conway(self,normalize=False,max_rank=None):
        ''' Spherical embedding of subspaces.
        
        Isometric embedding of rank k-dimensional subspaces in R^n as points
        on a sphere in dimension (n − 1)(n + 2)/2. Chordal distance becomes euclidean distance
        and geodesic distance becomes spherical distance

        "Packing Lines, Planes, etc.: Packings in Grassmannian Spaces", 
        J. H. Conway, R. H. Hardin and N. J. A. Sloane
        
        Parameters
        ----------
        normalize: bool
            If True, return sphere normalized to radius=1 
        max_rank: int
            Clip subspaces rank to max_rank for computation before embedding
        
        Returns
        -------
        self.subspaces_conway_: np.array
            Conway embedding of subspaces, centered at zero
        self.embed_conway_max_rank_: int
            Stores max_rank parameter
        self.embed_conway_orig_center:
            Theoretical (original) center of the sphere
        '''
        if max_rank is not None:
            subs = [np.ascontiguousarray(s[:,:max_rank]) for s in self.subspaces_]
        else:
            subs = self.subspaces_

        projs, orig_center = embedding.embed_conway_loop(subs,normalize=normalize)

        self.subspaces_conway_ = projs
        self.embed_conway_max_rank = max_rank
        self.embed_conway_normalize = normalize
        self.embed_conway_orig_center_ = orig_center
 
    def grassmann_distances(self,Subspaces=None,metric='geodesic',contrast_unequal=True,
                            gpu=True,max_nbytes = 1e8,max_rank=None,return_angles=True):
        ''' Compute Grassmannian distance matrix 
        from principal angles or Conway's spherical embedding (see Subspaces.embed_conway)
        
        Parameters
        ----------
        Subspaces: subspace.Subspaces 
            Another Subspaces instance to compute distance matrix against 
            (with shape self.n_points_ x Subspaces.n_points_).
            If provided, outputs will be directly returned
        metric: str
            Which Grassmannian metric to use:
              Distances from principal angles:
                'asimov', 'binet_cauchy','chordal','fubini_study',
                'geodesic','martin','procrutes','projection','spectral'
              Distances from Conway embedding:
                'euclidean_conway': euclidean distance (~= proportional to 'chordal')
                'geodesic_conway_abs': euclidean distance, flipping vectors with negative cosine similarity
                'chordal_conway': euclidean distance with scaled off-diagonal terms (='chordal')
                'geodesic_conway': distance on the surface of Conway sphere (~= proportional to 'geodesic')
                'geodesic_conway_abs': distance on the surface of Conway sphere, flipping vectors with negative cosine similarity
            
        contrast_unequal: bool
            Whether to contrast the distance between subspaces of different dimensions 
            as proposed in 
            "Schubert varieties and distances between subspaces of different dimensions", 
            Ke Ye & Lek-Heng Lim. SIAM J. MATRIX ANAL. APPL. 2016 Vol. 37, No. 3, pp. 1176–1197
        gpu: bool
            If True, computations are done on GPU using jax.numpy, otherwise numpy
        max_nbytes:
            Maximum block size, in bytes, to compute at once
        max_rank: int
            Clip subspaces rank to max_rank for computation of distances
        return_angles:
            Whether to return principal angles
            
        Returns
        -------
        self.pdist_: np.ndarray
            Grassmann distance matrix, in pdist format 
            (use scipy.spatial.distance.squareform to get square format)
        '''
        #compute X,Y else X,X distance matrix
        if Subspaces is None:
            self_, ij = self, None
        else:
            self_ = self.from_subspaces(self.subspaces_ + Subspaces.subspaces_,rtol=10,atol=10)
            n, m = self.n_points_, Subspaces.n_points_
            i, j = np.ones((n,m)).nonzero()
            ij = (i, j + n) # compute only last m columns of concatenated subspace list
            
        # conway embedding distances
        if 'conway' in metric:
            if not contrast_unequal:
                raise ValueError('contrast_unequal=False not supported when using Conway embedding.')
    
            if not hasattr(self_,'subspaces_conway_') or self_.embed_conway_max_rank != max_rank:
                self_.embed_conway(max_rank=max_rank)
    
            self_.pdist_ = grassmann_distances.conway_distance_matrix(self_.subspaces_conway_,ij=ij,metric=metric,gpu=gpu,max_nbytes=max_nbytes)
    
        # principal angle distances
        else:
            kwargs = dict(subspaces=self_.subspaces_,ij=ij,metric=metric,
                          contrast_unequal=contrast_unequal,
                          gpu=gpu,max_nbytes = max_nbytes,max_rank=max_rank,
                          return_angles=return_angles)
    
            if not(hasattr(self_,'thetas_')) or len(self_.thetas_) != len(self_.subspaces_):
                func = grassmann_distances.grassmann_distance_matrix
            else:
                func = grassmann_distances.grassmann_distance_matrix_precomputed_angles
                kwargs['thetas'] = self_.thetas_
                return_angles = False

            if return_angles:
                self_.pdist_, self_.thetas_ = func(**kwargs)
            else:
                self_.pdist_ = func(**kwargs)
    
        if Subspaces is not None:
            if return_angles: 
                return self_.pdist_.reshape(n,m), self_.thetas_
            else:
                return self_.pdist_.reshape(n,m)
        
        self_.grassmann_distances_metric=metric
        self_.grassmann_distances_max_rank=max_rank
        self_.grassmann_distances_contrast_unequal=contrast_unequal

    def grassmann_knn(self,n_neighbors=5,algorithm='auto',leaf_size=30,n_jobs=1,
                      metric='euclidean_conway',max_rank=None):

        ''' 
        Compute Grassmannian kNN from Conway's spherical embedding (see Subspaces.embed_conway)
        
        Parameters
        ----------
        metric: str
            Which Grassmannian metric to use from 'chordal_conway', 'geodesic_conway'
        max_rank: int
            Clip subspaces rank to max_rank for computation of distances
        Other parameters are passed to sklearn.neighbors.NearestNeighbors

        Returns
        -------
        self.grassmann_knndis_
        self.grassmann_knnidx_
        self.grassmann_knn_metric
        self.grassmann_knn_max_rank
        
        '''
        
        if metric not in ['euclidean_conway','chordal_conway','spherical_conway','geodesic_conway']:
            raise ValueError("Metric must be on the Conway sphere. See self.grassmann_distances ")
        if not hasattr(self,'subspaces_conway_') or self.embed_conway_max_rank != max_rank:
            self.embed_conway(max_rank)
    
        if metric != 'euclidean_conway':
            dist_fun = grassmann_distances.DISTANCE_FUNCTIONS[metric]
        else:
            dist_fun = 'minkowski'
            
        self.grassmann_knndis_, self.grassmann_knnidx_ = NearestNeighbors(
            metric=dist_fun,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            leaf_size=leaf_size,
            n_jobs=n_jobs,).fit(self.subspaces_conway_).kneighbors()
        
        self.grassmann_knn_metric = metric
        self.grassmann_knn_max_rank = max_rank
        
    def mean(self,kind='flag'):
        ''' 
        Compute mean of subspaces. 
        Subspaces need to all be of same rank except for kind='flag'
        
        Parameters
        ----------
        kind: str
            Which mean formula to apply
        Returns
        -------
        self.mean_: np.array
            Mean subspace basis
        self.svals_: np.array
            Singular values associated with the mean
        
        '''  
        if kind != 'flag':
            raise ValueError('Only flag mean is currently implemented')
        self.mean_, self.mean_svals_ = moments.flag_mean(self.subspaces_)
        
    def median(self,kind='irls',rank=None,n_its=100):
        ''' 
        Compute median of subspaces. 
        Subspaces need to all be of same rank except for kind='irls'
        
        Mankovich, Nathan, et al. "The Flag Median and FlagIRLS.", 2022.
        Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 
        
        Parameters
        ----------
        kind: str
            Which mean formula to apply
        rank: int
            Rank of median subspace to compute. Default is max(self.ranks)
        n_its: int
            Number of iterations
            
        Returns
        -------
        self.median_: np.array
            Mean subspace basis
        self.median_err_: list
            Objective function error
        
        '''  
        if kind != 'irls':
            raise ValueError('Only flag irls median is currently implemented')
        if rank is None:
            rank = max(self.ranks_)
        self.median_, self.median_err_ = moments.irls_flag(self.subspaces_,r=rank,n_its=n_its)
        
    def flatten(self):
        '''
        Returns a copy where basis vectors 
        for each subspace are considered as different 1-d subspaces.
        '''
        data = [np.ascontiguousarray(v.reshape(-1,1)) for s in self.subspaces_ for v in s.T]
        return self.from_subspaces(data=data)

    def flip_sign(self):
        ''' 
        Harmonize basis vector directions 
        by multiplying with the sign of the first coordinate
        '''
        for s in self.subspaces_:
            s*=np.sign(s[[0]])
        
    def _check_shape(self,data,project):
        if project and len(np.unique([d.shape[1] for d in data])) != 1:
            raise ValueError('embedding dimension (data[i].shape[1]) '
                             'must be equal for all arrays in data')
        elif not project and len(np.unique([d.shape[0] for d in data])) != 1:
            raise ValueError('embedding dimension (data[i].shape[0]) '
                             'must be equal for all arrays in data')
            
    def _check_point(self,array,rtol=1e-05, atol=1e-07):
         if not (array.shape[0]==self.n and array.ndim == 2 and ('float' in array.dtype.name) and
                     np.allclose(array.T @ array, np.eye(array.shape[1]),rtol=rtol,atol=atol)):
            raise ValueError('point(s) do not belong to the manifold')
    
    def _add_projections(self,data,ranks):
        
        if isinstance(ranks, int):
            ranks = np.repeat(ranks,len(data))
        elif isinstance(ranks, str) and ranks == "auto":
            ranks = np.repeat(None,len(data))
        elif len(ranks) == len(data):
            pass
        else:
            raise ValueError("The input parameter ranks must be either"
                             "'auto', an integer, or an array of integers with same length as data")  
        
        for X,rank in zip(data,ranks):
            pca = PCA(n_components=rank).fit(X)
            if rank is None:
                rank = skdim.id.lPCA(alphaFO=.1,fit_explained_variance=True).fit_transform(pca.explained_variance_)
            self.subspaces_.append(np.ascontiguousarray(pca.components_[:rank].T))
            self.sigmas_.append(pca.explained_variance_)
            self.ranks_.append(rank)   
            self.means_.append(pca.mean_)
                        
    def _add_subspaces(self,subspaces,sigmas=None,means=None,rtol=1e-05, atol=1e-07):
        if sigmas is None:
            sigmas = np.repeat(None,len(subspaces))
        if means is None:
            means = np.repeat(None,len(subspaces))
        for X,sigma,mean in zip(subspaces,sigmas,means):
            self._check_point(X,rtol=rtol,atol=atol)
            self.subspaces_.append(X)
            self.sigmas_.append(sigma)
            self.ranks_.append(X.shape[1])
            self.means_.append(mean)

    def _store_attributes(self):
        self.n_points_ = len(self.subspaces_)            
        self.ranks_ = [s.shape[1] for s in self.subspaces_]
        self.unique_ranks_ = np.unique(self.ranks_)
        self.max_rank_ = max(self.ranks_)     