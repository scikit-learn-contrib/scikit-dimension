#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
import numba as nb
import itertools
import numbers
import warnings
from scipy.sparse import coo_array
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.base import BaseEstimator
from joblib import delayed, Parallel
from abc import abstractmethod
from itertools import chain

def indComb(NN):
    pt1 = np.tile(range(NN), NN)
    pt2 = np.repeat(range(NN), NN)

    un = pt1 > pt2

    pt1 = pt1[un]
    pt2 = pt2[un]

    return pt1, pt2, np.hstack((pt2[:, None], pt1[:, None]))


@nb.njit
def indnComb(NN, n):
    if n == 1:
        return np.arange(NN).reshape((-1, 1))
    prev = indnComb(NN, n - 1)
    lastind = prev[:, -1]
    ind_cf1 = np.repeat(lastind, NN)
    # ind_cf2 = np.tile(np.arange(NN), len(lastind))
    ind_cf2 = (
        np.repeat(np.arange(NN), len(lastind)).reshape(-1, len(lastind)).T.flatten()
    )  # np.tile replacement

    new_ind = np.where(ind_cf1 < ind_cf2)[0]
    new_ind1 = (new_ind - 1) // NN
    new_ind2 = new_ind % NN
    new_ind2[new_ind2 == 0] = NN
    return np.hstack((prev[new_ind1, :], np.arange(NN)[new_ind2].reshape((-1, 1))))


def efficient_indnComb(n, k, random_generator_):
    """
    memory-efficient indnComb:
    uniformly takes 5000 samples from itertools.combinations(n,k)
    """
    ncomb = binom_coeff(n, k)
    pop = itertools.combinations(range(n), k)
    targets = set(random_generator_.choice(ncomb, min(ncomb, 5000), replace=False))
    return np.array(
        list(itertools.compress(pop, map(targets.__contains__, itertools.count())))
    )


def lens(vectors):
    return np.sqrt(np.sum(vectors**2, axis=1))


def proxy(tup):
    function, X, Dict = tup
    return function(X, **Dict)


def get_nn(X, k, n_jobs=1):
    """Compute the k-nearest neighbors of a dataset np.array (n_samples x n_dims)"""
    neigh = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs)
    neigh.fit(X)
    dists, inds = neigh.kneighbors(return_distance=True)
    return dists, inds


def asPointwise(X, class_instance, precomputed_knn=None, n_neighbors=100, n_jobs=1):
    """Use a global estimator
    as a pointwise one by creating kNN neighborhoods
    """
    if precomputed_knn is not None:
        knn = precomputed_knn
    else:
        _, knn = get_nn(X, k=n_neighbors, n_jobs=n_jobs)

    if n_jobs > 1:
        with Parallel(n_jobs=n_jobs) as parallel:
            def fit_estimator(class_instance, data_slice):
                return class_instance.fit(data_slice).dimension_
            # Asynchronously apply the `fit` function to each data point and collect the results
            results = parallel(
                delayed(fit_estimator)(
                    class_instance, X[i, :]
                ) for i in knn
            )
            return np.array(results)
    else:
        return np.array([class_instance.fit(X[i, :]).dimension_ for i in knn])


# class DocInheritorBase(type):
#    """ A metaclass to append GlobalEstimator or LocalEstimator Attributes section docstring to each estimator"""
#
#    def __new__(mcs, class_name, class_bases, class_dict):
#        # inherit class docstring: the docstring is constructed by traversing
#        # the mro for the class and merging their docstrings, with each next
#        # docstring as serving as the 'parent', and the accumulated docstring
#        # serving as the 'child'
#        this_doc = class_dict.get("__doc__", None)
#        for mro_cls in (mro_cls for base in class_bases for mro_cls in base.mro()):
#            prnt_cls_doc = mro_cls.__doc__
#            if prnt_cls_doc is not None:
#                if prnt_cls_doc == "The most base type":
#                    prnt_cls_doc = None
#            this_doc = mcs.class_doc_inherit(prnt_cls_doc, this_doc)
#
#        class_dict["__doc__"] = this_doc
#
#        return type.__new__(mcs, class_name, class_bases, class_dict)
#
#    @staticmethod
#    def class_doc_inherit(prnt_doc, child_doc):
#        """ Merge the docstrings of a parent class and its child.
#
#        Parameters
#        ----------
#        prnt_cls_doc: Union[None, str]
#        child_doc: Union[None, str]
#        """
#        if prnt_doc is None or "dimension_" not in prnt_doc:
#            return child_doc
#        else:
#            if "Attributes" in child_doc:
#                prnt_doc_attr = prnt_doc.index("dimension_")
#                child_doc = child_doc + prnt_doc[prnt_doc_attr:] + "\n"
#            else:
#                prnt_doc_attr = prnt_doc.index("Attributes")
#                child_doc = child_doc + "\n    " + prnt_doc[prnt_doc_attr:]
#        return child_doc


class GlobalEstimator(BaseEstimator):  # , metaclass=DocInheritorBase):
    """Template base class: inherit BaseEstimator, define transform, fit_transform, fit_pw, transform_pw, fit_transform_pw

    Attributes
    ----------
    dimension_ : {int, float}
        The estimated intrinsic dimension
    dimension_pw_ : np.array with dtype {int, float}
        Pointwise ID estimates
    dimension_pw_smooth_ : np.array with dtype float
        Smoothed pointwise ID estimates returned if self.fit_pw(smooth=True)
    """

    def _more_tags(self):
        return {
            "_skip_test": "check_methods_subset_invariance"
        }  # skip a test from sklearn.utils.estimator_checks because ID estimators are not subset invariant

    def transform(self, X=None):
        """Predict dimension after a previous call to self.fit

        Parameters
        ----------
        X : Dummy parameter

        Returns
        -------
        dimension_ : {int, float}
            The estimated ID
        """
        check_is_fitted(self, "is_fitted_")
        return self.dimension_

    def fit_transform(self, X, y=None):
        """Fit estimator and return ID

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        dimension_ : {int, float}
            The estimated intrinsic dimension
        """
        return self.fit(X).dimension_

    def fit_pw(self, X, precomputed_knn=None, smooth=False, n_neighbors=100, n_jobs=1):
        """Creates an array of pointwise ID estimates (self.dimension_pw_) by fitting the estimator in kNN of each point.

        Parameters
        ----------
        X: np.array (n_samples x n_neighbors)
            Dataset to fit
        precomputed_knn: np.array (n_samples x n_dims)
            An array of precomputed (sorted) nearest neighbor indices
        n_neighbors:
            Number of nearest neighbors to use (ignored when using precomputed_knn)
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
           smooth_

        Returns
        -------
        self : object
            Returns self
        """
        X = check_array(X, ensure_min_samples=n_neighbors + 1, ensure_min_features=2)

        if precomputed_knn is not None:
            knnidx = precomputed_knn
        else:
            _, knnidx = get_nn(X, k=n_neighbors, n_jobs=n_jobs)

        if n_jobs > 1:
            def fit_estimator(data_slice):
                return self.fit(data_slice).dimension_
            with Parallel(n_jobs=n_jobs) as parallel:
                # Asynchronously apply the `fit` function to each data point and collect the results
                results = parallel(
                    delayed(fit_estimator)(
                        X[i, :]
                    ) for i in knnidx
                )
            self.dimension_pw_ = np.array(results)
        else:
            self.dimension_pw_ = np.array(
                [self.fit(X[i, :]).dimension_ for i in knnidx]
            )

        if smooth:
            self.dimension_pw_smooth_ = np.zeros(len(knnidx))
            for i, point_nn in enumerate(knnidx):
                self.dimension_pw_smooth_[i] = np.mean(
                    np.append(self.dimension_pw_[i], self.dimension_pw_[point_nn])
                )
        return self

    def transform_pw(self, X=None):
        """Return an array of pointwise ID estimates after a previous call to self.fit_pw

        Parameters
        ----------
        X : Dummy parameter

        Returns
        -------
        dimension_pw_ : np.array with dtype {int, float}
            Pointwise ID estimates
        dimension_pw_smooth_ : np.array with dtype float
            Smoothed pointwise ID estimates returned if self.fit_pw(smooth=True)
        """

        check_is_fitted(
            self,
            "dimension_pw_",
            msg=(
                "This class instance is not fitted yet. Call 'fit_pw' with "
                "appropriate arguments before using this method."
            ),
        )

        if hasattr(self, "dimension_pw_smooth_"):
            return self.dimension_pw_, self.dimension_pw_smooth_
        else:
            return self.dimension_pw_

    def fit_transform_pw(
        self, X, precomputed_knn=None, smooth=False, n_neighbors=100, n_jobs=1
    ):
        """Returns an array of pointwise ID estimates by fitting the estimator in kNN of each point.

        Parameters
        ----------
        X: np.array (n_samples x n_neighbors)
            Dataset to fit
        precomputed_knn: bool
            An array of precomputed (sorted) nearest neighbor indices
        n_neighbors:
            Number of nearest neighbors to use (ignored when using precomputed_knn)
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
           smooth_

        Returns
        -------
        dimension_pw_ : np.array with dtype {int, float}
            Pointwise ID estimates
        dimension_pw_smooth_ : np.array with dtype float
            Smoothed pointwise ID estimates returned if self.fit_pw(smooth=True)
        """

        X = check_array(X, ensure_min_samples=n_neighbors + 1, ensure_min_features=2)

        if precomputed_knn is not None:
            knnidx = precomputed_knn
        else:
            _, knnidx = get_nn(X, k=n_neighbors, n_jobs=n_jobs)

        if n_jobs > 1:
            def fit_estimator(data_slice):
                return self.fit(data_slice).dimension_
            with Parallel(n_jobs=n_jobs) as parallel:
                # Asynchronously apply the `fit` function to each data point and collect the results
                results = parallel(
                    delayed(fit_estimator)(
                        X[i, :]
                    ) for i in knnidx
                )
            self.dimension_pw_ = np.array(results)
        else:
            dimension_pw_ = np.array([self.fit(X[i, :]).dimension_ for i in knnidx])

        if smooth:
            dimension_pw_smooth_ = np.zeros(len(knnidx))
            for i, point_nn in enumerate(knnidx):
                dimension_pw_smooth_[i] = np.mean(
                    np.append(dimension_pw_[i], dimension_pw_[point_nn])
                )
            return dimension_pw_, dimension_pw_smooth_
        else:
            return dimension_pw_
        

class FlexNbhdEstimator(BaseEstimator):
    '''
        Returns an array of pointwise ID estimates by fitting the estimator in kNN of each point.

        Parameters
        ----------
        X: np.array (n_samples x n_neighbors)
            Dataset to fit
        nbhd_dict: dict(landmark: list of nbhd indices)
        

        Returns
        -------
        dimension_pw : np.array
            Pointwise ID estimates
        dimension_pw_smooth : np.array
            If smooth is True, additionally returns smoothed pointwise ID estimates
        
    Also: add a distance matrix option
    in fersiwn 2
        # expand definition of locality: take subset of landmark points at which to evaulate local dim, and associated neighbourhoods, as dict {landmark: neighbourhood}
        # options to toggle between:
            # user defined nbhd
            # k-nn neighbourhoods, as in current version, but with landmarks
            # epsilon nbhds for landmarks, and maybe some default min epsilon = max_i min_{j \neq i} d(x_i, x_j)
        # fit should do the following:
            # takes in all the data
            # chops them up into subsets (not necessarily necessary for all local methods)
            # pass all data, nbhd dists, nbhd indices to _fit
            # local smoothing
            # averages out the dimensions
    '''
    @abstractmethod
    def _fit(self, X, nbhd_dict):
        """
        Custom method to each local ID estimator, called in fit
        
        """
        self._my_ID_estimator_func(X, nbhd_dict) 
        self.is_fitted_pw_ = True


    def fit(
        self,
        X,
        y=None,
        nbhd_dict=None,
        nbhd_type = 'knn',
        metric = 'minkowski',
        comb="mean",
        smooth=False,
        n_jobs=1,
        **kwargs,
    ): 
        '''
        Parameters
        X: (n_samples, n_features) or (n_samples, n_samples) if metric=’precomputed’
        nbhd_type: either 'knn' (k nearest neighbour) or 'eps' (eps nearest neighbour)
        metric: defaults to standard euclidean metric (minkowski with p = 2); if X a distance matrix, then set to 'precomputed'
        comb: method of averaging either 'mean' or 'median'
        smooth: if true, average over dimension estimates of local neighbourhoods using comb
        n_jobs: number of parallel processes in inferring local neighbourhood
        kwargs: keyword arguments, such as 'n_neighbors', or 'radius' for sklearn NearestNeighbor to infer local neighbourhoods

        '''
        if nbhd_dict is None:
            nbhd_dict = self.get_neigh(X, nbhd_type = nbhd_type, metric = metric, n_jobs=n_jobs, **kwargs)
        
        self._fit(X=X, nbhd_dict=nbhd_dict)
        
        self.aggr(comb)
        if smooth: self.smooth(nbhd_dict, comb)

        return self
    
    @staticmethod
    def get_neigh(X, nbhd_type = 'knn', metric = 'minkowski', n_jobs=1, **kwargs):
        """
        Parameters:

        X: (n_samples, n_features) or (n_samples, n_samples) if metric=’precomputed’
        nbhd_type: either 'knn' (k nearest neighbour) or 'eps' (eps nearest neighbour)
        metric: defaults to standard euclidean metric (minkowski with p = 2); if X a distance matrix, then set to 'precomputed'

        Returns:

        nbhd_dict: dictionary {point_index: list of neighbour point indices}
        dist: ndarray of shape (n_samples,) of arrays recording distance from point to neighbours          
        """
        neigh = NearestNeighbors(metric = metric, n_jobs = n_jobs, return_distance = False, **kwargs)
        neigh.fit(X)
        if nbhd_type == 'knn':
            indices = neigh.kneighbors() # Find k-nearest neighbors of each data sample
        elif nbhd_type == 'eps':
            indices = neigh.radius_neighbors() # Find eps-nearest neighbors of each data sample
        # Convert indices (either array of lists or n x k array) to dict
        nbhd_dict = {i: list(indices[i]) for i in range(X.shape[0])}

        return nbhd_dict
    
    def aggr(self, comb = 'mean'):
        #computes self.dimension_ from self.dimension_pw_
        if self.is_fitted_pw:
            if comb not in ['mean', 'median']:
                raise ValueError("Invalid comb parameter. It has to be 'mean' or 'median'")
            else:
                if comb == "mean":
                    self.dimension_ = np.mean(self.dimension_pw_)
                elif comb == "median":
                    self.dimension_ = np.median(self.dimension_pw_)  
                      
                self.is_fitted_ = True
        else:
            raise ValueError("No pointwise dimension fitted.")
        
        return None 
    
    def smooth(self, nbhd_dict, comb = 'mean'):
        # compute smoothed local estimates (aggregate over local neighbourhoods)
        if self.is_fitted:
            if comb not in ['mean', 'median']:
                raise ValueError("Invalid comb parameter. It has to be 'mean' or 'median'")
            else:
                self.dimension_pw_smooth_ = dict()
                for i in nbhd_dict:
                    nbhd_dims = [self.dimension_pw_[i]] + [self.dimension_pw_[j] for j in nbhd_dict[i]]
                    if comb == 'mean':
                        self.dimension_pw_smooth_[i] = np.mean(nbhd_dims)
                    elif comb == 'median':
                        self.dimension_pw_smooth_[i] = np.median(nbhd_dims)   
                self.is_fitted_pw_smooth_ = True
        else:
            raise ValueError("No pointwise dimension fitted.")
        
        return None
    

    @staticmethod
    def dmat_sparse(self, D, nbhd_dict):
        relevant_entries = self.dmat_relevant_entries(nbhd_dict)
        D_sparse = coo_array((np.array(D[pair] for pair in relevant_entries), # distance entries
                                ([p[0] for p in relevant_entries], # row entries
                                [p[1] for p in relevant_entries])), # column entries
                                shape = (D.shape[0], D.shape[0]))
        return D_sparse
        

    @staticmethod
    def dmat_relevant_entries(nbhd_dict):
        '''
        nbhd_dict: {landmark: neighbours}
        return: list pairs [(i,j)] of distinct points, where (i,j) either landmark neighbour, or pairs of neighbours of the same landmark
        '''
        relevant_entries = []
        for u in nbhd_dict:
            one_neigh = [(u,n) for n in nbhd_dict[u] if n > u]
            second_neigh = chain([[v for v in nbhd_dict[n] if v > u and v not in nbhd_dict[u]] for n in nbhd_dict[u]]) #search other points its in the same neighbourhood with, upper diag only
            second_neigh = list(set(second_neigh)) #avoid double counting
            second_neigh = [(u,v) for v in second_neigh]
            neigh = one_neigh + second_neigh
            relevant_entries.extend(neigh)
        return relevant_entries

    
    def transform(self, X=None):
        """Predict ID after a previous call to self.fit

        Parameters
        ----------
        X : Dummy parameter

        Returns
        -------
        dimension_ : {int, float}
            The estimated ID
        """
        check_is_fitted(self, "is_fitted_")
        return self.dimension_

    def fit_transform(
        self,
        X,
        y=None,
        nbhd_dict=None,
        smooth=False,
        nbhd = 'custom',
        distance_matrix = False,
        comb="mean",
        n_jobs=1,
        eps = None,
        n_neighbors = None,
        **kwargs,
    ):

        return self.fit(
            X,
            y=None,
            nbhd_dict=nbhd_dict,
            smooth=smooth,
            nbhd =nbhd,
            distance_matrix = distance_matrix,
            comb= comb,
            n_jobs= n_jobs,
            eps = eps,
            n_neighbors = n_neighbors,
            **kwargs,
        ).dimension_

    def transform_pw(self, X=None):
        """Return an array of pointwise ID estimates after a previous call to self.fit_pw

        Parameters
        ----------
        X : Dummy parameter

        Returns
        -------
        dimension_pw : np.array
            Pointwise ID estimates
        dimension_pw_smooth : np.array
            If self.fit_pw(smooth=True), additionally returns smoothed pointwise ID estimates
        """

        check_is_fitted(
            self,
            "dimension_pw_",
            msg=(
                "This class instance is not fitted yet. Call 'fit_pw' with "
                "appropriate arguments before using this method."
            ),
        )

        if hasattr(self, "dimension_pw_smooth_"):
            return self.dimension_pw_, self.dimension_pw_smooth_
        else:
            return self.dimension_pw_

    def fit_transform_pw(
        self,
        X,
        y=None,
        nbhd_dict=None,
        smooth=False,
        nbhd = 'custom',
        distance_matrix = False,
        comb="mean",
        n_jobs=1,
        eps = None,
        n_neighbors = None,
        **kwargs):

        self.fit(
            X,
            y=None,
            nbhd_dict=nbhd_dict,
            smooth=smooth,
            nbhd =nbhd,
            distance_matrix = distance_matrix,
            comb= comb,
            n_jobs= n_jobs,
            eps = eps,
            n_neighbors = n_neighbors,
            **kwargs,
        )

        if smooth:
            return self.dimension_pw_, self.dimension_pw_smooth_
        else:
            return self.dimension_pw_




class LocalEstimator(BaseEstimator):  # , metaclass=DocInheritorBase):
    """Template base class: generic _fit, fit, transform_pw for local ID estimators

    Attributes
    ----------
    dimension_ : {int, float}
        The estimated intrinsic dimension
    dimension_pw_ : np.array with dtype {int, float}
        Pointwise ID estimates
    dimension_pw_smooth_ : np.array with dtype float
        Smoothed pointwise ID estimates returned if self.fit(smooth=True)
    """

    _N_NEIGHBORS: int = 100  # default neighborhood parameter

    def _more_tags(self):
        """Skips a test from sklearn.utils.estimator_checks because ID estimators are not subset invariant"""
        return {"_skip_test": "check_methods_subset_invariance"}

    @abstractmethod
    def _fit(self, X, dists=None, knnidx=None, n_jobs=1):
        """Custom method to each local ID estimator, called in fit"""
        self._my_ID_estimator_func(X, dists, knnidx, n_jobs=n_jobs)

    def fit(
        self,
        X,
        y=None,
        precomputed_knn_arrays=None,
        smooth=False,
        n_neighbors=None,
        comb="mean",
        n_jobs=1,
    ):
        """Fitting method for local ID estimators

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API
        precomputed_knn_arrays: tuple[ np.array (n_samples x n_dims), np.array (n_samples x n_dims) ]
            Provide two precomputed arrays: (sorted nearest neighbor distances, sorted nearest neighbor indices)
        n_neighbors: int, default=self._N_NEIGHBORS
            Number of nearest neighbors to use (ignored when using precomputed_knn)
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
            smooth_

        Returns
        -------
        self : object
            Returns self.
        """
        # check inputs and define internal parameters
        if n_neighbors is None:
            n_neighbors = self._N_NEIGHBORS
        if n_neighbors >= len(X):
            warnings.warn("n_neighbors >= len(X), setting n_neighbors = len(X)-1")
            n_neighbors = len(X) - 1
        self.n_neighbors = n_neighbors
        self.comb = comb

        X = check_array(
            X, ensure_min_samples=self.n_neighbors + 1, ensure_min_features=2
        )

        if precomputed_knn_arrays is not None:
            dists, knnidx = precomputed_knn_arrays
        else:
            dists, knnidx = get_nn(X, k=self.n_neighbors, n_jobs=n_jobs)

        # fit
        self._fit(X=X, dists=dists, knnidx=knnidx, n_jobs=n_jobs)

        # combine local estimates
        if comb == "mean":
            self.dimension_ = np.mean(self.dimension_pw_)
        elif comb == "median":
            self.dimension_ = np.median(self.dimension_pw_)
        else:
            raise ValueError("Invalid comb parameter. It has to be 'mean' or 'median'")

        # compute smoothed local estimates
        if smooth:
            self.dimension_pw_smooth_ = np.zeros(len(knnidx))
            for i, point_nn in enumerate(knnidx):
                self.dimension_pw_smooth_[i] = np.mean(
                    np.append(self.dimension_pw_[i], self.dimension_pw_[point_nn])
                )
            self.is_fitted_pw_smooth_ = True
        self.is_fitted_pw_ = True
        self.is_fitted_ = True
        return self

    def transform(self, X=None):
        """Predict ID after a previous call to self.fit

        Parameters
        ----------
        X : Dummy parameter

        Returns
        -------
        dimension_ : {int, float}
            The estimated ID
        """
        check_is_fitted(self, "is_fitted_")
        return self.dimension_

    def fit_transform(
        self,
        X,
        y=None,
        precomputed_knn_arrays=None,
        smooth=False,
        n_neighbors=None,
        comb="mean",
        n_jobs=1,
    ):
        """Fit-transform method for local ID estimators

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API
        precomputed_knn_arrays: tuple[ np.array (n_samples x n_dims), np.array (n_samples x n_dims) ]
            Provide two precomputed arrays: (sorted nearest neighbor distances, sorted nearest neighbor indices)
        n_neighbors: int, default=self._N_NEIGHBORS
            Number of nearest neighbors to use (ignored when using precomputed_knn)
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
            smooth_

        Returns
        -------
        dimension_ : {int, float}
            The estimated intrinsic dimension
        """

        return self.fit(
            X,
            precomputed_knn_arrays=precomputed_knn_arrays,
            smooth=smooth,
            n_neighbors=n_neighbors,
            comb=comb,
            n_jobs=n_jobs,
        ).dimension_

    def transform_pw(self, X=None):
        """Return an array of pointwise ID estimates after a previous call to self.fit_pw

        Parameters
        ----------
        X : Dummy parameter

        Returns
        -------
        dimension_pw : np.array
            Pointwise ID estimates
        dimension_pw_smooth : np.array
            If self.fit_pw(smooth=True), additionally returns smoothed pointwise ID estimates
        """

        check_is_fitted(
            self,
            "dimension_pw_",
            msg=(
                "This class instance is not fitted yet. Call 'fit_pw' with "
                "appropriate arguments before using this method."
            ),
        )

        if hasattr(self, "dimension_pw_smooth_"):
            return self.dimension_pw_, self.dimension_pw_smooth_
        else:
            return self.dimension_pw_

    def fit_transform_pw(
        self, X, precomputed_knn_arrays=None, smooth=False, n_neighbors=None, n_jobs=1
    ):
        """
        Returns an array of pointwise ID estimates by fitting the estimator in kNN of each point.

        Parameters
        ----------
        X: np.array (n_samples x n_neighbors)
            Dataset to fit
        precomputed_knn_arrays: tuple[ np.array (n_samples x n_dims), np.array (n_samples x n_dims) ]
            Provide two precomputed arrays: (sorted nearest neighbor distances, sorted nearest neighbor indices)
        n_neighbors: int, default=self._N_NEIGHBORS
            Number of nearest neighbors to use (ignored when using precomputed_knn).
        n_jobs: int
            Number of processes
        smooth: bool, default = False
            Additionally computes a smoothed version of pointwise estimates by
            taking the ID of a point as the average ID of each point in its neighborhood (self.dimension_pw_)
           smooth_

        Returns
        -------
        dimension_pw : np.array
            Pointwise ID estimates
        dimension_pw_smooth : np.array
            If smooth is True, additionally returns smoothed pointwise ID estimates
        """

        self.fit(
            X,
            precomputed_knn_arrays=precomputed_knn_arrays,
            smooth=smooth,
            n_neighbors=n_neighbors,
            n_jobs=n_jobs,
        )

        if smooth:
            return self.dimension_pw_, self.dimension_pw_smooth_
        else:
            return self.dimension_pw_


# def mean_local_id(local_id, knnidx):
#    """
#    Compute the mean ID of all neighborhoods in which a point appears
#
#    Parameters
#    ----------
#    local_id : list or np.array
#        list of local ID for each point
#    knnidx : np.array
#        indices of kNN for each point returned by function get_nn
#
#    Results
#    -------
#    dimension_pw_smooth_ : np.array
#        list of mean local ID for each point
#
#    """
#    dimension_pw_smooth_ = np.zeros(len(local_id))
#    for point_i in range(len(local_id)):
#        # get all points which have this point in their neighbourhoods
#        all_neighborhoods_with_point_i = np.append(
#            np.where(knnidx == point_i)[0], point_i
#        )
#        # get the mean local ID of these points
#        dimension_pw_smooth_[point_i] = local_id[all_neighborhoods_with_point_i].mean()
#    return dimension_pw_smooth_


def binom_coeff(n, k):
    """
    Taken from : https://stackoverflow.com/questions/26560726/python-binomial-coefficient
    Compute the number of ways to choose $k$ elements out of a pile of $n.$

    Use an iterative approach with the multiplicative formula:
    $$\frac{n!}{k!(n - k)!} =
    \frac{n(n - 1)\dots(n - k + 1)}{k(k-1)\dots(1)} =
    \prod_{i = 1}^{k}\frac{n + 1 - i}{i}$$

    Also rely on the symmetry: $C_n^k = C_n^{n - k},$ so the product can
    be calculated up to $\min(k, n - k).$

    :param n: the size of the pile of elements
    :param k: the number of elements to take from the pile
    :return: the number of ways to choose k elements out of a pile of n
    """

    # When k out of sensible range, should probably throw an exception.
    # For compatibility with scipy.special.{comb, binom} returns 0 instead.
    if k < 0 or k > n:
        return 0

    if k == 0 or k == n:
        return 1

    total_ways = 1
    for i in range(min(k, n - k)):
        total_ways = total_ways * (n - i) // (i + 1)

    return total_ways


def check_random_generator(seed):
    """Turn seed into a numpy.random._generator.Generator' instance
    Parameters
    ----------
    seed : None | int | instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.default_rng()
    if isinstance(seed, numbers.Integral):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random._generator.Generator):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random._generator.Generator"
        " instance" % seed
    )
