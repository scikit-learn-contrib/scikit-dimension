### Credits to Kerstin Johnsson
### https://cran.r-project.org/web/packages/intrinsicDimension/index.html
### for the original R implementation

import itertools
import numpy as np
import bisect
from scipy.special import gamma
from functools import lru_cache
from _commonfuncs import binom_coeff, lens, indComb, indnComb, efficient_indnComb
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_random_state


class ESS(BaseEstimator):
    """
    Class to calculate the intrinsic dimension of the provided data points with the ESS algorithm.
    
    -----------
    Attributes
    
   ver : str, 'a' or 'b'
       See Johnsson et al. (2015).
    d : int, default=1
        For ver ='a', any value of d is possible,  for ver ='b', only d = 1 is supported.
    -----------
    Returns
    
    dimension_ : int
        Intrinsic dimension of the dataset
    ess_ : float
        The Expected Simplex Skewness value.
        
    -----------
    References:
    """
    
    
    def __init__(self, ver = 'a', d = 1, random_state=None):
        self.ver = ver
        self.d = d
        self.random_state = random_state
        
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
            
            
        self.random_state_ = check_random_state(self.random_state)
        
        self.dimension_, self.essval_ = self._essLocalDimEst(X)
        
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self  


    def _essLocalDimEst(self, X):
        essval = self._computeEss(X, verbose = False)
        if (np.isnan(essval)):
            return(dict(de = np.nan, ess = np.nan))

        mindim = 1
        maxdim = 20
        dimvals = self._essReference(maxdim, mindim)
        while ((self.ver == 'a' and essval > dimvals[maxdim-1]) or
                (self.ver == 'b' and essval < dimvals[maxdim-1])):
            mindim = maxdim + 1
            maxdim = 2*(maxdim-1)
            dimvals=np.append(dimvals, self._essReference(maxdim, mindim))

        if (self.ver == 'a'):
            i = bisect.bisect(dimvals[mindim-1:maxdim],essval)
        else:
            i = len(range(mindim,maxdim+1)) - bisect.bisect(dimvals[mindim-1:maxdim][::-1],essval)

        de_integer = mindim+i-1
        de_fractional = (essval-dimvals[de_integer-1])/(dimvals[de_integer]-dimvals[de_integer-1])
        de = de_integer + de_fractional

        return(de, essval)

    ################################################################################

    def _computeEss(self, X, verbose = False):

        p = self.d + 1

        n = X.shape[1]
        if (p > n):
            if (self.ver == 'a'):
                return(0)
            if (self.ver == 'b'):
                return(1)
            else:
                raise ValueError('Not a valid version')

        vectors = self._vecToCs_onedir(X, 1)  
        if (verbose):
            print('Number of vectors:', len(vectors), '\n')

        #groups = indnComb(len(vectors), p)
        #if (len(groups) > 5000):
        #    groups = groups[np.random.choice(range(len(groups)),size=5000, replace=False),:]

        if len(vectors)>100: #sample 5000 combinations
            groups = efficient_indnComb(len(vectors), p, self.random_state_)
        else: #generate all combs with the original function
            groups = indnComb(len(vectors), p)

        if (verbose):
            print('Number of simple elements:', len(groups), '\n')

        Allist = [vectors[group] for group in groups]
        Alist = Allist

        # Compute weights for each simple element
        weight = np.prod([lens(l) for l in Alist],axis=1)
        if (self.ver == 'a'):
            # Compute the volumes of the simple elements
            vol = [np.sqrt(np.linalg.det(vecgr.dot(vecgr.T)))  for vecgr in Alist]
            return(np.sum(vol)/np.sum(weight))

        elif (self.ver == 'b'):
            if (self.d == 1):
                # Compute the projection of one vector onto one other
                proj = [np.abs(np.sum(vecgr[0,:] * vecgr[1,:])) for vecgr in Alist]
                return(np.sum(proj)/np.sum(weight))
            else:
                raise ValueError('For ver == "b", d > 1 is not supported.')

        else:
            raise ValueError('Not a valid version')

    ################################################################################

    @staticmethod
    def _vecToC_onedir(points, add_mids = False, weight_mids = 1,mids_maxdist = float('inf')):

        # Mean center data
        center = np.mean(points,axis=0)
        vecOneDir = points - center

        if (add_mids): # Add midpoints
            pt1,pt2,ic = indComb(len(vecOneDir))
            mids = (vecOneDir[ic[pt1], ] + vecOneDir[ic[pt2], ])/2
            dist = lens(vecOneDir[ic[pt1], ] - vecOneDir[ic[pt2], ])
            mids = mids[dist <= mids_maxdist, ] # Remove midpoints for very distant 
                                              # points
            vecOneDir = np.vstack((vecOneDir, weight_mids * mids))

        return(vecOneDir)

    def _vecToCs_onedir(self,points, n_group):
        if (n_group == 1):
            return(self._vecToC_onedir(points))

        NN = len(points)
        ind_groups = indnComb(NN, n_group)
        reshape_ind_groups = ind_groups.reshape((n_group,-1))
        point_groups = points[reshape_ind_groups,:].reshape((-1,n_group))
        group_centers = np.array([points[ind_group,:].mean(axis=0) for ind_group in ind_groups])
        centers = group_centers[np.repeat(np.arange(len(group_centers)), n_group),:]
        return(point_groups - centers)

    @lru_cache()
    def _essReference(self, maxdim, mindim=1):

        if (maxdim <= self.d + 2):
            raise ValueError("maxdim (", maxdim, ") must be larger than d + 2 (", self.d + 2, ")")

        if (self.ver == 'a'):
            ## ID(n) = factor1(n)**d * factor2(n)
            # factor1(n) = gamma(n/2)/gamma((n+1)/2)
            # factor2(n) = gamma(n/2)/gamma((n-d)/2)

            ## compute factor1
            # factor1(n) = gamma(n/2)/gamma((n+1)/2)
            # [using the rule gamma(n+1) = n * gamma(n)] repeatedly
            # = gamma(1/2)/gamma(2/2) * prod{j \in J1} j/(j+1) if n is odd
            # = gamma(2/2)/gamma(3/2) * prod(j \in J2) j/(j+1) if n is even
            # where J1 = np.arange(1, n-2, 2), J2 = np.arange(2, n-2, 2)
            J1 = np.array([1+i for i in range(0,maxdim+2,2) if 1+i<=maxdim])
            J2 = np.array([2+i for i in range(0,maxdim+2,2) if 2+i<=maxdim])
            factor1_J1 = gamma(1/2)/gamma(2/2) * np.concatenate((np.array([1]), np.cumprod(J1/(J1+1))[:-1]))
            factor1_J2 = gamma(2/2)/gamma(3/2) * np.concatenate((np.array([1]), np.cumprod(J2/(J2+1))[:-1]))
            factor1 = np.repeat(np.nan,maxdim)
            factor1[J1-1] = factor1_J1
            factor1[J2-1] = factor1_J2

            ## compute factor2
            # factor2(n) = gamma(n/2)/gamma((n-d)/2)
            # = gamma((d+1)/2)/gamma(1/2) * prod{k \in K1} k/(k-d) if n-d is odd
            # = gamma((d+2)/2)/gamma(2/2) * prod(k \in K2) k/(k-d) if n-d is even
            # where K1 = np.arange(d+1, n-2, 2), K2 = np.arange(d+2, n-2, 2)
            # if n > d+2, otherwise 0.
            K1 = np.array([self.d+1+i for i in range(0,maxdim+2,2) if self.d+1+i<=maxdim])
            K2 = np.array([self.d+2+i for i in range(0,maxdim+2,2) if self.d+2+i<=maxdim])
            factor2_K1 = gamma((self.d+1)/2)/gamma(1/2) * np.concatenate((np.array([1]), np.cumprod(K1/(K1-self.d))[:-1]))
            factor2_K2 = gamma((self.d+2)/2)/gamma(2/2) * np.concatenate((np.array([1]), np.cumprod(K2/(K2-self.d))[:-1]))
            factor2 = np.zeros(maxdim)
            factor2[K1-1] = factor2_K1
            factor2[K2-1] = factor2_K2
            # compute ID
            ID = factor1**self.d * factor2
            ID = ID[mindim-1:maxdim]
            return(ID)

        if (self.ver == 'b'):
            if (self.d == 1):
                # ID(n) = 2*pi**(-1/2)/n *gamma((n+1)/2)/gamma((n+2)/2)
                # = gamma(2/2)/gamma(3/2) * prod{j \in J1} (j+1)/(j+2) * 2/sqrt(pi)/n if n is odd
                # = gamma(3/2)/gamma(4/2) * prod(j \in J2) (j+1)/(j+2) * 2/sqrt(pi)/n if n is even
                # where J1 = np.arange(1, n-2, 2), J2 = np.arange(2, n-2, 2)
                J1 = np.array([1+i for i in range(0,maxdim+2,2) if 1+i<=maxdim])
                J2 = np.array([2+i for i in range(0,maxdim+2,2) if 2+i<=maxdim])
                ID_J1 = gamma(3/2)/gamma(2/2) * np.concatenate((np.array([1]), np.cumprod((J1+2)/(J1+1))[:-1]))
                ID_J2 = gamma(4/2)/gamma(3/2) * np.concatenate((np.array([1]), np.cumprod((J2+2)/(J2+1))[:-1]))
                ID = np.repeat(np.nan,maxdim)
                ID[J1-1] = ID_J1
                ID[J2-1] = ID_J2
                # n = mindim:maxdim
                # return(gamma((n+2)/2)/gamma((n+1)/2) * 2/sqrt(pi)/n)

                return(ID[mindim-1:maxdim] * 2/np.sqrt(np.pi)/np.array(range(mindim,maxdim+1)))

            raise ValueError('For ver == "b", d > 1 is not supported.')

        raise ValueError('Not a valid version')