import inspect
import scipy.integrate
import numpy as np
from _commonfuncs import lens, get_nn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

class MLE(BaseEstimator):    
    """ Intrinsic dimension estimation with the Maximum Likelihood method. The estimators are based on the referenced paper by Haro et al. (2008), using the assumption that there is a single manifold. The estimator in the paper is obtained using default parameters and dnoise = dnoiseGaussH.

    With integral.approximation = 'Haro' the Taylor expansion approximation of r^(m-1) that Haro et al. (2008) used are employed. With integral.approximation = 'guaranteed.convergence', r is factored out and kept and r^(m-2) is approximated with the corresponding Taylor expansion. This guarantees convergence of the integrals. Divergence might be an issue when the noise is not sufficiently small in comparison to the smallest distances. With integral.approximation = 'iteration', five iterations is used to determine m.

    maxLikLocalDimEst assumes that the data set is local i.e. a piece of a data set cut out by a sphere with a radius such that the data set is well approximated by a hyperplane (meaning that the curvature should be low in the local data set). See localIntrinsicDimension. 
    
    ----------
    Attributes
    
    mode : str, default='global'
        Whether to compute 'global', 'local' or 'pointwise' intrinsic dimension
    k : int
        Number of neighbors used for each dimension estimation.
    dnoise : function
        Vector valued function giving the transition density.
    sigma : float, default=0
        Estimated standard deviation for the noise.
    n : int, default='None'
        Dimension of the noise (at least data.shape[1])
    integral.approximation : str, default='Haro'
        Can take values 'Haro', 'guaranteed.convergence', 'iteration'
    neighborhood.based : bool, default='True'
        means that estimation is made for each neighborhood, otherwise the estimation is based on distances in the entire data set.
    K : int, default=5
        Number of neighbors per data point that is considered, only used for neighborhood.based = FALSE
        
    ---------
    Returns
    
    dimension_ : float
        The estimated intrinsic dimension
      
    ---------  
    References
    
    Haro, G., Randall, G. and Sapiro, G. (2008) Translated Poisson Mixture Model for Stratification Learning. Int. J. Comput. Vis., 80, 358-374.

    Hill, B. M. (1975) A simple general approach to inference about the tail of a distribution. Ann. Stat., 3(5) 1163-1174.

    Levina, E. and Bickel., P. J. (2005) Maximum likelihood estimation of intrinsic dimension. Advances in Neural Information Processing Systems 17, 777-784. MIT Press. 
    
    """
    def __init__(self, mode = 'global', k = 20, dnoise = None, sigma = 0, n = None,
                    integral_approximation = 'Haro', unbiased = False,
                    neighborhood_based = True,
                    neighborhood_aggregation = 'maximum.likelihood',
                    iterations = 5, K = 5):
        
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)
            
    def fit(self,X,y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A data set for which the intrinsic dimension is estimated.
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
            
        #if self.k >= len(X):
        #    warnings.warn('k larger or equal to len(X), using len(X)-1')
        #if self.K >= len(X):
        #    warnings.warn('k larger or equal to len(X), using len(X)-1')
        
        if self.mode == 'global':
            self.dimension_ = self.maxLikGlobalDimEst(X)            
        elif self.mode == 'pointwise':
            self.dimension_ = self.maxLikPointwiseDimEst(X)            
        elif self.mode == 'local':
            self.dimension_ = self.maxLikLocalDimEst(X)
            
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self        
            
    def maxLikGlobalDimEst(self,X):
        # 'k' is the number of neighbors used for each dimension estimation.
        # 'dnoise' is a vector valued function giving the transition density.
        # 'sigma' is the estimated standard deviation for the noise.
        # 'n' is the dimension of the noise (at least dim(data)[2])
        # integral.approximation can take values 'Haro', 'guaranteed.convergence', 'iteration'
        # neighborhood.based means that estimation is made for each neighborhood,
        # otherwise estimation is based on distances in entire data set.
        # 'K' is number of neighbors per data point that is considered, only used for 
        # neighborhood.based = False

        if self.neighborhood_based:
            mi = self.maxLikPointwiseDimEst(X)

            if self.neighborhood_aggregation == 'maximum.likelihood':
                de = 1/np.mean(1/mi) 

            elif self.neighborhood_aggregation == 'mean':
                de = np.mean(mi) 
            elif self.neighborhood_aggregation == 'median':
                de = np.median(mi)
            return(de)

        else:
            dist, idx = get_nn(X, min(self.K,len(X)-1))
            Rs = np.sort(np.array(list(set(dist.flatten(order='F')))))[:self.k]
            de = self._maxLikDimEstFromR(Rs, np.sqrt(2)*sigma) # Since distances between points are used, noise is 
                                                               # added at both ends, i.e. variance is doubled. 
                                                               #likelihood = np.nan
            return(de) 

    def maxLikPointwiseDimEst(self,X):
        ## estimates dimension around each point in data[indices, ]
        #
        # 'indices' give the indexes for which local dimension estimation should
        # be performed.
        # 'k' is the number of neighbors used for each local dimension estimation.
        # 'dnoise' is a vector valued function giving the transition density.
        # 'sigma' is the estimated standard deviation for the noise.
        # 'n' is the dimension of the noise (at least dim(data)[2])

        N = len(X)
        nbh_dist, idx = get_nn(X, min(self.k,len(X)-1))
        de = np.repeat(np.nan, N) # This vector will hold local dimension estimates

        for i in range(N):
            Rs = nbh_dist[i,:]
            de[i] = self._maxLikDimEstFromR(Rs,self.sigma)

        return(de)

    def maxLikLocalDimEst(self,X):
        # assuming data set is local
        center = np.mean(X, axis=0)
        cent_X = X - center
        Rs = np.sort(lens(cent_X))
        de = self._maxLikDimEstFromR(Rs, sigma)
        return(de)

    def _maxLikDimEstFromR(self, Rs, sigma):

        if self.integral_approximation not in ['Haro', 'guaranteed.convergence', 'iteration']:
            raise ValueError('Wrong integral approximation')

        if self.dnoise == 'dnoiseGaussH':
            self.dnoise = self._dnoiseGaussH

        if not self.integral_approximation == 'Haro' and self.dnoise is not None:
            self.dnoise = lambda r, s, sigma, k: r*self.dnoise(r, s, sigma, k)

        de = self._maxLikDimEstFromR_haro_approx(Rs, self.sigma)
        if (self.integral_approximation == 'iteration'):
            raise ValueError("integral_approximation='iteration' not implemented yet. See R intdim package")
            #de = maxLikDimEstFromRIterative(Rs, dnoise_orig, sigma, n, de, unbiased)

        return(de)

    def _maxLikDimEstFromR_haro_approx(self, Rs, sigma):
        # if dnoise is the noise function this is the approximation used in Haro.
        # for 'guaranteed.convergence' dnoise should be r times the noise function
        # with 'unbiased' option, estimator is unbiased if no noise or boundary

        k = len(Rs)
        kfac = k-2 if self.unbiased else k-1

        Rk = np.max(Rs)
        if self.dnoise is None:
            return(kfac/(np.sum(np.log(Rk/Rs))))

        Rpr = Rk + 100*sigma

        numerator = np.repeat(np.nan, k - 1)
        denominator = np.repeat(np.nan, k - 1)

        numInt = lambda x: self.dnoise(x, Rj, sigma, n) * np.log(Rk/x)
        denomInt = lambda x: self.dnoise(x, Rj, sigma, n)

        for j in range(k-1):
            Rj = Rs[j]
            numerator[j] = scipy.integrate.quad(numInt, 0, Rpr, epsrel = 1e-2, epsabs = 1e-2)[0]
            denominator[j] = scipy.integrate.quad(denomInt, 0, Rpr, epsrel = 1e-2, epsabs = 1e-2)[0]


        return(kfac/np.sum(numerator/denominator))

    @staticmethod
    def _dnorm(x,mu=0,sigma=1): 
        return np.exp(-.5*((x-mu)/sigma)**2)/(sigma*np.sqrt(2*np.pi))

    def _dnoiseGaussH(r, s, sigma, k = None):
        return self._dnorm(s,mu=r,sigma=sigma) 
                                            # f(s|r) in Haro et al. (2008) w/ Gaussian
                                            # transition density
                                            # 'k' is not used, but is input
                                            # for compatibility

    #def maxLikDimEstFromRIterative(Rs, dnoise, sigma, n, init = 5,
    #                  unbiased = False, iterations = 5, verbose = False):
    #    m = init
    #    if verbose:
    #        print("Start iteration, intial value:", m, "\n")
    #    for i in range(iterations):
    #        m = maxLikDimEstFromRIterative_inner(Rs, dnoise, sigma, n, m, unbiased)
    #        if verbose:
    #            print("Iteration", i, ":", m, "\n")
    #        if verbose:
    #            print("\n")
    #    return(m)
    #
    #def maxLikDimEstFromRIterative_inner(Rs, dnoise, sigma, n, m, unbiased):
    #
    #    k = len(Rs)  
    #    kfac = k-2 if unbiased else k-1
    #
    #    Rk = np.max(Rs)
    #    if dnoise is None:
    #        return(kfac/(np.sum(np.log(Rk/Rs))))
    #    Rpr = Rk + 100*sigma
    #
    #    numerator = np.repeat(np.nan, k - 1)
    #    denominator = np.repeat(np.nan, k - 1)
    #    
    #    numInt = lambda x: x**(m-1)*dnoise(x, Rj, sigma, n) * np.log(Rk/x)
    #    denomInt = lambda x: x**(m-1)*dnoise(x, Rj, sigma, n)
    #    
    #    for j in range(k-1):
    #        Rj = Rs[j]
    #        m = np.maximum(m, 1)
    #        numerator[j] = scipy.integrate.quad(numInt, 0, Rpr, epsrel = 1e-2,epsabs = 1e-2)[0]
    #        denominator[j] = scipy.integrate.quad(denomInt, 0, Rpr, epsrel = 1e-2,epsabs = 1e-2)[0]
    #
    #    return(kfac/sum(numerator/denominator))