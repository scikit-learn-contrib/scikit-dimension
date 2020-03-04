import inspect
import scipy.integrate
import numpy as np
from _commonfuncs import lens, get_nn
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array


class MLE(BaseEstimator):    
    """ A template estimator to be used as a reference implementation.
    For more information regarding how to build your own estimator, read more
    in the :ref:`User Guide <user_guide>`.
    Parameters
    ----------
    demo_param : str, default='demo_param'
        A parameter used for demonstation of how to pass and store paramters.
    """
    def __init__(self, k, dnoise = None, sigma = 0, n = None,
                    integral_approximation = 'Haro', unbiased = False,
                    neighborhood_based = True,
                    neighborhood_aggregation = 'maximum.likelihood',
                    iterations = 5, K = 5):
        
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

################################################################################

def maxLikGlobalDimEst(data, k, dnoise = None, sigma = 0, n = None,
                    integral_approximation = 'Haro', unbiased = False,
                    neighborhood_based = True,
                    neighborhood_aggregation = 'maximum.likelihood',
                    iterations = 5, K = 5):
  # 'k' is the number of neighbors used for each dimension estimation.
  # 'dnoise' is a vector valued function giving the transition density.
  # 'sigma' is the estimated standard deviation for the noise.
  # 'n' is the dimension of the noise (at least dim(data)[2])
  # integral.approximation can take values 'Haro', 'guaranteed.convergence', 'iteration'
  # neighborhood.based means that estimation is made for each neighborhood,
  # otherwise estimation is based on distances in entire data set.
  # 'K' is number of neighbors per data point that is considered, only used for 
  # neighborhood.based = False

    N = len(data)

    if (neighborhood_based):
        mi = maxLikPointwiseDimEst(data, k, dnoise, sigma, n,
                           integral_approximation = integral_approximation,
                           unbiased = unbiased)
        
        if neighborhood_aggregation == 'maximum.likelihood':
            de = 1/np.mean(1/mi) 
            
        elif neighborhood_aggregation == 'mean':
            de = np.mean(mi) 
        elif neighborhood_aggregation == 'median':
            de = robust = np.median(mi)
        return(de)


    dist, idx = get_nn(data, K)
    #dist = dist[~duplicated(dist[range(K*N)])]
    Rs = dist
    de = maxLikDimEstFromR(Rs, dnoise, np.sqrt(2)*sigma, n, integral_approximation, unbiased,
            iterations) # Since distances between points are used, noise is 
                        # added at both ends, i.e. variance is doubled. 
        
              #likelihood = np.nan
    return(de, np.nan) 


def maxLikPointwiseDimEst(data, k, dnoise = None, sigma = 0, n = None, indices = None,
                         integral_approximation = 'Haro', unbiased = False,
                         iterations = 5):
    ## estimates dimension around each point in data[indices, ]
    #
    # 'indices' give the indexes for which local dimension estimation should
    # be performed.
    # 'k' is the number of neighbors used for each local dimension estimation.
    # 'dnoise' is a vector valued function giving the transition density.
    # 'sigma' is the estimated standard deviation for the noise.
    # 'n' is the dimension of the noise (at least dim(data)[2])

    if indices is None:
        indices = np.arange(len(data))

    N = len(indices)
    nbh_dist, idx = get_nn(data, k)
    de = np.repeat(np.nan, N) # This vector will hold local dimension estimates

    for i in range(N):
        Rs = nbh_dist[i,:]
        de[i] = maxLikDimEstFromR(Rs, dnoise, sigma, n, integral_approximation, unbiased, iterations)

    return(de)


def maxLikLocalDimEst(data, dnoise = None, sigma = 0, n = None,
                   integral_approximation = 'Haro',
                   unbiased = False, iterations = 5):
    
    # assuming data set is local
    center = np.mean(data, axis=0)
    cent_data = data - center
    Rs = np.sort(lens(cent_data))
    de = maxLikDimEstFromR(Rs, dnoise, sigma, n, integral_approximation, unbiased, iterations)
    return(de)

################################################################################

def maxLikDimEstFromR(Rs, dnoise, sigma, n, integral_approximation = 'Haro',
                unbiased = False, iterations = 5):

    if integral_approximation not in ['Haro', 'guaranteed.convergence', 'iteration']:
        raise ValueError('Wrong integral approximation')

    if not type(dnoise) == 'closure' and dnoise is not None: 
        if not callable(dnoise):
            raise ValueError('dnoise must be a function')
        dnoise_orig = dnoise
    if not integral_approximation == 'Haro' and dnoise is not None:
        dnoise = lambda r, s, sigma, k: r*dnoise_orig(r, s, sigma, k)

    de = maxLikDimEstFromR_haro_approx(Rs, dnoise, sigma, n, unbiased)
    if (integral_approximation == 'iteration'):
        de = maxLikDimEstFromRIterative(Rs, dnoise_orig, sigma, n, de, unbiased)

    return(de)
################################################################################

def maxLikDimEstFromR_haro_approx(Rs, dnoise, sigma, n, unbiased = False):
    # if dnoise is the noise function this is the approximation used in Haro.
    # for 'guaranteed.convergence' dnoise should be r times the noise function
    # with 'unbiased' option, estimator is unbiased if no noise or boundary

    k = len(Rs)
    kfac = k-2 if unbiased else k-1

    Rk = max(Rs)
    if dnoise is None:
        return(kfac/(sum(np.log(Rk/Rs))))

    Rpr = Rk + 100*sigma

    numerator = np.repeat(np.nan, k - 1)
    denominator = np.repeat(np.nan, k - 1)

    for j in range(k-1):
        Rj = Rs[j]
        
        try:
            tc = scipy.integrate.quad(lambda x: dnoise(x, Rj, sigma, n) * np.log(Rk/x), 0, Rpr, epsrel = 1e-2)
        except:
            tc = np.nan

        if np.isnan(tc)[0]:
            return(np.nan)

        numerator[j] = numInt['value']

        try:
            tc = scipy.integrate.quad(lambda x: dnoise(x, Rj, sigma, n), 0, Rpr, epsrel = 1e-2)
        except:
            tc = np.nan

        if np.isnan(tc)[0]:
            return(np.nan)
        
        denominator[j] = denomInt['value']

    return(kfac/sum(numerator/denominator))

################################################################################

def maxLikDimEstFromRIterative(Rs, dnoise, sigma, n, init = 5,
                  unbiased = False, iterations = 5, verbose = False):
    m = init
    if verbose:
        print("Start iteration, intial value:", m, "\n")
    for i in range(iterations):
        m = maxLikDimEstFromRIterative_inner(Rs, dnoise, sigma, n, m, unbiased)
        if verbose:
            print("Iteration", i, ":", m, "\n")
        if verbose:
            print("\n")
    return(m)

def maxLikDimEstFromRIterative_inner(Rs, dnoise, sigma, n, m, unbiased):

    k = len(Rs)  
    kfac = k-2 if unbiased else k-1

    Rk = max(Rs)
    if dnoise is None:
        return(kfac/(sum(log(Rk/Rs))))
    Rpr = Rk + 100*sigma

    numerator = np.repeat(np.nan, k - 1)
    denominator = np.repeat(np.nan, k - 1)

    for j in range(k-1):
        Rj = Rs[j]
        m = max(m, 1)
        numInt = scipy.integrate.quad(lambda x: x**(m-1)*dnoise(x, Rj, sigma, n) * np.log(Rk/x), 0, Rpr, epsrel = 1e-2)
        numerator[j] = numInt['value']

        denomInt = scipy.integrate.quad(lambda x: x**(m-1)*dnoise(x, Rj, sigma, n), 0, Rpr, epsrel = 1e-2)
        denominator[j] = denomInt['value']

    return(kfac/sum(numerator/denominator))



def dnoiseGaussH(r, s, sigma, k = None):
    if (len(r) > 1 and len(s) > 1):
        raise ValueError('r and s cannot both be vectors')
    dnorm(s, r, sigma)              # f(s|r) in Haro et al. (2008) w/ Gaussian
                                        # transition density
                                        # 'k' is not used, but is input
                                        # for compatibility

################################################################################

def dnoiseGaussB(r, s, sigma, k):

    if (len(r) > 1 and len(s) > 1):
        raise ValueError('r and s cannot both be vectors')
    w1 = inthr(s, k, sigma)
    mu = intrhr(s, k, sigma)/w1
    tsigma2 = intr2hr(s, k, sigma)/w1 - mu^2
    w2 = 1 - pnorm(0, mean = mu, sd = sqrt(tsigma2))
    w = w1/w2
    gau = w*dnorm(r, mu, sqrt(tsigma2))
    return(gau*(r > 0))   # Best approximation of f(s|r) by truncated Gaussian
                        # when Gaussian k-dim noise.
                        # NB! The noncentral chi distribution is
                        # f(r|s), not f(s|r).

################################################################################

def dnoiseNcChi(r, s, sigma, k):

    if len(r) > 1 and len(s) > 1:
        raise ValueError('r and s cannot both be vectors')
    _lambda = r/sigma
    return 2*s/sigma**2*dchisq((s/sigma)**2, k, _lambda**2) # 'k' is the number of
                                                 # degrees of freedom