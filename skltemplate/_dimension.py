
# dimension.py
# Author: Johannes von Lindheim, Zuse-Institute Berlin, 2018
# Used for "On Intrinsic Dimension Estimation and Minimal Diffusion Maps Embeddings of Point Clouds",
# Johannes von Lindheim, master thesis (FU Berlin)
#
# This file provides the functionality for the Gaussian annulus based ID estimation techniques, as well as another
# angle-based technique.
# 
# IMPORTANT REMARK: To be able to execute this code, a file called "gaussian_kde.py" needs to lie in the same
# directory, containing the code available at
# https://gist.github.com/tillahoffmann/f844bce2ec264c1c8cb5
# (accessed last at 02 May 2018). This code performs the weighted kernel density estimation
# and is written by Till Hoffmann, 2014.

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize
import scipy.sparse as sps
from scipy.stats import gaussian_kde as sp_gaussian_kde
from scipy.stats import norm
from scipy.special import gamma
from scipy.special import logsumexp
from gaussian_kde import gaussian_kde # this is code written by Till Hoffmann, performing the weighted KDE.
import time


###################################
###################################
### Gaussian annulus estimators ###
###################################
###################################

def dim_annulus_heuristic(data, which='variance', single_estimate='mean', neighbors_method='eps', k=100, r=3.0,
    n_sigma_orders_margin = 2, n_sigma_steps_per_order = 50, kl_part=0.5, dim_kde_rel=0.1, n_kde_steps=200, rounded=False, verbose=True):
    '''
    Estimates the intrinsic dimension d of a given (extrinsically) D-dimensional dataset in multiple ways:
    measuring the variance and/or the density peak of the density function, which is obtained by integrating
    the angular coordinates of Gauss kernels around the data points.

    Parameters:
        data: (n, D)-shaped array, dataset of which to estimate the intrinsic dimension
        which='variance': can be 'variance' or 'peak'; which heuristic to compute (variance or peak estimator)
        single_estimate='mean': can be 'peak', 'mean' or None. Since an estimate for every point is computed
        neighbors_method='eps': can be 'eps' or 'k' and determined, which method of nearest neighbors to use.
        k=100: determines number of nearest neighbors in the k-NN method, which is just performed for
            the heuristic determining the correct epsilon
        r=3.0: If eps-nearest-neighbors is chosen, then the cutoff will be where points have lower kernel
            value than exp(-r).
        n_sigma_orders_margin=2: Number of ordners of magitude above and below the chosen cutoff radius epsilon, in
            which it is searched for the optimal sigma in the sigma-heuristic.
        n_sigma_steps_per_order=50: How many different sigma are tried in every order of magnitude (factor 10).
        kl_part=0.5: How many points are discarded for the global ID estimate after the Kullback-Leibler-optimization.
        dim_kde_rel=0.1: when doing a kernel density estimation with the estimators as well as calculating
            the KL divergence, this is the low percentile of dimensions not taken into account, the high percentile
            is 1-dim_kde_rel (after that, dim_kde_abs is added). E.g. if set 0.1, then the smallest 10%, as well
            as the highest 10% of the estimators are discarded. After that, the dimension range is again widened
            by 5.0 dimensions. The procedure is meant to discard extreme outliers, to be able to just use a
            linear range, without throwing away resolution.
        rounded=False: whether the estimation is allowed to be any floating number or whether it should be rounded to the next integer.
        verbose=True: whether to print some verbose data.
    returns: either an array of estimators and an array for the indices, for which the estimate was computed (if single_estimate=None),
        else this is a scalar (mean or max-peak-value, depending on the single_estimate-parameter. If rounded is True,
        the result is rounded to an integer.

    '''

    assert any((which=='variance', which=='peak')), "Unknown which-parameter"
    assert any((neighbors_method=='eps', neighbors_method=='k')), "Unknown neighbors_method-parameter"
    assert any((single_estimate=='peak', single_estimate=='mean', single_estimate is None)), "Unknown single_estimate-parameter"

    n_pts = data.shape[0]
    estimators = np.zeros(n_pts)
    considered_pts = np.arange(n_pts)
    result = -1.0

    if verbose:
        print("Compute eps-heuristic")

    # Compute epsilon-heuristic with given k and
    eps = eps_heuristic_bgh(data, k=k)
    #eps = eps_heuristic_bgh_original(data, k=k)
    if verbose:
        print("eps = {:.3f}".format(eps))

    if verbose:
        print("Do nearest-neighbors routine")

    # Compute nearest neighbor distances and connectivity
    neighbor_obj = None
    if neighbors_method == 'eps':
        cutoff = np.sqrt(r*eps)
        neighbor_obj = NearestNeighbors(radius=cutoff, algorithm='auto')
        neighbor_obj.fit(data)
        dists = neighbor_obj.radius_neighbors_graph(mode='distance')
        connectivity = neighbor_obj.radius_neighbors_graph(mode='connectivity')
        inds= neighbor_obj.kneighbors(return_distance=False)
        
    elif neighbors_method == 'k':
        neighbor_obj = NearestNeighbors(n_neighbors=k, algorithm='auto')
        neighbor_obj.fit(data)
        dists = neighbor_obj.kneighbors_graph(n_neighbors=k, mode='distance')
        connectivity = neighbor_obj.kneighbors_graph(n_neighbors=k, mode='connectivity')
        inds= neighbor_obj.kneighbors(return_distance=False)
        
    if verbose:
        print("Compute sigma-heuristic")

    # Compute sigma-heuristic
    sigma_range = geom_range(eps, n_sigma_orders_margin, n_sigma_steps_per_order)
    start = time.clock()
    pre_sigmas = sigma_heuristic_center_dev(data, dists, connectivity, sigma_range)
    end = time.clock()
    if verbose:
        print("Time spent: {:.4f}".format(end - start))

    # Cut out non-successful sigma-estimations
    for i in range(len(pre_sigmas)):
        if pre_sigmas[i] <= 0.0:
            pre_sigmas[i] = np.mean(pre_sigmas[pre_sigmas[inds[i][:10]]>0.0])
    valid_inds = np.arange(n_pts)[pre_sigmas > 0.0]
 
    if len(valid_inds) < 0.1 * n_pts:
        raise ValueError("Variances for less than 10% of the data points - {0}/{1} - were found!".format(len(valid_inds), n_pts))
    sigmas = pre_sigmas[valid_inds]
    dists = dists[valid_inds, :]
    considered_pts = considered_pts[valid_inds]

    if verbose:
        print("Computed sigmas, {0}/{1} points left".format(len(valid_inds), n_pts))

    if verbose:
        print("Compute pre-estimators")

    # Zwischenschätzer berechnen
    if which == 'variance':
        estimators = variance_estimators(dists, sigmas)
    elif which == 'peak':
        estimators = peak_estimators(dists, sigmas, cut_part=dim_kde_rel, n_kde_steps=n_kde_steps)

    if verbose:
        print("Cut out the ones with high Kullback-Leibler divergence")

    # Cut out the part of the estimators with a high Kullback-Leibler divergence
    divergences = kl_divergences(dists, estimators, sigmas)
    sorted_indices = np.argsort(divergences)
    low_kl_inds = sorted_indices[:np.floor(kl_part * n_pts).astype(int)]
    estimators = estimators[low_kl_inds]
    considered_pts = considered_pts[low_kl_inds]

    # Either compute single estimate from all data points, or take whole array
    if single_estimate == 'peak':
        # Do kde
        cut_estimators = cut_bdry_percentiles(estimators)
        x = np.linspace(min(cut_estimators), max(cut_estimators), n_kde_steps)
        estimator_pdf = sp_gaussian_kde(estimators)
        y = estimator_pdf(x)
        result = x[np.argmax(y)]
    elif single_estimate == 'mean':
        result = estimators.mean()
    elif single_estimate is None:
        result = estimators

    if rounded:
        result = np.round(result).astype(int)

    if all((verbose, single_estimate is not None)):
        print("Final result: {:.2f}".format(result))

    if single_estimate is None:
        return result, considered_pts
    else:
        return result


def variance_estimators(dists, sigmas):

    n_pts = dists.shape[0]
    estimators = np.zeros(n_pts)

    for i in range(n_pts):
        norms = dists[i, :].data
        weights = norm.pdf(norms, scale=sigmas[i])
        estimators[i] = np.dot(np.power(norms, 2), weights) / np.sum(weights) / np.power(sigmas[i], 2)

    return estimators


def peak_estimators(dists, sigmas, cut_part=0.1, n_kde_steps=200):

    n_pts = dists.shape[0]
    estimators = np.zeros(n_pts)

    var_estimators = variance_estimators(dists, sigmas)
    var_estimators = cut_bdry_percentiles(var_estimators, cut_part*100)

    for i in range(n_pts):
        x = np.linspace(max(0.0, sigmas[i]*np.sqrt(min(var_estimators)-1)), sigmas[i]*np.sqrt(max(var_estimators)-1), n_kde_steps)
        norms = dists[i, :].data
        pdf_weighted = gaussian_kde(dataset=norms, weights=norm.pdf(norms, scale=sigmas[i]))
        y_weighted = pdf_weighted(x)
        peak_weighted = x[np.argmax(y_weighted)]
        estimators[i] = np.square(peak_weighted/sigmas[i]) + 1

    return estimators


def kl_divergences(dists, estimators, sigmas, num_steps=300):

    n_pts = dists.shape[0]
    divergences = np.zeros(n_pts)

    rough_estimate = np.mean(estimators)
    margin = 3.0 * np.std(estimators, ddof=1)

    for i in range(n_pts):

        # do kde for a (large enough) linspace, evaluate at all these points
        x = np.linspace(max(0, sigmas[i]*(np.sqrt(rough_estimate-1)-margin)), sigmas[i]*(np.sqrt(rough_estimate-1)+margin), num_steps)
        norms = dists[i, :].data
        estimator_pdf = gaussian_kde(norms, weights=norm.pdf(norms, scale=sigmas[i]))
        p = estimator_pdf(x)
        bump_f = annulus_bump_func(x, estimators[i], sigma=sigmas[i])
        zero_inds = np.union1d(np.where(bump_f == 0)[0], np.where(p == 0)[0])
        non_zero_inds = np.setdiff1d(np.arange(num_steps), zero_inds)
        p = p[non_zero_inds]
        bump_f = bump_f[non_zero_inds]

        divergences[i] = np.sum(p * np.log(p/bump_f))/len(non_zero_inds)

    return divergences

def annulus_bump_func(r, d, sigma):
    '''
    Here, d is the (estimation of) the dimension of the manifold, the data points lie on
    '''
    return np.exp(-np.power(r/sigma, 2)/2)*np.power(r, d-1)/np.power(2, d/2 - 1)/gamma(d/2)/np.power(sigma, d)

################
## Heuristics ##
################

def eps_heuristic_bgh(data, k=100, epsilons=None):

    result = -1.0

    # Fit k-NN-object
    neighbor_obj = NearestNeighbors(n_neighbors=k, algorithm='auto')
    neighbor_obj.fit(data)
    all_dists = neighbor_obj.kneighbors_graph(mode='distance').data

    n_iterations = 1

    # 1. iteration: If no epsilon-range is given, start with rough range
    if epsilons is None:
        n_iterations = 2
        epsilons = geom_range(1.0, 50, 3)

    while n_iterations > 0:
        epsilons = np.sort(epsilons).astype('float')
        sumexp = np.array([np.sum(np.exp(-all_dists**2/eps)) for eps in epsilons])
        differential = np.diff(sumexp) # works better on logarithmic scale, so we do not need to
            # divide by the - not difference but - ratio, because the ratio is const
        result = epsilons[np.argmax(differential)]

        # 2. iteration: Refine the range
        epsilons = geom_range(result, 1, 30)
        n_iterations -= 1

    return result

def eps_heuristic_bgh_original(data, k=100, epsilons=None):

    result = -1.0

    # Fit k-NN-object
    neighbor_obj = NearestNeighbors(n_neighbors=k, algorithm='auto')
    neighbor_obj.fit(data)
    all_dists = neighbor_obj.kneighbors_graph(mode='distance').data
    all_dists = np.append(all_dists, np.zeros(data.shape[0])) # include zero distances to points themselves!

    n_iterations = 1

    # 1. iteration: If no epsilon-range is given, start with rough range
    if epsilons is None:
        n_iterations = 2
        epsilons = geom_range(1.0, 50, 3)

    while n_iterations > 0:
        epsilons = np.sort(epsilons).astype('float')
        log_T = [logsumexp(-all_dists/eps) for eps in epsilons]
        log_eps = np.log(epsilons)
        log_deriv = np.diff(log_T)/np.diff(log_eps)
        result = epsilons[np.argmax(log_deriv)]

        # 2. iteration: Refine the range
        epsilons = geom_range(result, 1, 30)
        n_iterations -= 1

    return result

def sigma_heuristic_center_dev(data, dists, connectivity, sigma_range):

    n_pts = data.shape[0]
    dim = data.shape[1]
    n_sigmas = len(sigma_range)
    scaled_deviations = np.zeros(n_sigmas)
    opt_sigmas = np.zeros(n_pts)
    np.seterr(all='ignore')

    for i in range(n_pts):
        neighbor_inds = np.arange(n_pts)[np.squeeze(connectivity[i, :].toarray()).astype(bool)]
        neighbor_inds = neighbor_inds[neighbor_inds != i] # remove the point itself
        neighbor_norms = dists[i, neighbor_inds].data

        for j, sigma in enumerate(sigma_range): # one could vectorize this... But then the inf-problems
                # are more difficult to handle
            try:
                # Measure deviation from center point, rescaled by sigma and np.sqrt(np.square(weights).sum())
                weights = norm.pdf(neighbor_norms, scale=sigma)
                weights = weights / weights.sum() # normalize weights
                mean = np.dot(weights, data[neighbor_inds, :]) # weighted arithmetic mean
                scaled_deviations[j] = np.linalg.norm(data[i, :] - mean) / sigma / np.sqrt(np.square(weights).sum())
            except FloatingPointError as err:
                scaled_deviations[j] = np.inf # better would be to just exclude the inf-points

            if j > 0:
                if scaled_deviations[j] - scaled_deviations[j-1] >= 0.0:
                    break

        # Find first extremum (minimum)
        differences = (scaled_deviations[1:] - scaled_deviations[:-1])/np.diff(sigma_range) # differentiate
        pos_slope_indices = np.arange(n_sigmas-1)[np.vstack((np.isfinite(differences), differences > 0)).all(axis=0)]
        if len(pos_slope_indices) > 0:
            opt_sigmas[i] = sigma_range[pos_slope_indices[0]]
        else:
            opt_sigmas[i] = -1.0

    return opt_sigmas

#######################
## Utility functions ##
#######################

def geom_range(center, n_orders_margin, n_steps_per_order, base=10.0):
    '''
    Give a geometric range of possible sigmas (i.e., a factor to a range of powers).

    center: center (median) of range
    n_orders_margin: orders of magnitude (in the given base, default 10) above and below the center
    n_steps_per_order: number of sigmas in each oder of magnitude
    base: one order of magnitude up is the multiplication with factor base.
    '''
    step_factor = np.power(base, np.power(n_steps_per_order, -1.0))
    return center * (step_factor ** np.arange(-n_orders_margin*n_steps_per_order, n_orders_margin*n_steps_per_order+1))

def cut_bdry_percentiles(data, cutoff=10):
    low = np.percentile(data, q=cutoff)
    high = np.percentile(data, q=100-cutoff)
    return data[(data >= low) * (data <= high)]

##############
## Plotting ##
##############

def plot_estimators(dists, sigmas, estimators=None, which='variance'):

    if estimators is None:
        valid_inds = np.arange(dists.shape[0])[sigmas > 0.0]
        if which == 'variance':
            estimators = variance_estimators(dists[valid_inds], sigmas[valid_inds])
        elif which == 'peak':
            estimators = peak_estimators(dists[valid_inds], sigmas[valid_inds])
        else:
            raise ValueError("Did not recognize which-value")

    num_steps = 300
    cut_estimators = cut_bdry_percentiles(estimators)
    x = np.linspace(min(cut_estimators), max(cut_estimators), num_steps)
    estimator_pdf = sp_gaussian_kde(estimators)
    y = estimator_pdf(x)
    plt.plot(x, y)

    print("Mean of quad norm estimators: {:.3}".format(np.mean(estimators)))
    print("Maximum density of quad norm estimators at: {:.3}".format(x[np.argmax(y)]))



################################
################################
### Varnace-based estimators ###
################################
################################

def dim_eps_heuristic(data, epsilons, center=True, smooth=False, rounded=False, **kwargs):
    '''
    Estimates the intrinsic dimension d of a given (extrinsically) D-dimensional dataset measuring the variance of the cosigns of neighbors.

    Parameters:
        data: (n, D)-shaped array, dataset of which to estimate the intrinsic dimension
        epsilons: array of epsilons to test for each datapoint. For every x in epsilons and every data point p, the two neighbors with distance closest to x are taken to measure angle. See warning below!
        center: whether to subtract the mean of the data from all points.
        smooth: whether to use the neighbor-smoother (averaging over all eps-neighbors), in which case the smooth_eps-keyword has to be given.
        PLANNED: project=True: whether to project down one dimension in the estimation to prevent concentration of measure issues.
        rounded=False: whether the estimation is allowed to be any floating number or whether it should be rounded to the next integer.
    returns: d (scalar) - estimation of the intrinsic dimension. Can be continuous number, depending on the rounded-parameter.

    WARNING: Distances are just computed up to the maximum epsilon given. Therefore, neighbors may be chosen, that do not have the distance exactly closest to the epsilons-entry
    '''

    # Init
    neighbors_per_pt = 2 # possible spot to improve
    n_pts = data.shape[0]
    dim_extr = data.shape[1]
    n_epsilons = len(epsilons)
    eps_sorted = np.sort(epsilons)
    eps_max = eps_sorted[-1]

    # Allocate memory
    cosigns = np.zeros((n_pts, n_epsilons))

    # Smooth noise out, if requested
    if smooth:
        eps = kwargs.get('smooth_eps')
        assert eps is not None, "specify smooth_eps-parameter"
        data = neighbor_smoother(data, compute_eps_nn=True, eps=eps)

    # Center data, if requested
    if center:
        data = data - np.mean(data, axis=0)

    # Do eps-nearest-neighbors with maximum epsilon
    neighbor_obj = NearestNeighbors(radius=eps_max, algorithm='auto')
    neighbor_obj.fit(data)
    # graph = neighbor_obj.kneighbors_graph(data) - sps.identity(n_pts) # only care about two nearest neighbors, not point itself
    graph = neighbor_obj.radius_neighbors_graph(radius=eps_max, mode='distance')

    # For every epsilon
    for j, eps in enumerate(eps_sorted):

        # Sample heuristic from every point
        for i in range(n_pts):

            # Find neighbors, which have distance closest to eps
            v = data[i, :]
            dists = graph.data[graph.indptr[i]:graph.indptr[i+1]]
            dist_args = np.argsort(np.abs(dists - eps))
            x1 = data[graph.indices[graph.indptr[i] + dist_args[0]]]
            x2 = data[graph.indices[graph.indptr[i] + dist_args[1]]]

            # Project neighbor vectors onto plane orthogonal to v (and normalize) to prevent concentration of measure issues
            vv = v.dot(v)
            x1_proj = x1 - x1.dot(v)/vv * v
            x1_proj = sk_normalize(x1_proj.astype(float).reshape(1, -1), norm='l2', copy=True)
            x2_proj = x2 - x2.dot(v)/vv * v
            x2_proj = sk_normalize(x2_proj.astype(float).reshape(1, -1), norm='l2', copy=True)

            # Calculate cosign (= scalar prduct) of normalized vectors
            cosigns[i, j] = x1_proj.dot(x2_proj.T)

    # Calculate sample variance of cosigns
    variances = np.var(cosigns, axis=0, ddof=1)

    # The inverse is the estimation. This will be wrong by one dimension, when dataset was already full-dimensional!
    result = 1/variances
    if rounded:
        result = np.round(result).astype(int)

    return result

def dim_2NN_heuristic(data, center=True, smooth=True, rounded=False, **kwargs):
    '''
    Estimates the intrinsic dimension d of a given (extrinsically) D-dimensional dataset measuring the variance of the cosigns of neighbors.

    Parameters:
        data: (n, D)-shaped array, dataset of which to estimate the intrinsic dimension
        center: whether to subtract the mean of the data from all points.
        smooth: whether to use the neighbor-smoother (averaging over all eps-neighbors), in which case the smooth_eps-keyword has to be given.
        PLANNED: project=True: whether to project down one dimension in the estimation to prevent concentration of measure issues.
        rounded: whether the estimation is allowed to be any floating number or whether it should be rounded to the next integer.
    returns: d (scalar) - estimation of the intrinsic dimension. Can be continuous number, depending on the rounded-parameter.
    '''

    # Init
    neighbors_per_pt = 2 # possible spot to improve
    n_pts = data.shape[0]
    dim_extr = data.shape[1]

    # Allocate memory
    cosigns = np.zeros(n_pts)

    # Smooth noise out, if requested
    if smooth:
        eps = kwargs.get('smooth_eps')
        assert eps is not None, "specify smooth_eps-parameter"
        data = neighbor_smoother(data, compute_eps_nn=True, eps=eps)

    # Center data, if requested
    if center:
        data = data - np.mean(data, axis=0)

    # Do 2-nearest-neighbor
    neighbor_obj = NearestNeighbors(n_neighbors=neighbors_per_pt + 1, algorithm='auto') # +1: account for point itself
    neighbor_obj.fit(data)
    graph = neighbor_obj.kneighbors_graph(data) - sps.identity(n_pts) # only care about two nearest neighbors, not point itself

    # Sample heuristic from every point
    for i in range(n_pts):

        # Project neighbor vectors onto plane orthogonal to v (and normalize) to prevent concentration of measure issues
        v = data[i, :]
        x1 = data[graph.indices[2*i], :]
        x2 = data[graph.indices[2*i + 1], :]
        vv = v.dot(v)
        x1_proj = x1 - x1.dot(v)/vv * v
        x1_proj = sk_normalize(x1_proj.astype(float).reshape(1, -1), norm='l2', copy=True)
        x2_proj = x2 - x2.dot(v)/vv * v
        x2_proj = sk_normalize(x2_proj.astype(float).reshape(1, -1), norm='l2', copy=True)

        # Calculate cosign (= scalar prduct) of normalized vectors
        cosigns[i] = x1_proj.dot(x2_proj.T)

    # Calculate sample variance of cosigns
    variance = np.var(cosigns, ddof=1)

    # The inverse is the estimation. This will be wrong by one dimension, when dataset was already full-dimensional!
    result = 1/variance
    if rounded:
        result = np.round(result).astype(int)

    return result

def neighbor_smoother(data, compute_eps_nn=True, **kwargs):
    '''
    Smoothes the data by calculating the mean of each data point from itself and its neighbors.

    Parameters:
        data: (n, D)-shaped array of data to smooth
        compute_eps_nn: whether to compute the necessary epsilon-nearest-neighbor-graph by the routine, in which
            case the 'eps'-keyword has to be passed, or it is passed by the user, in which case the 'adj_mat'-keyword
            has to be passed
        kwargs:
            'eps': radius of the eps-nearest-neighbor-algorithm
            'adj_mat': sparse adjacency matrix of eps-nearest-neighbor-graph of the data
    returns: data - (n, D)-shaped array of smoothed data
    '''

    # Get keywords
    eps = kwargs.get('eps')
    adj_mat = kwargs.get('adj_mat')

    if compute_eps_nn:
        assert eps is not None, "Specify eps-parameter"
        neighbor_obj = NearestNeighbors(radius=eps, algorithm='auto')
        neighbor_obj.fit(data)
        # adj_mat = neighbor_obj.radius_neighbors_graph(mode='distance') + sps.identity(data.shape[0])
        adj_mat = neighbor_obj.radius_neighbors_graph(mode='connectivity') + sps.identity(data.shape[0])
    else:
        assert adj_mat is not None, "Specify adj_mat-parameter"

    # Calculate, how many neighbors (including itself) each data point has, and compute normalizing diagonal matrix
    normalizer = sps.spdiags(np.power(np.sum(adj_mat.toarray(), axis=0), -1), 0, data.shape[0], data.shape[0])

    # Calculate mean
    return normalizer.dot(adj_mat.dot(data))
