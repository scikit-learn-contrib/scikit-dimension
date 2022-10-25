import numpy as np

def flag_mean(subspaces):
    ''' Flag mean of subspaces

    Code taken from "Finding the subspace mean or median to fit your need."
    T. Marrinan, J. R. Beveridge, B. Draper, M. Kirby, and C. Peterson.
    Proc. IEEE Conference on Computer Vision and Pattern Recognition (CVPR), (2014): 1082-1089.

    Parameters
    ----------
    subspaces: list[np.array]
        Collection of subspaces of shape (n x rank[i]) to average

    Returns
    -------
    basis: np.array
       Flag mean basis
    s: np.array
       Flag mean singular values
    '''

    A_prime = np.concatenate(subspaces,axis=1)
    flip = A_prime.shape
    if flip[0] >= flip[1]:
        u, s, vt = np.linalg.svd(A_prime,0)
        mu = u
    else:
        v, s, uT = np.linalg.svd(A_prime.T,0)
        mu = uT.T

    return mu, s



def irls_flag(subspaces, r, n_its=100, sin_cos='sine', opt_err = 'sine', init = 'random', seed = 0): 
    ''' Code taken from https://github.com/nmank/Flag_Median_and_FlagIRLS
    Use FlagIRLS on subspaces to output a representative for a point in Gr(r,n) 
    which solves the input objection function
    Repeats until iterations = n_its or until objective function values of consecutive
    iterates are within 0.0000000001 and are decreasing for every algorithm (except increasing for maximum cosine)
    Inputs:
        subspaces - list of numpy arrays representing points on Gr(k_i,n)
        r - the number of columns in the output
        n_its - number of iterations for the algorithm
        sin_cos - a string defining the objective function for FlagIRLS
                    'sine' = flag median
        opt_err - string for objective function values in err (same options as sin_cos)
        init - string 'random' for random initlalization. 
            otherwise input a numpy array for the inital point
        seed - seed for random initialization, for reproducibility of results
    Outputs:
        Y - a numpy array representing the solution to the chosen optimization algorithm
        err - a list of the objective function values at each iteration (objective function chosen by opt_err)
    '''
    err = []
    n = subspaces[0].shape[0]


    #initialize
    if init == 'random':
        #randomly
        np.random.seed(seed)
        Y_raw = np.random.rand(n,r)-.5
        Y = np.linalg.qr(Y_raw)[0][:,:r]
    elif init == 'subspaces':
        np.random.seed(seed)
        Y = subspaces[np.random.randint(len(subspaces))]
    else:
        Y = init

    err.append(_calc_error_1_2(subspaces, Y, opt_err))

    #flag mean iteration function
    #uncomment the commented lines and 
    #comment others to change convergence criteria
    
    itr = 1
    diff = 1
    while itr <= n_its and diff > 0.0000000001:
        Y0 = Y.copy()
        Y = _flag_mean_iteration(subspaces, Y, sin_cos)
        err.append(_calc_error_1_2(subspaces, Y, opt_err))
        diff  = err[itr-1] - err[itr]
           
        itr+=1
    


    if diff > 0:
        return Y, err
    else:
        return Y0, err[:-1]
    
def _flag_mean_iteration(subspaces, Y0, weight, eps = .0000001):
    ''' Code taken from https://github.com/nmank/Flag_Median_and_FlagIRLS
    Calculates a weighted Flag Mean of data using a weight method for FlagIRLS
    eps = .0000001 for paper examples
    Inputs:
        subspaces - list of numpy arrays representing points on Gr(k_i,n)
        Y0 - a numpy array representing a point on Gr(r,n)
        weight - a string defining the objective function
                    'sine' = flag median
                    'sinsq' = flag mean
        eps - a small perturbation to the weights to avoid dividing by zero
    Outputs:
        Y- the weighted flag mean
    '''
    r = Y0.shape[1]
    
    aX = []
    al = []

    ii=0

    for x in subspaces:
        if weight == 'sine':
            m = np.min([r,x.shape[1]])
            sinsq = m - np.trace(Y0.T @ x @ x.T @ Y0)
            al.append((sinsq+eps)**(-1/4))
        else:
            print('unrecognized weight')
        aX.append(al[-1]*x)
        ii+= 1

    Y = flag_mean(aX)[0][:,:r]

    return Y
    
def _calc_error_1_2(subspaces, Y, sin_cos):
    ''' Code taken from https://github.com/nmank/Flag_Median_and_FlagIRLS
    Calculate objective function value. 
    Inputs:
        subspaces - a list of numpy arrays representing points in Gr(k_i,n)
        Y - a numpy array representing a point on Gr(r,n) 
        sin_cos - a string defining the objective function
                    'sine' = Sine Median
                    'sinsq' = Flag Mean
    Outputs:
        err - objective function value
    '''
    k = Y.shape[1]
    err = 0
    if sin_cos == 'sine':
        for x in subspaces:
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y.T @ x @ x.T @ Y)
            #fixes numerical errors
            if sin_sq < 0:
                sin_sq = 0
            err += np.sqrt(sin_sq)
    elif sin_cos == 'sinesq':
        for x in subspaces:
            r = np.min([k,x.shape[1]])
            sin_sq = r - np.trace(Y.T @ x @ x.T @ Y)
            #fixes numerical errors
            if sin_sq < 0:
                sin_sq = 0
            err += sin_sq
    return err