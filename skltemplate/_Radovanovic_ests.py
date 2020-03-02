import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform


def get_nn_radovanovic(X, k, n_jobs=1):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    dists, inds = neigh.kneighbors(return_distance=True)
    return dists.T, inds.T


def idmle(dists):
    # dists - nearest-neighbor distances (k x 1, or k x n), sorted
    # equivalent to R intdim.maxLikPointwiseDimEst with no kwargs
    k = len(dists)
    xi = np.sum(np.log(dists / dists[[k-1], :]), axis=0) / (k-1)
    ID = -(1 / xi)
    return ID


def idmom(dists):
    # dists - nearest-neighbor distances (k x 1, or k x n), sorted
    k = len(dists)
    w = dists[k-1, :]
    m1 = np.sum(dists, axis=0)/k
    ID = -m1 / (m1-w)
    return ID


def idpca(X, theta):
    # X - data set (n x d)
    # theta - ratio of variance to preserve (theta \in [0,1])
    pca = PCA().fit(X)
    explained = pca.explained_variance_ratio_
    d = X.shape[1]
    theta = theta*100
    ID = 1
    sumexp = explained[ID-1]
    while ID < d and sumexp < theta:
        ID += 1
    sumexp = sumexp + explained[ID-1]
    return ID


def idmind_ml1(dists):
    n = dists.shape[1]
    dists2 = dists**2  # need only squared dists to first 2 neighbors
    s = np.sum(np.log(dists2[0, :]/dists2[1, :]), axis=0)
    ID = -2/(s/n)
    return ID


def idmind_mli(dists, D):
    k = len(dists)
    n = dists.shape[1]
    nlogk = n*np.log(k)
    rho = dists[0, :]/dists[k-1, :]
    ll = np.zeros(D)
    sum1 = np.sum(np.log(rho))
    for d in range(1, D+1):
        sum2 = np.sum(np.log(1-rho**d))
        ll[d-1] = nlogk + n*np.log(d) + (d-1)*sum1 + (k-1)*sum2
    ID = np.argmax(ll)+1
    return ID


def idtle(X, dists, epsilon=0.0001):

    # X - matrix of nearest neighbors (k x d), sorted by distance
    # dists - nearest-neighbor distances (k x 1), sorted
    dists = dists.T
    r = dists[0, -1]  # distance to k-th neighbor

    # Boundary case 1: If $r = 0$, this is fatal, since the neighborhood would be degenerate.
    if r == 0:
        raise ValueError('All k-NN distances are zero!')
    # Main computation
    k = dists.shape[1]
    V = squareform(pdist(X))
    Di = np.tile(dists.T, (1, k))
    Dj = Di.T
    Z2 = 2*Di**2 + 2*Dj**2 - V**2
    S = r * (((Di**2 + V**2 - Dj**2)**2 + 4*V**2 * (r**2 - Di**2))
             ** 0.5 - (Di**2 + V**2 - Dj**2)) / (2*(r**2 - Di**2))
    T = r * (((Di**2 + Z2 - Dj**2)**2 + 4*Z2 * (r**2 - Di**2))
             ** 0.5 - (Di**2 + Z2 - Dj**2)) / (2*(r**2 - Di**2))
    Dr = (dists == r).squeeze()  # handle case of repeating k-NN distances
    S[Dr, :] = r * V[Dr, :]**2 / (r**2 + V[Dr, :]**2 - Dj[Dr, :]**2)
    T[Dr, :] = r * Z2[Dr, :] / (r**2 + Z2[Dr, :] - Dj[Dr, :]**2)
    # Boundary case 2: If $u_i = 0$, then for all $1\leq j\leq k$ the measurements $s_{ij}$ and $t_{ij}$ reduce to $u_j$.
    Di0 = (Di == 0).squeeze()
    T[Di0] = Dj[Di0]
    S[Di0] = Dj[Di0]
    # Boundary case 3: If $u_j = 0$, then for all $1\leq j\leq k$ the measurements $s_{ij}$ and $t_{ij}$ reduce to $\frac{r v_{ij}}{r + v_{ij}}$.
    Dj0 = (Dj == 0).squeeze()
    T[Dj0] = r * V[Dj0] / (r + V[Dj0])
    S[Dj0] = r * V[Dj0] / (r + V[Dj0])
    # Boundary case 4: If $v_{ij} = 0$, then the measurement $s_{ij}$ is zero and must be dropped. The measurement $t_{ij}$ should be dropped as well.
    V0 = (V == 0).squeeze()
    np.fill_diagonal(V0, False)
    T[V0] = r  # by setting to r, $t_{ij}$ will not contribute to the sum s1t
    S[V0] = r  # by setting to r, $s_{ij}$ will not contribute to the sum s1s
    # will subtract twice this number during ID computation below
    nV0 = np.sum(V0)
    # Drop T & S measurements below epsilon (V4: If $s_{ij}$ is thrown out, then for the sake of balance, $t_{ij}$ should be thrown out as well (or vice versa).)
    TSeps = (T < epsilon) | (S < epsilon)
    np.fill_diagonal(TSeps, 0)
    nTSeps = np.sum(TSeps)
    T[TSeps] = r
    T = np.log(T/r)
    S[TSeps] = r
    S = np.log(S/r)
    np.fill_diagonal(T, 0)  # delete diagonal elements
    np.fill_diagonal(S, 0)
    # Sum over the whole matrices
    s1t = np.sum(T)
    s1s = np.sum(S)
    # Drop distances below epsilon and compute sum
    Deps = dists < epsilon
    nDeps = np.sum(Deps, dtype=int)
    dists = dists[nDeps:]
    s2 = np.sum(np.log(dists/r))
    # Compute ID, subtracting numbers of dropped measurements
    ID = -2*(k**2-nTSeps-nDeps-nV0) / (s1t+s1s+2*s2)
    return ID
