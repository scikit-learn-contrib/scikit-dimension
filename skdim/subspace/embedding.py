import numpy as np
import numba as nb

@nb.njit
def _triu_to_flat(X):
    m = len(X)
    flat = []
    for i in range(m):
        for j in range(i,m):
            flat.append(X[i,j])
    return np.array(flat)

@nb.njit
def _embed_conway(subspace):
    proj = subspace @ subspace.T
    proj_triu_flat = _triu_to_flat(proj)
    return proj_triu_flat

def embed_conway(subspace):
    m = subspace.shape[0]
    d = subspace.shape[1]
    radius = np.sqrt(d*(m - d)/m)
    dim = int((m-1)*(m+2)/2)
    proj = _embed_conway(subspace)
    return proj,m,d,radius,dim

def unembed_conway(proj):
    #Given an embedding of a subspace p,return the projection matrix
    dim = int((np.sqrt(8*len(proj)+1)-1)/2)
    P = np.zeros((dim,dim))
    P[np.triu_indices_from(P)]=proj
    P += P.T - np.diag(np.diag(P))
    return P

@nb.njit
def embed_conway_loop(subspaces,normalize=True):
    
    m = subspaces[0].shape[0]
    n = len(subspaces)
    Im = np.diag(np.ones(m)/2)
    sphere_all_center = _triu_to_flat(Im)
    
    projs = np.zeros((n,len(sphere_all_center)))

    for i,subspace in enumerate(subspaces):
        projs[i] = _embed_conway(subspace)

    projs -= sphere_all_center
    if normalize:
        projs /= np.sqrt(np.sum(projs**2,axis=1)).reshape(-1,1)
    return projs, sphere_all_center

