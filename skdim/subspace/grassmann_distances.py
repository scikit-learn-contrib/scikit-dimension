import itertools
import numpy as np
import numba as nb
import pandas as pd
try:
    import jax
    import jax.numpy as jnp
except:
    pass
from functools import partial

def grassmann_distance_matrix(subspaces,ij=None,metric='geodesic',contrast_unequal=True,gpu=True,max_nbytes = 1e8,max_rank=None,return_angles=True):

    dist_func = DISTANCE_FUNCTIONS[metric]
    xp = jnp if gpu else np
    
    
    n = len(subspaces)
    length = len(subspaces[0])
    ranks = np.array([subs.shape[1] for subs in subspaces])
    if max_rank is None:
        max_rank = int(xp.max(ranks))
    elif max_rank > int(xp.max(ranks)):
        raise ValueError('max_rank must be <= to the maximum rank of provided matrices')
    else:
        ranks = np.clip(ranks,None,max_rank)
    dtype_size = subspaces[0].dtype.itemsize
    blocksize = int((max_nbytes*dtype_size)/(length*max_rank*dtype_size))

    # pad input subspaces to concatenate them and vectorize computations
    subspaces_pad = np.zeros((n,length,max_rank))
    for r0,rank in enumerate(ranks):
        subspaces_pad[r0,:,:rank] = subspaces[r0][:,:rank]
    subspaces_pad = xp.array(subspaces_pad)
    subspaces_padT = xp.transpose(subspaces_pad, axes=(0,2,1))

    # loop over blocks of indices
    if ij is not None:
        i, j = ij[0], ij[1]
    else:
        i, j = np.triu_indices(n, k=1)

    pdist = np.full(len(i),np.nan)
    thetas = []     
    
    print('Computing distances... 0%',end='\r')
    for start in range(0, len(pdist), blocksize):
        # Define last element for calculation
        end = min(start + blocksize, len(pdist)) 

        AT = subspaces_padT[i[start:end]]
        B = subspaces_pad[j[start:end]]

        if gpu:
            theta = np.asarray(_principal_angles_jax(AT,B))
        else:
            theta =_principal_angles(AT,B)

        if return_angles:
            thetas.append(theta)

        pdist[start:end] = _reduce_to_rank_and_get_dist(theta,i[start:end],j[start:end],ranks,contrast_unequal,dist_func)
        print(f"Computing distances... {end/len(pdist):.0%}",end='\r')
    if return_angles:
        return pdist, thetas
    else:
        return pdist

def grassmann_distance_matrix_precomputed_angles(subspaces,thetas,ij=None,metric='geodesic',gpu=True,contrast_unequal=True,max_nbytes = 1e8,max_rank=None):

    dist_func = DISTANCE_FUNCTIONS[metric]
    xp = jnp if gpu else np
    
    
    n = len(subspaces)
    length = len(subspaces[0])
    ranks = np.array([subs.shape[1] for subs in subspaces])
    if max_rank is None:
        max_rank = int(xp.max(ranks))
    elif max_rank > int(xp.max(ranks)):
        raise ValueError('max_rank must be <= to the maximum rank of provided matrices')
    else:
        ranks = np.clip(ranks,None,max_rank)
    dtype_size = subspaces[0].dtype.itemsize
    blocksize = int((max_nbytes*dtype_size)/(length*max_rank*dtype_size))

    # loop over blocks of indices
    if ij is not None:
        i, j = ij[0], ij[1]
    else:
        i, j = np.triu_indices(n, k=1)
    pdist = np.full(len(i),np.nan)
    
    print('Computing distances... 0%',end='\r')
    for start in range(0, len(pdist), blocksize):
        # Define last element for calculation
        end = min(start + blocksize, len(pdist)) 
        theta = thetas[start:end]
        pdist[start:end] = _reduce_to_rank_and_get_dist(theta,i[start:end],j[start:end],ranks,contrast_unequal,dist_func)
        print(f"Computing distances... {end/len(pdist):.0%}",end='\r')
    return pdist

def conway_distance_matrix(subs_conway,ij=None,metric='geodesic_conway',gpu=True,max_nbytes = 1e8,):

    if gpu: metric+='_gpu' 
    dist_func = DISTANCE_FUNCTIONS[metric]

    n = len(subs_conway)
    length = subs_conway.shape[1]

    dtype_size = subs_conway.dtype.itemsize
    blocksize = int((max_nbytes*dtype_size)/(length*dtype_size))

    # loop over blocks of indices
    if ij is not None:
        i, j = ij[0], ij[1]
    else:
        i, j = np.triu_indices(n, k=1)
    pdist = np.full(len(i),np.nan)
    
    print('Computing distances... 0%',end='\r')
    for start in range(0, len(pdist), blocksize):
        # Define last element for calculation
        end = min(start + blocksize, len(pdist)) 

        cA = subs_conway[i[start:end]]
        cB = subs_conway[j[start:end]]
        pdist[start:end] = dist_func(cA,cB)
        
        print(f"Computing distances... {end/len(pdist):.0%}",end='\r')

    return pdist

@nb.njit
def _reduce_to_rank_and_get_dist(theta,iblock,jblock,ranks,contrast_unequal,dist_func):

    blockdist=[]
    for p in range(len(iblock)):
        ranki, rankj = ranks[iblock[p]],ranks[jblock[p]]
        _theta = theta[p,:min(ranki,rankj)]
        
        if contrast_unequal:
            blockdist.append(dist_func(_theta,ranki,rankj))
        else:
            blockdist.append(dist_func(_theta,0,0))
    return blockdist

def _pad_and_concatenate(subspaces,shape,ranks=None):
    '''Add zeros to a list of subspaces along last axis so they can be concatenated '''
    if ranks is None:
        ranks = [s.shape[1] for s in subspaces]
    subspaces_pad = np.zeros(shape)
    for r0,rank in enumerate(ranks):
        subspaces_pad[r0,:,:rank] = subspaces[r0][:,:rank]
    return subspaces_pad

##### Principal Angles #####

@jax.jit
def _principal_angles_jax(AT,B):
    M = jnp.matmul(AT, B)
    S = jnp.linalg.svd(M, compute_uv=False)
    S = jnp.clip(S,None,1.)  
    return jnp.arccos(S)

def _principal_angles(AT,B):
    M = AT @ B
    S = np.linalg.svd(M, compute_uv=False)
    S = np.clip(S,None,1.)  
    return np.arccos(S)



##### Distance from frincipal Angles #####

@nb.njit
def asimov(theta,rank_i,rank_j):
    return np.max(theta)

@nb.njit
def asimov_reverse(theta,rank_i,rank_j):
    return np.min(theta)

@nb.njit
def binet_cauchy(theta,rank_i,rank_j):
    return np.sqrt(1 - np.prod(np.cos(theta) ** 2))

@nb.njit
def chordal(theta,rank_i,rank_j):
    contrast = abs(rank_i - rank_j) 
    return np.sqrt(contrast + np.sum(np.sin(theta) ** 2))

@nb.njit
def fubini_study(theta,rank_i,rank_j):
    return np.arccos(np.prod(np.cos(theta)))

@nb.njit
def geodesic(theta,rank_i,rank_j):
    contrast = abs(rank_i - rank_j) * np.pi ** 2 / 4
    return np.sqrt(contrast + np.sum(theta ** 2))

@nb.njit
def martin(theta,rank_i,rank_j):
    sys_float_info_min = 2.2250738585072014e-308
    cos_sq = np.cos(theta) ** 2
    cos_sq[np.where(cos_sq < sys_float_info_min)] = sys_float_info_min
    return np.sqrt(np.log(np.prod(np.reciprocal(cos_sq))))

@nb.njit
def procrustes(theta,rank_i,rank_j):
    contrast = abs(rank_i - rank_j)
    return 2 * np.sqrt(contrast + np.sum(np.sin(theta / 2) ** 2))

@nb.njit
def projection(theta,rank_i,rank_j):
    contrast = abs(rank_i - rank_j) 
    return np.sqrt(contrast + np.sum(np.sin(theta)))

@nb.njit
def spectral(theta,rank_i,rank_j):
    return 2 * np.sin(np.max(theta) / 2)



##### Distance from Conway embedding #####
def diag_indices_from_triu_flat_jax(n):
    return jnp.cumsum(jnp.append(0,jnp.arange(n,1,-1)))

@nb.njit
def diag_indices_from_triu_flat(n):
    return np.cumsum(np.append(0,np.arange(n,1,-1)))

def euclidean_conway_jax(cA,cB,abs_dot=False):
    diffs = jnp.sum((cA-cB)**2)
    if abs_dot:
        #w unit vectors cosine = 1 - sqeucl/2 ---> sqeucl > 2 = negative cosine
        idx = diffs > 2 
        diffs[idx] = jnp.sum((cA[idx]+cB[idx])**2)
    return jnp.sqrt(diffs,axis=-1)

@nb.njit
def euclidean_conway(cA,cB,abs_dot=False):
    diffs = np.sum((cA-cB)**2)
    if abs_dot:
        #w unit vectors cosine = 1 - sqeucl/2 ---> sqeucl > 2 = negative cosine
        idx = diffs > 2 
        diffs[idx] = np.sum((cA[idx]+cB[idx])**2)
    return np.sqrt(diffs,axis=-1)

def chordal_conway_jax(cA,cB):
    dim = int((jnp.sqrt(8*cA.shape[1]+1)-1)/2)
    diff = (cA-cB)**2
    diff[:,diag_indices_from_triu_flat_jax(dim)]/=2
    return jnp.sqrt(jnp.sum(2*diff,axis=-1))/jnp.sqrt(2)

@nb.njit
def chordal_conway(cA,cB):
    dim = int((np.sqrt(8*cA.shape[1]+1)-1)/2)
    diff = (cA-cB)**2
    diff[:,diag_indices_from_triu_flat(dim)]/=2
    return np.sqrt(np.sum(2*diff,axis=-1))/np.sqrt(2)

def geodesic_conway_jax(cA,cB,abs_dot=False):
    norms_A = jnp.linalg.norm(cA,axis=-1)
    norms_B = jnp.linalg.norm(cB,axis=-1)
    inner_prod = jnp.sum(jnp.multiply(cA,cB),axis=-1)
    if abs_dot:
        inner_prod = jnp.abs(inner_prod)
    cos_angle = inner_prod / (norms_A*norms_B)
    cos_angle = jnp.clip(cos_angle, -1, 1)
    return jnp.arccos(cos_angle)

@nb.njit
def geodesic_conway(cA,cB,abs_dot=False):
    norms_A = np.sqrt(np.sum(cA**2,axis=-1))
    norms_B = np.sqrt(np.sum(cB**2,axis=-1))
    inner_prod = np.sum(cA*cB,axis=-1)
    if abs_dot:
        inner_prod = np.abs(inner_prod)
    cos_angle = inner_prod / (norms_A*norms_B)
    cos_angle = np.clip(cos_angle, -1, 1)
    return np.arccos(cos_angle)

euclidean_conway_abs = partial(euclidean_conway,abs_dot=True)
euclidean_conway_abs_jax = jax.jit(partial(euclidean_conway_jax,abs_dot=True))
euclidean_conway_jax = jax.jit(euclidean_conway_jax)

geodesic_conway_abs = partial(geodesic_conway,abs_dot=True)
geodesic_conway_abs_jax = jax.jit(partial(geodesic_conway_jax,abs_dot=True))
geodesic_conway_jax = jax.jit(geodesic_conway_jax)

DISTANCE_FUNCTIONS = {
'asimov':asimov,
'asimov_reverse':asimov_reverse,
'binet_cauchy':binet_cauchy,
'chordal':chordal,
'fubini_study':fubini_study,
'geodesic':geodesic,
'martin':martin,
'procrustes':procrustes,
'projection':projection,
'spectral':spectral,
    
'euclidean_conway':euclidean_conway,
'euclidean_conway_gpu':euclidean_conway_jax,
'euclidean_conway_abs':euclidean_conway_abs,
'euclidean_conway_abs_gpu':euclidean_conway_abs_jax,
'chordal_conway':chordal_conway,
'chordal_conway_gpu':chordal_conway_jax,
'geodesic_conway':geodesic_conway,
'geodesic_conway_gpu':geodesic_conway_jax,
'geodesic_conway_abs':geodesic_conway_abs,
'geodesic_conway_abs_gpu':geodesic_conway_abs_jax,

}