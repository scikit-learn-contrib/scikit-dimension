import numpy as np
from sklearn.metrics import pairwise_distances_chunked
from ._commonfuncs import get_nn

def corint(x, k1 = 10, k2 = 20, DM = None):
    
    n_elements = len(x)**2 #number of elements
    
    dists, _ = get_nn(x,k2)
    
    if DM is None:
        chunked_distmat = pairwise_distances_chunked(x)
    else:
        chunked_distmat = DM
        
    r1 = np.median(dists[:, k1-1])
    r2 = np.median(dists[:, -1])
    
    
    n_diagonal_entries = len(x) #remove diagonal from sum count
    s1=-n_diagonal_entries
    s2=-n_diagonal_entries
    for chunk in chunked_distmat:
        s1+=(chunk<r1).sum()
        s2+=(chunk<r2).sum()


    Cr = np.array([s1/n_elements, s2/n_elements])
    estq = np.diff(np.log(Cr))/np.log(r2/r1)
    return(estq)
