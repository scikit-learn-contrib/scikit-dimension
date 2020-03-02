import numpy as np
from scipy.spatial.distance import pdist, squareform


def mada(x, k = None, comb = "average", DM = False, local = False, 
    maxDim = 5):
     
    if (DM == False):
        distmat = squareform(pdist(x))
     
    else:
        distmat = x

    n = len(distmat)
    if k is None:  
        k = int(np.floor(2 * np.log(n)))
     
    if (local == False):  
        ID = np.random.choice(n, size=int(np.round(n/2)), replace = False)
        tmpD = distmat[ID,:]
        tmpD[tmpD == 0] = np.max(tmpD)
     
    else:
        tmpD = distmat
        tmpD[tmpD == 0] = np.max(tmpD)
    
    sortedD = np.sort(tmpD,axis=0,kind='mergesort')
    RK = sortedD[k-1,:]
    RK2 = sortedD[int(np.floor(k/2)-1), ]
    ests = np.log(2)/np.log(RK/RK2)
    if (local == True):  
        return(ests)