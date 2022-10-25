import numpy as np
import pandas as pd
import sklearn
import scipy
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
try:
    from kneed import KneeLocator
except:
    pass

def draw_vector(v0, v1, ax=None,c='k',text=''):
    ax = ax or plt.gca()
    ax.annotate('', v1, v0, arrowprops=dict(arrowstyle='->',
                                            linewidth=2, 
                                            shrinkA=0, shrinkB=0,color=c))
    ax.annotate(text, v1, v1+v1*.02, horizontalalignment="center")

def kmeans_centroids(X,n_clusters=None):
    ''' Downsample data by finding n_clusters points closest to kmeans centroids
    sel_idx: index of selected representative points. Might be less than n_clusters if no points are assigned to some clusters 
    sel_part: points assignment to closest representative point, i.e., sel_part[1]=2 means X[0] is assigned to X[sel_idx[2]])
    '''
    if n_clusters is None:
        n_clusters = min(int(len(X)*.1),5000)
        
    centroids = sklearn.cluster.KMeans(n_clusters=n_clusters).fit(X).cluster_centers_
    cpart,dis = scipy.cluster.vq.vq(X,centroids)
    upart = np.unique(cpart)
    ppart = np.zeros(n_clusters,dtype=int)
    for p in (upart):
        _ix = np.where(cpart==p)[0]
        if len(_ix):
            ppart[p] = _ix[dis[_ix].argmin()]
        else: 
            ppart[p] = -1

    sel_idx = [i for i in ppart if i != -1]
    sel_part = scipy.cluster.vq.vq(X,X[sel_idx])[0]
    return sel_idx,sel_part

def upsample_distance_matrix_pdist(sD,part):
    triu_ix = np.array(np.triu_indices(len(sD),1)).T
    pdist = np.zeros(int(len(part)*(len(part)-1)/2))
    for i,j in triu_ix:
        rows = np.where(part==i)[0]
        cols = np.where(part==j)[0]
        flat_idx = np.ravel_multi_index(np.ix_(rows,cols),(len(part),len(part)))
        pdist[flat_idx] = sD[i,j]
    return pdist

def upsample_distance_matrix_squareform(sD,part):
    
    triu_ix = np.array(np.triu_indices(len(sD),1)).T
    D = scipy.sparse.lil_matrix((len(part),len(part)))
    for i,j in triu_ix:
        rows = part==i
        cols = part==j

        D[np.ix_(rows,cols)] = D[np.ix_(cols,rows)] = sD[i,j]
    return D

def upsample_labels(labels,part):
    labels_all = np.zeros(len(part),dtype=int)
    upart = np.unique(part)
    for p in upart:
        labels_all[part==p]=labels[p]
    return labels_all

def find_orth(O):
    ''' 
    Find and concatenate a unit vector orthogonal 
    to all column vectors of provided subspace
    '''
    O_n_plus_1 = np.vstack((O,np.zeros(O.shape[1])))
    rand_vec = np.random.rand(O_n_plus_1.shape[0], 1)
    A = np.hstack((O_n_plus_1, rand_vec))
    b = np.zeros(O_n_plus_1.shape[1] + 1)
    b[-1] = 1
    orth_vec = np.linalg.lstsq(A.T, b)[0]
    orth_vec /= np.linalg.norm(orth_vec)
    A[:,-1] = orth_vec
    return A

def graff2grass(subspace,mean):
    '''Graff(k,n) to Gr(k+1,n+1)

    Returns
    -------
    subspace_n_plus_1: np.array
        Subspace in R^n+1 with mean added to the last basis vector
    grassmann_basis: np.array
        Gr(k+1,n+1) basis corresponding to the input Graff(k,n) basis
    '''
    k = subspace.shape[1]
    # Append new vector orthogonal to subspace basis (n->n+1, k->k+1)
    subspace_n_plus_1 = find_orth(subspace)
    # Add mean to shift the new basis vector
    subspace_n_plus_1[:,-1] += np.append(mean,0.)
    u,s,vh=np.linalg.svd(subspace_n_plus_1)
    grassmann_basis = u[:,:k+1]
    return subspace_n_plus_1, grassmann_basis


def refine_clustering(labels,knnidx,k=20,min_size=0,smooth=True):
    
    clus_min_size = labels.copy()
    if min_size > 0:
        ulabels = np.unique(labels)
        counts = np.bincount(labels,minlength=ulabels.max()+1)
        for i,ix in enumerate(labels):
            xi = np.where(labels==ix)[0]
            if len(xi)<min_size:
                count=np.bincount(labels[knnidx[i][:k]],minlength=ulabels.max()+1)
                count[counts<min_size] = -1 #ignore bad labels
                clus_min_size[xi]=count.argmax()
        clus_min_size=np.array(pd.get_dummies(clus_min_size)).argmax(1)
        
    clus_smooth = clus_min_size.copy()
    if smooth:
        for i in range(len(labels)):
            clus_smooth[i]=np.bincount(clus_min_size[knnidx[i][:k]]).argmax()
            
    return clus_smooth

class AgglomerativeClustering:
    import fastcluster
    def __init__(self,method="complete"):
        self.method = method
        
    def link(self,pdist):
        self.Z = fastcluster.linkage(pdist,method=self.method,metric='precomputed',preserve_input=True)
        return self
    
    def cluster(self,n_clusters=None,distance=None):
        if (   (distance is None     and n_clusters is None)
            or (distance is not None and n_clusters is not None)):
                raise ValueError('Specify one of n_clusters or distance ')
                
        if distance is not None:
            clus = scipy.cluster.hierarchy.fcluster(self.Z,distance,criterion="distance")
        elif n_clusters is not None:
            children, n_leaves = self.Z[:,:2].astype(int), int(self.Z[-1,-1])
            clus = sklearn.cluster._agglomerative._hc_cut(n_clusters,children,n_leaves)
        return clus
    
    
def nested_basis_to_variables(basis_vars,basis_nested):
    ''' Given a basis expressed in terms of another basis' vectors (e.g., PCA components),
    returns it expressed in terms of the basis' variables
    
    Parameters
    ----------
    basis_vars: 
        Basis expressed in terms of original variables
    basis_nested:
        Basis expressed in terms of basis_vars components
        
    Returns
    ------
    newbasis:
        basis_nested expressed in terms of original variables
    '''
    ### X=np.random.random((500,4))
    ### 
    ### pca=sklearn.decomposition.PCA(4).fit(X)
    ### Xpca=pca.transform(X)
    ### 
    ### pca2=sklearn.decomposition.PCA(2).fit(Xpca[:30])
    ### Xpca2=pca2.transform(Xpca[:30])
    ### 
    ### pca3=sklearn.decomposition.PCA(2).fit(X[:30])
    ### Xpca3=pca3.transform(X[:30])
    ### 
    ### plt.scatter(*Xpca[:,:2].T)
    ### plt.scatter(*Xpca[:30,:2].T)
    ### plt.scatter(*Xpca2[:,:2].T)
    ### plt.scatter(*Xpca3[:,:2].T)
    ### 
    ### comp1 = pca.components_.T
    ### comp2 = pca2.components_.T
    ### 
    ### comp2to1 = nested_basis_to_variables(comp1,comp2)
    ### 
    ### assert np.allclose(pca3.components_.T,comp2to1)
    newbasis = np.zeros((len(basis_vars),basis_nested.shape[1]))
    for ix in range(basis_nested.shape[1]):
        newbasis[:,ix] += np.dot(basis_vars,basis_nested[:,ix])
    return newbasis

def hellinger(p, q): 
    return np.linalg.norm(np.sqrt(p) - np.sqrt(q))/np.sqrt(2)

def find_best_cut(D,children,n_leaves,max_clusters=30,plot=True,figsize=(10,4)):
    from kneed import KneeLocator
    
    scan_range = range(1,max_clusters)
    costs = []
    for n_clusters in scan_range:
        clus = sklearn.cluster._agglomerative._hc_cut(n_clusters,children,n_leaves)
        _cost = []
        for c in np.unique(clus):
            Dc = D[np.ix_(clus==c,clus==c)]
            _cost.append(Dc[np.triu_indices_from(Dc)].sum())
        costs.append(np.mean(_cost))

    kneedle = KneeLocator(scan_range, costs, S=1.0,
                          curve="convex",  direction="decreasing") 
    knee = kneedle.find_knee()[0]
    if plot:
        plt.figure(figsize=figsize)
        plt.plot(scan_range,costs,marker='o')
        plt.xticks(scan_range)
        plt.axvline(knee,c='r',linestyle='--')
        plt.show()
    return knee