import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammainc
from scipy.linalg import cholesky
from scipy.special import lambertw
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize
import skdim


def randsphere(n_points,ndim,radius,center = []):
    if center == []:
        center = np.array([0]*ndim)
    r = radius
    x = np.random.normal(size=(n_points, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    frtiled = np.tile(fr.reshape(n_points,1),(1,ndim))
    p = center + np.multiply(x,frtiled)
    return p, center

sphere1 = randsphere(n_points=1000,ndim=3,radius=.5,center = [0,0,0])[0].T
sphere2 = randsphere(n_points=1000,ndim=6,radius=.5,center = [1,0,0,0,0,0])[0].T
_2spheres = np.zeros((6,2000))
_2spheres[0:3,0:1000] = sphere1
_2spheres[:,1000:2000] = sphere2
X = _2spheres.T
u = PCA().fit_transform(X)
fishers = skdim.lid.FisherS(conditional_number=10000).fit(X)
ns = fishers.point_inseparability_to_pointID()[0]
edges, weights = fishers.getSeparabilityGraph()

nsw = winsorize(ns, limits=(0.01,0.01), inclusive=(True, True))
plt.figure(figsize=(10,5))
plt.scatter(u[:,0],u[:,1],c=nsw)
fishers.plotSeparabilityGraph(u[:,0],u[:,1],edges,alpha=0.2)
plt.colorbar()
plt.axis('equal')
plt.title('PCA on original data')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

for i in range(3):
    up = fishers.Xp_[:,i:i+2]
    plt.figure(figsize=(10,5))
    plt.scatter(up[:,0],up[:,1],c=nsw)
    fishers.plotSeparabilityGraph(up[:,0],up[:,1],edges,alpha=0.2)
    plt.colorbar()
    plt.axis('equal')
    plt.title('Pre-processed data')
    plt.xlabel('Axis '+str(i+1))
    plt.ylabel('Axis '+str(i+2))
    plt.show()

