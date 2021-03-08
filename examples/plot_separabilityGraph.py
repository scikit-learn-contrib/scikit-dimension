"""
===========================
FisherS separability graph example
===========================
"""

import skdim
import plotly.graph_objects as go

# generate data (n_samples x n_dimensions)
data, clusters = skdim.datasets.lineDiskBall(n = 2000, random_state = 0)

# compute local ID
pca = skdim.id.lPCA()
local_id = pca.fit_predict_pw(data,n_neighbors=100)

# plot
fig = go.Figure(
    go.Scatter3d( 
        dict(zip(['x','y','z'],data.T[:3]) ), 
        mode='markers',
        marker=dict( size=3, color=local_id, colorbar=dict(thickness=20,x=.95))))
fig.show()



import skdim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.stats.mstats import winsorize

ball1 = skdim.datasets.hyperBall(n=1000, d=3, radius=0.5, center=[0, 0, 0]).T
ball2 = skdim.datasets.hyperBall(n=1000, d=6, radius=0.5, center=[1, 0, 0, 0, 0, 0]).T

_2balls = np.zeros((6, 2000))
_2balls[:3, :1000] = ball1
_2balls[:, 1000:2000] = ball2
X = _2balls.T

u = PCA().fit_transform(X)
fishers = skdim.id.FisherS(conditional_number=10000).fit(X)
ns = fishers.point_inseparability_to_pointID()[0]
edges, weights = fishers.getSeparabilityGraph()

nsw = winsorize(ns, limits=(0.01, 0.01), inclusive=(True, True))
plt.figure(figsize=(10, 5))
plt.scatter(u[:, 0], u[:, 1], c=nsw)
fishers.plotSeparabilityGraph(u[:, 0], u[:, 1], edges, alpha=0.2)
plt.colorbar()
plt.axis("equal")
plt.title("PCA on original data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

for i in range(3):
    up = fishers.Xp_[:, i : i + 2]
    plt.figure(figsize=(10, 5))
    plt.scatter(up[:, 0], up[:, 1], c=nsw)
    fishers.plotSeparabilityGraph(up[:, 0], up[:, 1], edges, alpha=0.2)
    plt.colorbar()
    plt.axis("equal")
    plt.title("Pre-processed data")
    plt.xlabel("Axis " + str(i + 1))
    plt.ylabel("Axis " + str(i + 2))
    plt.show()
