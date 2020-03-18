"""
===========================
Local ID example
===========================
"""
import skdim
from skdim import local_id
import numpy as np
from matplotlib import pyplot as plt

X = np.random.random((1000,10))


#one neighborhood
res = local_id.ESS().fit(X)
res = local_id.FisherS().fit(X)
res = local_id.MOM().fit(X)
res = local_id.MiND_ML().fit(X)
res = local_id.TLE().fit(X)
res = local_id.lPCA().fit(X)


#all datapoint neighborhoods
pw_id = skdim.asPointwise(X,local_id.FisherS().fit,n_neighbors=100,n_jobs=1)


