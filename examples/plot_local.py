"""
===========================
Local ID example
===========================
"""
from skdim import local_id, asPointwise
import numpy as np

X = np.random.random((1000,10))


#one neighborhood
ess = local_id.ESS().fit(X)
fishers = local_id.FisherS().fit(X)
mom = local_id.MOM().fit(X)
mind_ml = local_id.MiND_ML().fit(X)
tle = local_id.TLE().fit(X)
lpca = local_id.lPCA().fit(X)


#all datapoint neighborhoods
pw_id = asPointwise(X,local_id.FisherS().fit,n_neighbors=100,n_jobs=1)


