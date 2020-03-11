"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
from skdim import local_id
import numpy as np
from matplotlib import pyplot as plt

X = np.random.random((200,10))


#one neighborhood
res = local_id.ESS().fit(X)
res = local_id.FisherS().fit(X)
res = local_id.MOM().fit(X)
res = local_id.MiND_ML().fit(X)
res = local_id.TLE().fit(X)
res = local_id.lPCA().fit(X)


#all datapoint neighborhoods
pw_id = skdim.commonfuncs.asPointwise(X,local_id.FisherS().fit,n_neighbors=200,n_jobs=1)
