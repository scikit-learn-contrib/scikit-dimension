"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`skltemplate.template.TemplateEstimator`
"""
from skdim import global_id
import numpy as np
from matplotlib import pyplot as plt

X=np.random.random((1000,10))

res = global_id.CorrInt().fit(X).dimension_
res = global_id.DANCo().fit(X).dimension_
res = global_id.KNN().fit(X).dimension_
res = global_id.Mada().fit(X).dimension_
res = global_id.MLE().fit(X).dimension_
res = global_id.TwoNN().fit(X).dimension_