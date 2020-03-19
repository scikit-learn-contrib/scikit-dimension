"""
===========================
Global ID example
===========================
"""
from skdim import global_id
import numpy as np

X=np.random.random((1000,10))

corrint = global_id.CorrInt().fit(X).dimension_
danco = global_id.DANCo().fit(X).dimension_
knn = global_id.KNN().fit(X).dimension_
mada = global_id.Mada().fit(X).dimension_
mle = global_id.MLE().fit(X).dimension_
twonn = global_id.TwoNN().fit(X).dimension_
