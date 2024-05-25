from sklearn.utils.validation import check_array

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_chunked
from sklearn.linear_model import LinearRegression
from .._commonfuncs import get_nn, GlobalEstimator


class Gride(GlobalEstimator):
    """Intrinsic dimension estimation using the GRIDE algorithm"""
    
    def __init__(scale=8, d0=0.001, d1=1000, eps=1e-7, range_max=64):
        self.scale = scale
        self.range_max = range_max
        self.d0 = d0
        self.d1 = d1
        self.eps = eps
    
    def _fit(self, X):
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)
        self.dimension_, self.x_, self.y_ = self._gride(X)
        self.is_fitted_ = True
        return self
    
    def transform_multiscale(self, X=None):
        self.check_is_fitted("is_fitted_")
        return self.dimension_array_
        