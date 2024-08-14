import numpy as np
from numpy.linalg import solve

from scipy.spatial.distance import pdist, squareform
from sklearn.utils.validation import check_array
from sklearn.linear_model import LinearRegression

from .._commonfuncs import GlobalEstimator


class Mag(GlobalEstimator):
    """Intrinsic dimension estimation using the Magnitude Dimension. 

    Parameters
    ----------  
    ts: list or one-dim array of positive floats
        Range of scale parameters
    metric: str
        scipy.spatial.distance metric parameter
   

    Attributes
    ----------

    y_: 1d array 
        np.array with the log(magnitude(t)) values.
    reg_: sklearn.linear_model.LinearRegression
        Regression object used to fit line to log tM vs log t
    """
    def __init__(self, ts = np.linspace(0.5,5.5,21), metric = 'euclidean'):
        self.ts = ts
        self.metric = metric 

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self: object
            Returns self.
        self.dimension_: float
            The estimated intrinsic dimension
        self.reg_: sklearn.linear_model.LinearRegression
            Regression object used to fit line to log tM vs log t
        """

        if np.any(self.ts <= 0):
            raise ValueError(
                    "Scale parameters need to be positive."
                )
        
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

        self.dimension_, self.reg_ = self._MagDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _MagDimEst(self, X):

        D = squareform(pdist(X))
        y = [self._mag(D, t) for t in self.ts]
        self.y_ = np.log(y)

        reg = LinearRegression(fit_intercept = True).fit(np.log(self.ts).reshape(-1, 1), self.y_)
        return reg.coef_[0], reg
    
    @staticmethod
    def _mag(D, t):
        Z = np.exp(-t * D)
        w = solve(Z, np.ones(D.shape[0]))
        return np.sum(w)