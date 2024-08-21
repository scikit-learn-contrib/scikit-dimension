#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import numpy as np
from numpy.linalg import solve
from skdim.id_flex import lPCA
from sklearn.utils.validation import check_array
from sklearn.utils.parallel import Parallel, delayed
from joblib import effective_n_jobs

from numpy.linalg import solve
from sklearn.decomposition import PCA
from skdim.errors import ConvergenceFailure
from scipy.stats import special_ortho_group

class lBPCA(lPCA):
    """Intrinsic dimension estimation using the Bayesian PCA algorithm. [Bishop98]

    Version 'FO' (Fukunaga-Olsen) returns eigenvalues larger than alphaFO times the largest eigenvalue.\n
    Version 'Fan' is the method by Fan et al.\n
    Version 'maxgap' returns the position of the largest relative gap in the sequence of eigenvalues.\n
    Version 'ratio' returns the number of eigenvalues needed to retain at least alphaRatio of the variance.\n
    Version 'participation_ratio' returns the number of eigenvalues given by PR=sum(eigenvalues)^2/sum(eigenvalues^2)\n
    Version 'Kaiser' returns the number of eigenvalues above average (the average eigenvalue is 1)\n
    Version 'broken_stick' returns the number of eigenvalues above corresponding values of the broken stick distribution\n

    Parameters
    ----------
    max_iter: int, default = 2000
        Number of maximum EM algorithm iterations 
    conv_tol: float, default = 1e-5
        Convergence tolerance criterion. If change in estimated W is less than conv_tol then EM stops iterating
    ver: str, default='FO'
        Version. Possible values: 'FO', 'Fan', 'maxgap','ratio', 'Kaiser', 'broken_stick'.
    alphaRatio: float in (0,1)
        Only for ver = 'ratio'. ID is estimated to be
        the number of principal components needed to retain at least alphaRatio of the variance.
    alphaFO: float in (0,1)
        Only for ver = 'FO'. An eigenvalue is considered significant
        if it is larger than alpha times the largest eigenvalue.
    alphaFan: float
        Only for ver = 'Fan'. The alpha parameter (large gap threshold).
    betaFan: float
        Only for ver = 'Fan'. The beta parameter (total covariance threshold).
    PFan: float
        Only for ver = 'Fan'. Total covariance in non-noise.
    verbose: bool, default=False

    """

    def __init__(
        self,
        max_iter = 2000,
        conv_tol = 1e-5,
        rotate = False,
        ver="FO",
        alphaRatio=0.05,
        alphaFO=0.05,
        alphaFan=10,
        betaFan=0.8,
        PFan=0.95,
        verbose=True,
        nbhd_type="knn",
        pt_nbhd_incl_pt=True,
        metric="euclidean",
        comb="mean",
        smooth=False,
        n_jobs=1,
        radius=1.0,
        n_neighbors=20,
    ):
        super().__init__(
            ver=ver,
            alphaRatio=alphaRatio,
            alphaFO=alphaFO,
            alphaFan=alphaFan,
            betaFan=betaFan,
            PFan=PFan,
            verbose=verbose,
            fit_explained_variance=False,
            nbhd_type=nbhd_type,
            pt_nbhd_incl_pt=pt_nbhd_incl_pt,
            metric=metric,
            comb=comb,
            smooth=smooth,
            n_jobs=n_jobs,
            radius=radius,
            n_neighbors=n_neighbors,
        )

        self.max_iter = max_iter
        self.conv_tol = conv_tol
        self.rotate = rotate

    def _fit(self, X, nbhd_indices, radial_dists):
        # override parent class
        self.child_attr_checks(X.shape[1])
        X = check_array(X, ensure_min_samples=2, ensure_min_features=2)

        if effective_n_jobs(self.n_jobs) > 1:
            with Parallel(n_jobs=self.n_jobs) as parallel:
                # Asynchronously apply the `fit` function to each data point and collect the results
                results = parallel(
                    delayed(self._pcaLocalDimEst)(np.take(X, nbhd, 0))
                    for nbhd in nbhd_indices
                )
            self.dimension_pw_ = np.array([result[0] for result in results])
        else:
            self.dimension_pw_ = np.array(
                [self._pcaLocalDimEst(np.take(X, nbhd, 0))[0] for nbhd in nbhd_indices]
            )
    def _pcaLocalDimEst(self, X):
        #override lPCA parent class
        N = X.shape[0]
        d = X.shape[1]
        if N >= d: #need number of points to be greater than dimensions for initial mle to work
            if self.rotate:
                X = X @ special_ortho_group.rvs(d).T #randomly rotate data to make matrix non-sparse, otherwise there may be convergence issues.
            try: 
                component_norm_sq, noise_variance,  = self._EM(X)
                #use the number of column vectors of weight matrix with large norm as the estimated dimension; borrow eigenvalue separation heuristics from PCA
                if self.ver == "FO":
                    q_eff, _ =  self._FO(component_norm_sq)
                elif self.ver == "Fan":
                    q_eff, _ = self._fan(component_norm_sq)
                elif self.ver == "maxgap":
                    q_eff, _ = self._maxgap(component_norm_sq)
                elif self.ver == "ratio":
                    q_eff, _ = self._ratio(component_norm_sq)
                elif self.ver == "participation_ratio":
                    q_eff, _ = self._participation_ratio(component_norm_sq)
                elif self.ver == "Kaiser":
                    q_eff, _ = self._Kaiser(component_norm_sq)
                elif self.ver == "broken_stick":
                    q_eff, _ = self._broken_stick(component_norm_sq)
            except ConvergenceFailure:
                q_eff, noise_variance, component_norm_sq = np.nan, np.nan, np.nan
        else:
            q_eff, noise_variance, component_norm_sq = np.nan, np.nan, np.nan

        return q_eff, component_norm_sq, noise_variance, 

    
    def _EM(self, X):
        Xm, W, nv, alpha, static_var = self._initialise(X)
        converge = False
        for _ in range(self.max_iter):
            Lmean, Lcov = self._expectation(Xm, W, nv)
            W1, nv1, alpha1, component_norm_sq = self._maximisation(Lmean, Lcov, Xm, alpha,nv, static_var)
            if np.max(np.abs(W -W1)) < self.conv_tol:
                converge = True
                return component_norm_sq, nv1
            else:
                W, nv, alpha = W1, nv1, alpha1
        if not converge:
            raise ConvergenceFailure("W matrix does not converge within " + str(self.max_iter) + " iterations, within a tolerance of |delta W| < " + f"{self.conv_tol:.3}")

    @staticmethod
    def _initialise(X):
        """
        X: n points x d feats np array
        """
        X -= np.mean(X, axis =0)
        d = X.shape[1]
        n = X.shape[0]
        if n < d:
            raise ValueError(str(n) + ' < ' + str(d))
        pca = PCA(n_components = 'mle').fit(X)

        W0 = pca.components_.T @ np.diag(np.sqrt(pca.singular_values_**2 - pca.noise_variance_)) # d x q
        alpha = d/np.maximum(1e-9, np.linalg.norm(W0, axis =0)**2) # q

        static_var = np.mean(np.var(X, axis = 0)) # trace over n then 
        return X, W0, pca.noise_variance_, alpha, static_var

    @staticmethod
    def _expectation(X, W, nv):
        """
        X: n x d
        W: d x q
        nv: float
        """
        q = W.shape[1] 

        B = (X @ W).T #  q x n
        M = W.T @ W + nv*np.eye(q) # q x q
        Lmean = solve(M, B) # q x n
        Lcov = nv * M + Lmean @ Lmean.T # q x q
        return Lmean, Lcov

    @staticmethod
    def _maximisation(Lmean, Lcov, X, alpha,nv, static_var):
        """
        Lmean: q x n
        Lcov: q x q 
        X: n x d
        nv: float
        alpha: q
        static_var: float
        """
        N,d = X.shape

        B = Lmean @ X # q x d
        C = Lcov + nv * np.diag(alpha) # q x q
        W = solve(C, B).T # d x q
        nv = static_var - (2*np.trace(X @ W @ Lmean) - np.trace(C @ W.T @ W))/(N * d)
        component_norm_sq = np.linalg.norm(W, axis =0)**2
        alpha = d/np.maximum(1e-9, component_norm_sq)

        return W, nv, alpha, component_norm_sq


    def child_attr_checks(self, d):
        if not isinstance(self.max_iter, int):
            raise ValueError("Max_iter needs to be a positive integer.")
        elif self.max_iter < 1:
            raise ValueError("Max_iter needs to be a positive integer.")
        
        if not isinstance(self.conv_tol, float):
            raise ValueError('Convolution tolerance needs to be a small positive float.') 
        elif self.conv_tol <= 0.0:
            raise ValueError('Convolution tolerance needs to be a small positive float.') 
        if not isinstance(self.rotate, bool):
            raise ValueError('Rotate parameter is either True of False.') 
        if self.n_neighbors < d:
            raise ValueError('Size of knn neighbourhoods needs to be greater than the extrinsic dimension.') 