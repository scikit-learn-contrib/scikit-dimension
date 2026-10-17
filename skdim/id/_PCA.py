# SPDX-License-Identifier: BSD-3-Clause AND MIT
# Adapted from R package intrinsicDimension (https://cran.r-project.org/package=intrinsicDimension),
# Copyright (c) 2017 Kerstin Johnsson, MIT License; see LICENSES_bundled.txt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import loggamma
from sklearn.decomposition import PCA
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from .._commonfuncs import _check_X, GlobalEstimator

# Upper quantiles s(alpha) of the Tracy-Widom (beta=1) law, P(TW1 > s) = alpha.
# Values from the TW1 tables of Bejan (2005), "Largest eigenvalues and sample covariance
# matrices", consistent with Johnstone (2001, Ann. Stat., Table 1); re-checked to 4 decimals
# with Bornemann's (2010) Fredholm-determinant quadrature. scipy has no TW distribution.
_TW1_UPPER = {0.1: 0.4501, 0.05: 0.9793, 0.01: 2.0234, 0.005: 2.4223, 0.001: 3.2722}
_RANK_VERS = ("MP", "GD", "KN", "Minka", "parallel")
_VERS = ("FO", "Fan", "maxgap", "ratio", "participation_ratio", "Kaiser", "broken_stick") + _RANK_VERS


class lPCA(GlobalEstimator):
    """Intrinsic dimension estimation using the PCA algorithm. [Cangelosi2007]_ [Fan2010]_ [Fukunaga2010]_ [IDJohnsson]_

    Version 'FO' (Fukunaga-Olsen) returns eigenvalues larger than alphaFO times the largest eigenvalue.\n
    Version 'Fan' is the method by Fan et al.\n
    Version 'maxgap' returns the position of the largest relative gap in the sequence of eigenvalues.\n
    Version 'ratio' returns the number of eigenvalues needed to retain at least alphaRatio of the variance.\n
    Version 'participation_ratio' returns the number of eigenvalues given by PR=sum(eigenvalues)^2/sum(eigenvalues^2)\n
    Version 'Kaiser' returns the number of eigenvalues above their mean. For the covariance of standardized data the mean is 1, so this is the usual 'eigenvalue > 1' rule; otherwise it is not.\n
    Version 'broken_stick' returns the number of leading eigenvalues above the corresponding values of the broken stick distribution\n

    The versions below estimate the **linear signal rank** under iid homoscedastic noise
    (spiked covariance model), not the manifold ID: low-variance manifold directions fall below
    the noise edge, so they can return less than the ID. On log-normalized scRNA-seq they usually
    overestimate (heteroscedastic noise); standardize or biwhiten first.
    Eigenvalues are on the covariance scale (divided by n-1 when centered, by n when not).
    With ``center=False`` a nonzero mean creates a spurious spike.
    Their asymptotics are shaky for small neighbourhoods (n around 100), e.g. in ``fit_pw``.

    Version 'MP' counts eigenvalues above the Marchenko-Pastur edge sigma^2 (1 + sqrt(p/n))^2 [Marchenko1967]_.\n
    Version 'GD' is the Gavish-Donoho optimal hard threshold, omega(beta) times the median singular value [Gavish2014]_. It is MSE-optimal and drops detectable but weak spikes by design.\n
    Version 'KN' is the Kritchman-Nadler sequential Tracy-Widom test at level alphaKN, with sigma^2 estimated jointly with the signal eigenvalues under each step's null [Kritchman2008]_.\n
    Version 'Minka' maximizes Minka's Laplace approximation of the PPCA evidence [Minka2000]_. Also valid when n_samples < n_features. Ported from the kmyim fork of scikit-dimension (KM Yim, John Harvey).\n
    Version 'parallel' is Horn's parallel analysis [Horn1965]_ [Buja1992]_ with a permutation (each column permuted independently) or sign-flip null: the rank is the number of leading eigenvalues above the ``quantile`` of their null distribution over ``n_permutations`` draws. Cost is n_permutations PCAs (times the number of neighbourhoods in ``fit_pw``). It can underestimate under strong factors (shadowing) and, as each eigenvalue is compared with the same-index null eigenvalue, overestimate when weak spikes are present.\n

    Parameters
    ----------
    ver: str, default='FO'
        Version. Possible values: 'FO', 'Fan', 'maxgap', 'ratio', 'participation_ratio', 'Kaiser', 'broken_stick',
        'MP', 'GD', 'KN', 'Minka', 'parallel'.
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
    verbose: bool, default=True
        Unused.
    center: bool, default=True
        If True, eigenvalues are computed from ``sklearn.decomposition.PCA``, which
        centers (demeans) the data. If False, they are computed as the eigenvalues of
        ``X.T @ X`` without centering, so for ver='participation_ratio' the estimate is
        ``sum(eigenvalues) ** 2 / sum(eigenvalues ** 2)`` of ``X.T @ X``.
        Ignored when ``fit_explained_variance`` is True.
    fit_explained_variance: bool, default=False
        If True, lPCA.fit(X) expects as input 
        a precomputed explained_variance vector: X = sklearn.decomposition.PCA().fit(X).explained_variance_
        For 'MP', 'GD', 'KN' and 'Minka' it must be of length n_features (pad with zeros if
        n_samples < n_features) and ``n_samples`` must be given. Not possible for 'parallel',
        which needs X.
    n_samples: int, default=None
        Number of samples the explained variance was computed from. Required for 'MP', 'GD', 'KN'
        and 'Minka' when ``fit_explained_variance`` is True; ignored otherwise.
    noise_var: float, default=None
        Known noise variance sigma^2 for 'MP', 'GD' and 'KN'. If None, it is estimated: from the
        median eigenvalue matched to the Marchenko-Pastur median ('MP', 'GD'), or jointly with the
        signal eigenvalues ('KN').
    alphaKN: float, default=0.05
        Only for ver = 'KN'. Significance level, one of 0.1, 0.05, 0.01, 0.005, 0.001
        (tabulated Tracy-Widom quantiles).
    n_permutations: int, default=100
        Only for ver = 'parallel'. Number of null draws. 19 with quantile=1.0 gives an exact
        alpha=0.05 test of each successive eigenvalue against the null maximum.
    quantile: float in (0, 1], default=0.95
        Only for ver = 'parallel'. Quantile of the null eigenvalue distribution to exceed.
    null: {'permute', 'signflip'}, default='permute'
        Only for ver = 'parallel'. Null: permute each column independently, or flip the sign of
        each (centered) entry independently.
    random_state: int, RandomState instance or None, default=None
        Only for ver = 'parallel'.
    
    Attributes
    ----------
    gap_:
        Ratio of each PC's explained variance (except the last) 
        with the following PC's explained variance

    References
    ----------
    .. [Marchenko1967] Marchenko, V. A., Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. Math. USSR Sb. 1, 457-483. doi:10.1070/SM1967v001n04ABEH001994
    .. [Gavish2014] Gavish, M., Donoho, D. L. (2014). The optimal hard threshold for singular values is 4/sqrt(3). IEEE Trans. Inf. Theory 60, 5040-5053. doi:10.1109/TIT.2014.2323359
    .. [Kritchman2008] Kritchman, S., Nadler, B. (2008). Determining the number of components in a factor model from limited noisy data. Chemom. Intell. Lab. Syst. 94, 19-32. doi:10.1016/j.chemolab.2008.06.002
    .. [Minka2000] Minka, T. P. (2000). Automatic choice of dimensionality for PCA. NeurIPS 13.
    .. [Horn1965] Horn, J. L. (1965). A rationale and test for the number of factors in factor analysis. Psychometrika 30, 179-185. doi:10.1007/BF02289447
    .. [Buja1992] Buja, A., Eyuboglu, N. (1992). Remarks on parallel analysis. Multivariate Behav. Res. 27, 509-540. doi:10.1207/s15327906mbr2704_2
    """

    def __init__(
        self,
        ver="FO",
        alphaRatio=0.05,
        alphaFO=0.05,
        alphaFan=10,
        betaFan=0.8,
        PFan=0.95,
        verbose=True,
        center=True,
        fit_explained_variance=False,
        n_samples=None,
        noise_var=None,
        alphaKN=0.05,
        n_permutations=100,
        quantile=0.95,
        null="permute",
        random_state=None,
    ):
        self.ver = ver
        self.alphaRatio = alphaRatio
        self.alphaFO = alphaFO
        self.alphaFan = alphaFan
        self.betaFan = betaFan
        self.PFan = PFan
        self.verbose = verbose
        self.center = center
        self.fit_explained_variance = fit_explained_variance
        self.n_samples = n_samples
        self.noise_var = noise_var
        self.alphaKN = alphaKN
        self.n_permutations = n_permutations
        self.quantile = quantile
        self.null = null
        self.random_state = random_state

    def fit(self, X, y=None):
        """A reference implementation of a fitting function.
        
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A local dataset of training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self : object
            Returns self.
        """
        if self.fit_explained_variance:
            X = _check_X(self, X, ensure_2d=False, ensure_min_samples=2)
        else:
            X = _check_X(self, X, ensure_min_samples=2, ensure_min_features=2)

        self.dimension_, self.gap_ = self._pcaLocalDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _fit_once(self, X, y=None):
        """A reference implementation of a fitting function.
        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            A local dataset of training input samples.
        y : dummy parameter to respect the sklearn API

        Returns
        -------
        self : object
            Returns self.
        """
        if self.fit_explained_variance:
            X = _check_X(self, X, ensure_2d=False, ensure_min_samples=2)
        else:
            X = _check_X(self, X, ensure_min_samples=2, ensure_min_features=2)

        self.dimension_, self.gap_ = self._pcaLocalDimEst(X)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def _check_params(self):
        if self.ver not in _VERS:
            raise ValueError(f"ver must be one of {_VERS}, got {self.ver!r}")
        if self.fit_explained_variance and self.ver == "parallel":
            raise ValueError(f"ver={self.ver!r} needs X; it cannot be used with fit_explained_variance=True")
        if self.fit_explained_variance and self.ver in _RANK_VERS and self.n_samples is None:
            raise ValueError(f"ver={self.ver!r} with fit_explained_variance=True requires n_samples")
        if self.ver == "KN" and self.alphaKN not in _TW1_UPPER:
            raise ValueError(f"alphaKN must be one of {sorted(_TW1_UPPER)}, got {self.alphaKN!r}")
        if self.ver == "parallel":
            if self.null not in ("permute", "signflip"):
                raise ValueError(f"null must be 'permute' or 'signflip', got {self.null!r}")
            if not 0 < self.quantile <= 1:
                raise ValueError(f"quantile must be in (0, 1], got {self.quantile!r}")

    def _spectrum(self, X):
        if self.center:
            return PCA().fit(X).explained_variance_
        # Eigenvalues of X.T @ X without centering the data (issue #14).
        # Computed from the singular values of X to avoid forming X.T @ X.
        return np.linalg.svd(X, compute_uv=False) ** 2

    def _pcaLocalDimEst(self, X):
        self._check_params()
        if self.fit_explained_variance:
            explained_var = X
        else:
            self.explained_var_ = explained_var = self._spectrum(X)

        if self.ver in _RANK_VERS:
            gaps = explained_var[:-1] / explained_var[1:]
            return self._rank(X, explained_var), gaps

        if self.ver == "FO":
            return self._FO(explained_var)
        elif self.ver == "Fan":
            return self._fan(explained_var)
        elif self.ver == "maxgap":
            return self._maxgap(explained_var)
        elif self.ver == "ratio":
            return self._ratio(explained_var)
        elif self.ver == "participation_ratio":
            return self._participation_ratio(explained_var)
        elif self.ver == "Kaiser":
            return self._Kaiser(explained_var)
        elif self.ver == "broken_stick":
            return self._broken_stick(explained_var)

    def _rank(self, X, explained_var):
        """Signal rank for the random-matrix versions. Works on covariance-scale eigenvalues
        `ev` of length p (zero-padded), with n_eff = n - 1 if centered else n."""
        if self.fit_explained_variance:
            n, p, n_eff = self.n_samples, len(X), self.n_samples - 1
            ev = np.asarray(explained_var, dtype=float)
        else:
            n, p = X.shape
            n_eff = n - 1 if self.center else n
            ev = explained_var if self.center else explained_var / n
        ev = np.concatenate([np.sort(ev)[::-1], np.zeros(p - len(ev))])

        if self.ver == "Minka":
            return self._minka(ev, n)
        if self.ver == "parallel":
            return self._parallel(X, explained_var)

        # nonzero eigenvalues: at most min(n_eff, p), and drop numerical zeros
        nz = ev[: min(n_eff, p)]
        nz = nz[nz > nz[0] * max(n, p) * np.finfo(float).eps]
        if self.ver == "KN":
            return self._kn(ev, n_eff)
        beta = min(n_eff, p) / max(n_eff, p)
        # sigma^2 such that the median nonzero eigenvalue matches the MP median
        # (nonzero eigenvalues / sigma^2 follow MP(beta) scaled by max(n_eff, p) / n_eff)
        sigma2 = self.noise_var
        if sigma2 is None:
            sigma2 = np.median(nz) / (_mp_median(beta) * max(n_eff, p) / n_eff)
        self.noise_var_ = sigma2
        if self.ver == "MP":
            thr = sigma2 * (1 + np.sqrt(p / n_eff)) ** 2
        else:  # GD: singular value > lambda*(beta) sqrt(max(n_eff, p)) sigma, Gavish-Donoho eq. (11)
            lam = np.sqrt(2 * (beta + 1) + 8 * beta / (beta + 1 + np.sqrt(beta ** 2 + 14 * beta + 1)))
            thr = lam ** 2 * max(n_eff, p) / n_eff * sigma2
        return int(np.sum(nz > thr))

    def _kn(self, ev, n):
        """Kritchman & Nadler (2008) sequential test, with El Karoui's n-1/2, p-1/2 centering.
        Step k tests H0: k-1 signals, so l_k is the largest of p-k+1 noise eigenvalues and sigma^2 is
        estimated under H0 (k-1 signals). Estimating it with k signals, as KN's own code is described
        as doing, underestimates sigma^2 under H0 and gave a false-positive rate of 0.08-0.10 at
        alpha=0.05 on pure noise (n=100-200, p=50-300); this version gave 0.03-0.05."""
        p = len(ev)
        s = _TW1_UPPER[self.alphaKN]
        for k in range(1, min(n, p)):
            sigma2 = self.noise_var if self.noise_var is not None else _kn_noise(ev, n, k - 1)
            a, b = np.sqrt(n - 0.5), np.sqrt(p - k + 1 - 0.5)
            mu = (a + b) ** 2
            xi = (a + b) * (1 / a + 1 / b) ** (1 / 3)
            if n * ev[k - 1] <= sigma2 * (mu + s * xi):
                return k - 1
        return min(n, p) - 1

    def _minka(self, ev, n):
        """argmax over k = 1..r-1 of Minka's (2000) Laplace evidence (r = numerical rank).
        Same terms as sklearn's _assess_dimension, accumulated in k so the cost is O(r p)."""
        d = len(ev)
        L = ev / ev[0]  # argmax is scale invariant; normalizing keeps the log clamp meaningful
        r = int(np.sum(L > d * np.finfo(float).eps))
        if r < 2:
            return r
        log = lambda x: np.log(np.maximum(x, np.finfo(float).tiny))
        ll = np.empty(r - 1)
        pu = ediff = ainv = 0.0
        for k in range(1, r):
            i = k - 1
            pu += loggamma((d - i) / 2) - np.log(np.pi) * (d - i) / 2
            ediff += log(L[i] - L[i + 1 :]).sum()  # log(l_i - l_j), j > i
            ainv += log((L[:i] - L[i]) / (L[:i] * L[i])).sum()  # log(1/l_j - 1/l_i), i < j <= k
            nu = L[k:].sum() / (d - k)
            binv = (d - k) * log((L[:k] - nu) / (L[:k] * nu)).sum()  # j > k: hat l_j = nu
            m = d * k - k * (k + 1) / 2
            pa = m * np.log(n) + ediff + ainv + binv
            ll[i] = (pu - k * np.log(2) - n / 2 * log(L[:k]).sum() - n * (d - k) / 2 * np.log(nu)
                     + np.log(2 * np.pi) * (m + k) / 2 - pa / 2 - k * np.log(n) / 2)
        self.log_evidence_ = ll
        return int(np.argmax(ll)) + 1

    def _parallel(self, X, explained_var):
        rng = check_random_state(self.random_state)
        n, p = X.shape
        if self.null == "signflip":
            X = X - X.mean(axis=0) if self.center else X
        null = np.empty((self.n_permutations, len(explained_var)))
        for b in range(self.n_permutations):
            if self.null == "permute":
                Xb = np.take_along_axis(X, np.argsort(rng.rand(n, p), axis=0), axis=0)
            else:
                Xb = X * np.where(rng.rand(n, p) < 0.5, -1.0, 1.0)
            # same eigenvalues as _spectrum, without sklearn's per-call overhead
            Xb = Xb - Xb.mean(axis=0) if self.center else Xb
            null[b] = np.linalg.svd(Xb, compute_uv=False)[: len(explained_var)] ** 2 / (n - 1 if self.center else 1)
        above = explained_var > np.quantile(null, self.quantile, axis=0)
        # sequential: number of leading eigenvalues above their null quantile
        return int(np.argmin(above)) if not above.all() else len(above)

    def _FO(self, explained_var):
        de = sum(explained_var > (self.alphaFO * explained_var[0]))
        gaps = explained_var[:-1] / explained_var[1:]
        return de, gaps


    @staticmethod
    def _maxgap(explained_var):
        gaps = explained_var[:-1] / explained_var[1:]
        de = np.nanargmax(gaps) + 1
        return de, gaps


    def _ratio(self, explained_var):
        sumexp = np.cumsum(explained_var)
        sumexp_norm = sumexp / np.max(sumexp)
        de = sum(sumexp_norm < self.alphaRatio) + 1
        gaps = explained_var[:-1] / explained_var[1:]
        return de, gaps


    def _participation_ratio(self, explained_var):
        PR = sum(explained_var) ** 2 / sum(explained_var ** 2)
        de = PR
        gaps = explained_var[:-1] / explained_var[1:]
        return de, gaps


    def _fan(self, explained_var):
        r = np.where(np.cumsum(explained_var) / sum(explained_var) > self.PFan)[0][0]
        sigma = np.mean(explained_var[r:])
        explained_var = explained_var - sigma  # do not mutate the caller's array
        gaps = explained_var[:-1] / explained_var[1:]
        de = 1 + np.min(
            np.concatenate(
                (
                    np.where(gaps > self.alphaFan)[0],
                    np.where(
                        (np.cumsum(explained_var) / sum(explained_var)) > self.betaFan
                    )[0],
                )
            )
        )
        return de, gaps


    def _Kaiser(self, explained_var):
        de = sum(explained_var > np.mean(explained_var))
        gaps = explained_var[:-1] / explained_var[1:]
        return de, gaps


    @staticmethod
    def _brokenstick_distribution(dim):
        distr = np.zeros(dim)
        for i in range(dim):
            for j in range(i, dim):
                distr[i] = distr[i] + 1 / (j + 1)
            distr[i] = distr[i] / dim
        return distr

    def _broken_stick(self, explained_var):
        bs = self._brokenstick_distribution(dim=len(explained_var))
        gaps = explained_var[:-1] / explained_var[1:]
        de = 0
        explained_var_norm = explained_var / np.sum(explained_var)
        for i in range(len(explained_var)):
            if bs[i] > explained_var_norm[i]:
                de = i  # number of leading eigenvalues above the broken stick
                break
        return de, gaps


def _mp_median(beta):
    """Median of the Marchenko-Pastur law with ratio beta <= 1 and unit variance."""
    a, b = (1 - np.sqrt(beta)) ** 2, (1 + np.sqrt(beta)) ** 2
    pdf = lambda x: np.sqrt((b - x) * (x - a)) / (2 * np.pi * beta * x)
    return brentq(lambda x: quad(pdf, a, x)[0] - 0.5, a + 1e-9, b)


def _kn_noise(ev, n, k):
    """Kritchman & Nadler (2008) noise variance given k signals: fixed point of
    sigma^2 = (sum_{j>k} l_j + sum_{j<=k} (l_j - rho_j)) / (p - k), rho_j the debiased spikes."""
    p = len(ev)
    sigma2 = ev[k:].sum() / (p - k)
    for _ in range(30):
        t = ev[:k] + sigma2 - sigma2 * (p - k) / n
        rho = (t + np.sqrt(np.maximum(t ** 2 - 4 * ev[:k] * sigma2, 0))) / 2
        new = (ev[k:].sum() + (ev[:k] - rho).sum()) / (p - k)
        if abs(new - sigma2) <= 1e-5 * sigma2:
            return new
        sigma2 = new
    return sigma2


##### dev in progress
# from sklearn.cluster import KMeans
# def pcaOtpmPointwiseDimEst(data, N, alpha = 0.05):
#    km = KMeans(n_clusters=N)
#    km.fit(data)
#    pt = km.cluster_centers_
#    pt_bm = km.labels_
#    pt_sm = np.repeat(np.nan, len(pt_bm))
#
#    for k in range(len(data)):
#        pt_sm[k] = np.argmin(lens(pt[[i for i in range(N) if i!=pt_bm[k]],:] - data[k,:]))
#        if (pt_sm[k] >= pt_bm[k]):
#            pt_sm[k] += 1
#
#    de_c = np.repeat(np.nan, N)
#    nbr_nb_c = np.repeat(np.nan, N)
#    for k in range(N):
#        nb = np.unique(np.concatenate((pt_sm[pt_bm == k], pt_bm[pt_sm == k]))).astype(int)
#        nbr_nb_c[k] = len(nb)
#        loc_dat = pt[nb,:] - pt[k,:]
#        if len(loc_dat) == 1:
#            continue
#        de_c[k] = lPCA().fit(loc_dat).dimension_ #pcaLocalDimEst(loc_dat, ver = "FO", alphaFO = alpha)
#
#    de = de_c[pt_bm]
#    nbr_nb = nbr_nb_c[pt_bm]
#    return de, nbr_nb
