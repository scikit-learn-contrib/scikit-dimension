# SPDX-License-Identifier: BSD-3-Clause AND MIT
# Adapted from R package intrinsicDimension (https://cran.r-project.org/package=intrinsicDimension),
# Copyright (c) 2017 Kerstin Johnsson, MIT License. ver="scRMT" follows the randomly package
# (https://github.com/RabadanLab/randomly), Copyright (c) 2018 Luis Aparicio, MIT License; see LICENSES_bundled.txt
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
_RANK_VERS = ("MP", "GD", "KN", "Minka", "parallel", "denoised", "scRMT")
_VERS = ("FO", "Fan", "maxgap", "ratio", "participation_ratio", "Kaiser", "broken_stick") + _RANK_VERS


class lPCA(GlobalEstimator):
    """Intrinsic dimension and signal rank from the PCA eigenvalue spectrum.

    Explained-variance rules [Fukunaga2010]_ [Fan2010]_ [Cangelosi2007]_ [IDJohnsson]_:

    - 'FO': eigenvalues larger than ``alphaFO`` times the largest one (Fukunaga-Olsen).
    - 'Fan': the method of Fan et al.
    - 'maxgap': position of the largest ratio between consecutive eigenvalues.
    - 'ratio': number of components needed to retain ``alphaRatio`` of the variance.
    - 'participation_ratio': sum(eigenvalues)^2 / sum(eigenvalues^2).
    - 'Kaiser': eigenvalues above their mean ("eigenvalue > 1" for standardized data).
    - 'broken_stick': leading eigenvalues above the broken-stick expectation.

    Signal-rank estimators, for data = low-rank signal + iid noise:

    - 'MP': eigenvalues above the Marchenko-Pastur edge sigma^2 (1 + sqrt(p/n))^2 [Marchenko1967]_.
    - 'GD': Gavish-Donoho optimal hard threshold [Gavish2014]_.
    - 'KN': Kritchman-Nadler sequential Tracy-Widom test at level ``alphaKN`` [Kritchman2008]_.
    - 'Minka': maximum of Minka's Laplace approximation of the PPCA evidence [Minka2000]_.
    - 'parallel': Horn's parallel analysis, with a permutation or sign-flip null [Horn1965]_ [Buja1992]_.
    - 'denoised': scran's rule: fewest leading components whose discarded variance does not exceed
      the technical (noise) variance [Lun2016]_.
    - 'scRMT': random matrix theory rule for single-cell data: eigenvalues of the standardized data
      above the fitted Marchenko-Pastur edge plus one Tracy-Widom scale unit [Aparicio2020]_.

    Parameters
    ----------
    ver: str, default='FO'
        Estimator, one of the versions listed above.
    center: bool, default=True
        Center the data before computing eigenvalues. If False, the eigenvalues of
        ``X.T @ X / n`` are used. Ignored when ``fit_explained_variance`` is True.
    fit_explained_variance: bool, default=False
        If True, ``fit`` takes a precomputed eigenvalue vector (e.g.
        ``sklearn.decomposition.PCA().fit(X).explained_variance_``) instead of data.
        Not available for 'parallel' and 'scRMT'.
    n_samples: int, default=None
        Number of samples behind a precomputed eigenvalue vector; required by the
        signal-rank versions when ``fit_explained_variance`` is True.
    alphaFO: float in (0, 1), default=0.05
        'FO': fraction of the largest eigenvalue an eigenvalue must exceed.
    alphaFan: float, default=10
        'Fan': large-gap threshold.
    betaFan: float, default=0.8
        'Fan': total-variance threshold.
    PFan: float, default=0.95
        'Fan': fraction of variance attributed to signal.
    alphaRatio: float in (0, 1), default=0.05
        'ratio': fraction of the variance to retain.
    noise_var: float or 1d array, default=None
        'MP', 'GD', 'KN', 'denoised': noise variance per feature. If None, it is estimated from the
        spectrum. 'denoised' also accepts one technical variance per feature.
    alphaKN: {0.1, 0.05, 0.01, 0.005, 0.001}, default=0.05
        'KN': significance level.
    n_permutations: int, default=100
        'parallel': number of null datasets.
    quantile: float in (0, 1], default=0.95
        'parallel': quantile of the null eigenvalues to exceed.
    null: {'permute', 'signflip'}, default='permute'
        'parallel': permute each feature independently, or flip the sign of each centered entry.
    random_state: int, RandomState instance or None, default=None
        'parallel', 'scRMT': random number generator (permutation nulls).
    verbose: bool, default=True
        Unused.

    Notes
    -----
    The signal-rank versions estimate the rank of a linear signal, not a manifold dimension: curved
    manifold directions with little variance fall below the noise level. They assume iid noise, so
    heteroscedastic data (e.g. log-normalized counts) should be standardized or biwhitened first.

    Attributes
    ----------
    dimension_: int
        Estimated dimension or signal rank.
    gap_: 1d array
        Ratio of each eigenvalue (except the last) to the next one.
    threshold_: float
        'scRMT' only: eigenvalue threshold (Marchenko-Pastur edge plus one Tracy-Widom unit).

    References
    ----------
    .. [Marchenko1967] Marchenko, V. A., Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. Math. USSR Sb. 1, 457-483. doi:10.1070/SM1967v001n04ABEH001994
    .. [Lun2016] Lun, A. T. L., McCarthy, D. J., Marioni, J. C. (2016). A step-by-step workflow for low-level analysis of single-cell RNA-seq data with Bioconductor. F1000Research 5, 2122. doi:10.12688/f1000research.9501.2
    .. [Aparicio2020] Aparicio, L., Bordyuh, M., Blumberg, A. J., Rabadan, R. (2020). A random matrix theory approach to denoise single-cell data. Patterns 1, 100035. doi:10.1016/j.patter.2020.100035
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
        *,
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
        if self.fit_explained_variance and self.ver in ("parallel", "scRMT"):
            raise ValueError(f"ver={self.ver!r} needs X; it cannot be used with fit_explained_variance=True")
        if self.fit_explained_variance and self.ver in _RANK_VERS and self.n_samples is None:
            raise ValueError(f"ver={self.ver!r} with fit_explained_variance=True requires n_samples")
        if np.ndim(self.noise_var) and self.ver != "denoised":
            raise ValueError("an array noise_var (per-feature technical variances) is only valid for ver='denoised'")
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
        if self.ver == "scRMT":
            return self._scrmt(X)

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
        if self.ver == "denoised":
            tech = np.sum(sigma2) if np.ndim(sigma2) else p * sigma2
            above = np.cumsum(ev[::-1]) > tech  # variance of the j smallest PCs vs technical variance
            return p - int(np.argmax(above)) if above.any() else 1
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

    def _scrmt(self, X):
        """Aparicio et al. (2020) rank, following the randomly package: eigenvalues of the sample
        Wishart matrix of standardized features, Marchenko-Pastur bulk fitted by moment matching
        from a permuted copy, threshold at the bulk edge plus one Tracy-Widom scale unit."""
        X = np.asarray(X, dtype=float)
        sd = X.std(axis=0)
        X = (X[:, sd > 0] - X[:, sd > 0].mean(axis=0)) / sd[sd > 0]  # constant features carry no signal
        rng = check_random_state(self.random_state)
        Xr = np.column_stack([rng.permutation(col) for col in X.T])
        L = np.linalg.eigvalsh(X @ X.T / X.shape[1])
        Lr = np.linalg.eigvalsh(Xr @ Xr.T / X.shape[1])

        def mp(S):  # moment-matched Marchenko-Pastur parameters
            m1, m2 = S.mean(), np.mean(S ** 2)
            g = m2 / m1 ** 2 - 1
            return g, m1 * (1 + np.sqrt(g)) ** 2, m1 * (1 - np.sqrt(g)) ** 2

        _, b_plus, _ = mp(Lr)
        _, new_b_plus, new_b_minus = mp(L[(L > mp(Lr)[2]) & (L < b_plus)])
        for _ in range(1000):  # fixed point of the bulk edges
            if (1 - new_b_plus / b_plus) ** 2 <= 1e-6:
                break
            b_plus = new_b_plus
            _, new_b_plus, new_b_minus = mp(L[(L > new_b_minus) & (L < new_b_plus)])
        bulk = L[(L > new_b_minus) & (L < new_b_plus)]
        gamma = mp(bulk)[0]
        tw = (len(L) / gamma) ** (-2 / 3) * gamma ** (5 / 6) * (1 + np.sqrt(gamma)) ** (4 / 3)
        self.threshold_ = bulk.mean() * (1 + np.sqrt(gamma)) ** 2 + tw
        return int(np.sum(L > self.threshold_))

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
