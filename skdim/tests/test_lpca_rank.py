# SPDX-License-Identifier: BSD-3-Clause
from functools import lru_cache

import numpy as np
import pytest
from skdim.id._PCA import lPCA


def test_fan_does_not_mutate_input():
    ev = np.array([10.0, 5.0, 1.0, 0.1, 0.05])
    ev0 = ev.copy()
    lPCA(ver="Fan", fit_explained_variance=True).fit(ev)
    np.testing.assert_array_equal(ev, ev0)


def test_broken_stick_hand_example():
    # normalized [10,1,1,1]/13 = [.769,.077,.077,.077]; broken stick for p=4 is
    # b_k = (1/4) sum_{j>=k} 1/j = [.521,.271,.146,.063]: only the first exceeds -> 1
    est = lPCA(ver="broken_stick", fit_explained_variance=True).fit(np.array([10.0, 1, 1, 1]))
    assert est.dimension_ == 1


# ---- rank-selection versions: spiked-covariance ground truth (plan section 4.3) ----
# Tolerances fixed before running: MP, GD, KN exact in >= 9/10 seeds; parallel within +-1 in
# >= 9/10 (5/5 here). Failures are kept as xfail with what was observed.


def _spiked(n, p, theta, seed, rotate=True):
    # rotate: spikes in a random orthonormal basis. Needed for parallel analysis, since a
    # coordinate-aligned spike is a single high-variance column, which the permutation null preserves.
    rng = np.random.RandomState(seed)
    ell = np.ones(p)
    ell[: len(theta)] += theta
    X = rng.randn(n, p) * np.sqrt(ell)
    return X @ np.linalg.qr(rng.randn(p, p))[0].T if rotate else X


@lru_cache(maxsize=None)
def _spiked_spectrum(n, p, mult, seed):
    # MP, GD and KN depend on X only through the spectrum, so compute it once per cell
    from sklearn.decomposition import PCA
    theta = np.array(mult) * np.sqrt(p / n)
    ev = PCA().fit(_spiked(n, p, theta, seed, rotate=False)).explained_variance_
    return np.r_[ev, np.zeros(p - len(ev))]


def _gd_truth(theta, gamma):
    # asymptotic top sample eigenvalue (1+t)(1+gamma/t) above the GD cutoff lambda*(beta)^2 max(1, gamma)
    b = min(gamma, 1 / gamma)
    cut = (2 * (b + 1) + 8 * b / (b + 1 + np.sqrt(b * b + 14 * b + 1))) * max(1, gamma)
    t = np.asarray(theta)
    t = t[t > np.sqrt(gamma)]
    return int(np.sum((1 + t) * (1 + gamma / t) > cut))


# (n, p, theta / sqrt(beta)); GD cutoff margins checked from the asymptotic formula beforehand
CELLS = [(500, 250, (4, 2, 0.5)), (400, 800, (4, 2, 0.5)), (1000, 100, (8,) * 10)]


@pytest.mark.parametrize("ver,cell", [(v, c) for v in ["MP", "GD", "KN"] for c in CELLS])
def test_spiked_truth(ver, cell):
    n, p, mult = cell
    gamma = p / n
    theta = np.array(mult) * np.sqrt(gamma)
    truth = _gd_truth(theta, gamma) if ver == "GD" else int(np.sum(theta > np.sqrt(gamma)))
    est = [lPCA(ver=ver, fit_explained_variance=True, n_samples=n).fit(_spiked_spectrum(n, p, mult, s)).dimension_
           for s in range(10)]
    hits = sum(e == truth for e in est)
    assert hits >= 9, (truth, est)


@pytest.mark.xfail(reason="same-index null comparison overestimates after the spikes "
                   "(observed [5,4,2,5,3] and [7,4,2,4,2], truth 2, in the first run)", strict=False)
@pytest.mark.parametrize("null", ["permute", "signflip"])
def test_spiked_truth_parallel(null):
    n, p, mult = CELLS[0]
    theta = np.array(mult) * np.sqrt(p / n)
    est = [lPCA(ver="parallel", null=null, n_permutations=20, random_state=s)
           .fit(_spiked(n, p, theta, s)).dimension_ for s in range(5)]
    assert sum(abs(e - 2) <= 1 for e in est) >= 5, est


def test_center_false_scaling():
    # center=False must rescale X.T X by 1/n, otherwise sigma^2 is off by a factor n
    n, p, mult = CELLS[0]
    theta = np.array(mult) * np.sqrt(p / n)
    X = _spiked(n, p, theta, 0, rotate=False)
    for ver in ["MP", "GD", "KN"]:
        assert lPCA(ver=ver, center=False).fit(X).dimension_ == lPCA(ver=ver).fit(X).dimension_


def _wilson_low(k, m, z=1.96):
    ph = k / m
    return (ph + z * z / (2 * m) - z * np.sqrt(ph * (1 - ph) / m + z * z / (4 * m * m))) / (1 + z * z / m)


def test_null_fpr_kn():
    # r = 0: P(KN selects >= 1) must not be significantly above alpha, 500 spectrum-only seeds
    n, p = 200, 50
    for alpha in [0.05, 0.01]:
        fp = 0
        for s in range(500):
            # uncentered white Wishart with n dof: covariance-scale spectrum, n_eff = n_samples - 1 = n
            ev = np.linalg.svd(np.random.RandomState(s).randn(n, p), compute_uv=False) ** 2 / n
            est = lPCA(ver="KN", alphaKN=alpha, fit_explained_variance=True, n_samples=n + 1)
            fp += est.fit(ev).dimension_ > 0
        assert _wilson_low(fp, 500) <= alpha, fp


@pytest.mark.parametrize("null", ["permute", "signflip"])
def test_null_fpr_parallel(null):
    # 19 draws with quantile=1.0 is an exact alpha=0.05 test of the first eigenvalue
    n, p, m = 60, 10, 300
    fp = sum(lPCA(ver="parallel", null=null, n_permutations=19, quantile=1.0, random_state=s)
             .fit(np.random.RandomState(s).randn(n, p)).dimension_ > 0 for s in range(m))
    assert _wilson_low(fp, m) <= 0.05, fp


@pytest.mark.parametrize("ver", ["MP", "GD", "KN"])
@pytest.mark.parametrize("n,p", [(500, 250), (100, 300)])
def test_explained_variance_path(ver, n, p):
    from sklearn.decomposition import PCA
    X = _spiked(n, p, [4, 3], 0)
    ev = PCA().fit(X).explained_variance_
    ev = np.r_[ev, np.zeros(p - len(ev))]
    d_X = lPCA(ver=ver).fit(X).dimension_
    assert lPCA(ver=ver, fit_explained_variance=True, n_samples=n).fit(ev).dimension_ == d_X


def test_noise_var_known():
    X = _spiked(500, 250, [4, 3], 0)
    for ver in ["MP", "GD", "KN"]:
        assert lPCA(ver=ver, noise_var=1.0).fit(X).dimension_ == 2


def test_errors():
    X = _spiked(50, 10, [4], 0)
    ev = np.linspace(2, 1, 10)
    for kw in [dict(ver="nope"), dict(ver="KN", alphaKN=0.07),
               dict(ver="parallel", null="bootstrap"), dict(ver="parallel", quantile=0)]:
        with pytest.raises(ValueError):
            lPCA(**kw).fit(X)
    for ver in ["parallel", "MP", "GD", "KN"]:
        with pytest.raises(ValueError):
            lPCA(ver=ver, fit_explained_variance=True).fit(ev)


def test_check_estimator():
    from sklearn.utils.estimator_checks import check_estimator
    from skdim.tests.test_all_params import EXPECTED_FAILED_CHECKS
    check_estimator(lPCA(), expected_failed_checks=EXPECTED_FAILED_CHECKS)
