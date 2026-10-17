# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
from skdim._commonfuncs import get_nn
import pytest
from sklearn.utils.estimator_checks import check_estimator

from skdim.datasets import hyperBall, hyperSphere
from skdim.id._TwoNN import TwoNN
from skdim.id._GRIDE import GRIDE, _gride

# same as skdim/tests/test_all_params.py
SCALAR_TRANSFORM = "transform returns the estimated ID, not a transformed X"
EXPECTED_FAILED_CHECKS = {
    "check_methods_subset_invariance": "ID estimators are not subset invariant",
    "check_methods_sample_order_invariance": SCALAR_TRANSFORM,
    "check_transformer_general": SCALAR_TRANSFORM,
    "check_transformer_data_not_an_array": SCALAR_TRANSFORM,
    "check_transformer_preserve_dtypes": SCALAR_TRANSFORM,
}

# tolerances fixed before running: SE of the MLE is ~d/sqrt(N) (2% at N=2000);
# ball boundaries bias nearest-neighbour estimators upwards, hence the wider ball tolerance
N = 2000
RTOL = {"ball": 0.15, "sphere": 0.10}
DATA = {
    "ball": lambda d: hyperBall(N, d, random_state=d),
    "sphere": lambda d: hyperSphere(N, d, random_state=d),
}
ESTIMATORS = {
    "twonn_linear": lambda: TwoNN(),
    "twonn_mle": lambda: TwoNN(method="mle"),
    "gride_1": lambda: GRIDE(),
    "gride_4": lambda: GRIDE(n1=4),
}


@pytest.mark.parametrize("Estimator", [TwoNN, GRIDE])
def test_check_estimator(Estimator):
    check_estimator(Estimator(), expected_failed_checks=EXPECTED_FAILED_CHECKS)


@pytest.mark.parametrize("name", ESTIMATORS)
@pytest.mark.parametrize("kind", DATA)
@pytest.mark.parametrize("d", [2, 5, 10])
def test_ground_truth(name, kind, d):
    est = ESTIMATORS[name]().fit(DATA[kind](d))
    assert est.dimension_ == pytest.approx(d, rel=RTOL[kind])


def test_twonn_linear_unchanged():
    # values from TwoNN before the method option was added (dev-0.4 @ 13c327b)
    assert TwoNN().fit(hyperBall(500, 5, random_state=0)).dimension_ == pytest.approx(
        4.159093274086606, rel=1e-12
    )
    # > 25 features: chunked distance path
    assert TwoNN().fit(hyperSphere(500, 30, random_state=1)).dimension_ == pytest.approx(
        21.534662028474745, rel=1e-12
    )


def test_twonn_mle_close_to_linear():
    X = hyperBall(N, 5, random_state=0)
    lin, mle = TwoNN().fit(X).dimension_, TwoNN(method="mle").fit(X).dimension_
    assert mle == pytest.approx(lin, rel=0.1)


def test_twonn_mle_closed_form_and_dist():
    X = hyperBall(500, 3, random_state=0)
    from skdim._commonfuncs import get_nn

    r = get_nn(X, 2)[0]
    log_mu = np.log(r[:, 1] / r[:, 0])
    est = TwoNN(method="mle", discard_fraction=0).fit(X)
    assert est.dimension_ == pytest.approx((len(X) - 1) / log_mu.sum(), rel=1e-12)
    assert TwoNN(method="mle", dist=True).fit(r).dimension_ == pytest.approx(
        TwoNN(method="mle").fit(X).dimension_, rel=1e-12
    )


def test_twonn_mle_unbiased_on_pareto():
    # exact model: mu ~ Pareto(1, d); censored MLE should be unbiased (tolerance fixed a priori)
    rng = np.random.default_rng(0)
    d, ests = 4.0, []
    for _ in range(300):
        mu = rng.uniform(size=200) ** (-1 / d)
        r = np.column_stack([np.ones(200), mu])
        ests.append(TwoNN(method="mle", dist=True).fit(r).dimension_)
    assert np.mean(ests) == pytest.approx(d, rel=0.02)


def test_duplicates_warn():
    X = hyperBall(300, 3, random_state=0)
    X = np.vstack([X, X[:10]])
    for est in [TwoNN(), TwoNN(method="mle"), GRIDE()]:
        with pytest.warns(UserWarning, match="duplicate"):
            assert np.isfinite(est.fit(X).dimension_)


def test_gride_n1_1_is_twonn_mle():
    # with n1=1, n2=2 Gride is the unbiased TwoNN MLE (N - 1) / sum(log mu), after DADApy's outlier cut on mu
    X = hyperBall(1000, 5, random_state=0)
    dists, _ = get_nn(X, k=2)
    mu = dists[:, 1] / dists[:, 0]
    q95, q50 = np.percentile(mu, [95, 50])
    mu = mu[mu < 20 * (q95 - q50) + q50]
    assert GRIDE().fit(X).dimension_ == pytest.approx((len(mu) - 1) / np.sum(np.log(mu)), rel=1e-8)


def test_gride_exact_model_and_fisher_error():
    # exact Poisson-process distances: r_k = (sum_{j<=k} E_j)^(1/d), E_j ~ Exp(1)
    rng = np.random.default_rng(0)
    d, n1, n2, ests, errs = 3.0, 4, 8, [], []
    for _ in range(200):
        dists = np.cumsum(rng.exponential(size=(500, n2)), axis=1) ** (1 / d)
        est, err = _gride(dists, n1, n2)
        ests.append(est)
        errs.append(err)
    assert np.mean(ests) == pytest.approx(d, rel=0.02)
    assert np.mean(errs) == pytest.approx(np.std(ests), rel=0.2)


def test_gride_multiscale_equals_single_scale():
    X = np.random.default_rng(782).normal(size=(300, 3))
    g = GRIDE(multiscale=True, range_max=32).fit(X)
    np.testing.assert_array_equal(g.scales_, [1, 2, 4, 8, 16])
    for n1, dim, err in zip(g.scales_, g.dimension_scales_, g.error_scales_):
        s = GRIDE(n1=int(n1)).fit(X)
        assert dim == pytest.approx(s.dimension_, rel=1e-10)
        assert err == pytest.approx(s.error_, rel=1e-10)
    assert np.all(np.diff(g.radius_scales_) > 0)


@pytest.mark.parametrize(
    "kwargs",
    [{"n1": 0}, {"n1": -1}, {"n1": 2, "n2": 1}, {"n1": 0.5}, {"n1": 2, "n2": 7.1}, {"n1": 200}],
)
def test_gride_invalid_params(kwargs):
    with pytest.raises(ValueError):
        GRIDE(**kwargs).fit(hyperBall(100, 2, random_state=0))


def test_twonn_invalid_method():
    with pytest.raises(ValueError):
        TwoNN(method="foo").fit(hyperBall(100, 2, random_state=0))
