# SPDX-License-Identifier: BSD-3-Clause
import pytest
import numpy as np
import skdim
import matplotlib.pyplot as plt
from inspect import getmembers, isclass
from sklearn.utils.estimator_checks import check_estimator


@pytest.fixture
def data():
    X = np.zeros((100, 10))
    X[:, :5] = skdim.datasets.hyperBall(n=100, d=5, radius=1, random_state=0)
    return X


# test all estimators pass check_estimator
estimators = [o[1] for o in getmembers(skdim.id) if isclass(o[1])]


# transform() returns the scalar ID, not a transformed X, so these checks do not apply
SCALAR_TRANSFORM = "transform returns the estimated ID, not a transformed X"
EXPECTED_FAILED_CHECKS = {
    "check_methods_subset_invariance": "ID estimators are not subset invariant",
    "check_methods_sample_order_invariance": SCALAR_TRANSFORM,
    "check_transformer_general": SCALAR_TRANSFORM,
    "check_transformer_data_not_an_array": SCALAR_TRANSFORM,
    "check_transformer_preserve_dtypes": SCALAR_TRANSFORM,
}
# check_estimator fits on 10 samples; DANCo needs k + 1 samples
ESTIMATOR_KWARGS = {"DANCo": {"k": 5}}


@pytest.mark.parametrize("Estimator", estimators)
def test_all_estimators(Estimator):
    est = Estimator(**ESTIMATOR_KWARGS.get(Estimator.__name__, {}))
    try:  # sklearn>=1.6
        return check_estimator(est, expected_failed_checks=EXPECTED_FAILED_CHECKS)
    except TypeError as e:
        if "expected_failed_checks" not in str(e):
            raise
        return check_estimator(est)


# test default and non-default parameters
def test_ess_params(data):
    x = skdim.id.ESS().fit(data)
    x = skdim.id.ESS(ver="b").fit(data)
    x = skdim.id.ESS(d=2).fit(data)
    x = skdim.id.ESS().fit_once(data)

    ### additionally test all common LocalEstimator base functions
    x = skdim.id.ESS().fit(data).transform()
    x = skdim.id.ESS().fit(data).transform_pw()
    x = skdim.id.ESS().fit_transform(data)
    x = skdim.id.ESS().fit_transform_pw(data)
    x = skdim.id.ESS().fit_transform_pw(data, smooth=True)


def test_fisher_params(data, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    x = skdim.id.FisherS().fit(data)
    x = skdim.id.FisherS(conditional_number=2).fit(data)
    x = skdim.id.FisherS(produce_plots=True).fit(data)
    x = skdim.id.FisherS(project_on_sphere=False).fit(data)
    x = skdim.id.FisherS(verbose=True).fit(data)
    x = skdim.id.FisherS(limit_maxdim=True).fit(data)
    x = skdim.id.FisherS().fit(data).point_inseparability_to_pointID()

    ### additionally test all common GlobalEstimator base functions
    x = skdim.id.FisherS().fit(data).transform()
    x = skdim.id.FisherS().fit_pw(data, n_neighbors=50).transform_pw()
    x = skdim.id.FisherS().fit_transform(data)
    x = skdim.id.FisherS().fit_transform_pw(data, n_neighbors=50)


def test_mind_ml_params(data):
    x = skdim.id.MiND_ML()
    x = skdim.id.MiND_ML(ver="MLi")
    x = skdim.id.MiND_ML(ver="ML1")
    x = skdim.id.MiND_ML(D=5)
    x = skdim.id.MiND_ML(k=5)


def test_mom_params(data):
    x = skdim.id.MOM()


def test_lpca_params(data):
    x = skdim.id.lPCA().fit(data)
    x = skdim.id.lPCA(ver="Fan").fit(data)
    x = skdim.id.lPCA(ver="ratio").fit(data)
    x = skdim.id.lPCA(ver="maxgap").fit(data)
    x = skdim.id.lPCA(ver="Kaiser").fit(data)
    x = skdim.id.lPCA(ver="broken_stick").fit(data)
    x = skdim.id.lPCA(ver="participation_ratio").fit(data)
    x = skdim.id.lPCA(ver="participation_ratio", center=False).fit(data)


def test_tle_params(data):
    x = skdim.id.TLE().fit(data)
    x = skdim.id.TLE(epsilon=0.01).fit(data)


def test_corrint_params(data):
    x = skdim.id.CorrInt().fit(data)
    x = skdim.id.CorrInt(k1=5, k2=15).fit(data)


def test_danco_params(data):
    x = skdim.id.DANCo().fit(data)
    x = skdim.id.DANCo(fractal=False).fit(data)
    x = skdim.id.DANCo(D=5).fit(data)
    x = skdim.id.DANCo(k=5).fit(data)
    x = skdim.id.DANCo(ver="MIND_MLk").fit(data)
    x = skdim.id.DANCo(ver="MIND_MLi").fit(data)


def test_knn_params(data):
    x = skdim.id.KNN().fit(data)
    x = skdim.id.KNN(k=5).fit(data)
    x = skdim.id.KNN(ps=np.arange(30, 32)).fit(data)
    x = skdim.id.KNN(M=2).fit(data)
    x = skdim.id.KNN(gamma=3).fit(data)


def test_mada_params(data):
    x = skdim.id.MADA().fit(data)


def test_mle_params(data):
    x = skdim.id.MLE().fit(data)
    x = skdim.id.MLE(n=20, sigma=0.1, dnoise="dnoiseGaussH").fit(data)
    x = skdim.id.MLE(unbiased=True).fit(data)
    x = skdim.id.MLE(K=10, neighborhood_based=False).fit(data)


def test_twonn_params(data):
    # to trigger the "n_features>25 condition"
    test_high_dim = np.zeros((len(data), 30))
    test_high_dim[:, : data.shape[1]] = data
    x = skdim.id.TwoNN().fit(test_high_dim)
    x = skdim.id.TwoNN(discard_fraction=0.05).fit(data)


def test_aspointwise(data):
    x = skdim.asPointwise(data, skdim.id.TwoNN(), n_neighbors=50)
    x = skdim.asPointwise(data, skdim.id.TwoNN(), n_neighbors=50, n_jobs=2)
    assert len(x) == len(data)


def test_datasets():
    skdim.datasets.hyperBall(100, 2)
    skdim.datasets.hyperSphere(100, 2)
    skdim.datasets.hyperTwinPeaks(100, 2)
    skdim.datasets.lineDiskBall(100)
    skdim.datasets.swissRoll3Sph(100, 100)
    skdim.datasets.BenchmarkManifolds(noise_type="uniform").generate(n=123)
    skdim.datasets.BenchmarkManifolds(noise_type="normal").generate(
        name="M5b_Helix2d", n=456, dim=3, d=2
    )


def test_fisher_separability_graph(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    import numpy as np
    from sklearn.decomposition import PCA
    from scipy.stats.mstats import winsorize

    ball1 = skdim.datasets.hyperBall(n=1000, d=3, radius=0.5, center=[0, 0, 0]).T
    ball2 = skdim.datasets.hyperBall(
        n=1000, d=6, radius=0.5, center=[1, 0, 0, 0, 0, 0]
    ).T

    _2balls = np.zeros((6, 2000))
    _2balls[:3, :1000] = ball1
    _2balls[:, 1000:2000] = ball2
    X = _2balls.T

    u = PCA().fit_transform(X)
    fishers = skdim.id.FisherS(conditional_number=10000).fit(X)
    ns = fishers.point_inseparability_to_pointID()[0]
    edges, weights = fishers.getSeparabilityGraph()

    nsw = winsorize(ns, limits=(0.01, 0.01), inclusive=(True, True))
    plt.figure(figsize=(10, 5))
    plt.scatter(u[:, 0], u[:, 1], c=nsw)
    fishers.plotSeparabilityGraph(u[:, 0], u[:, 1], edges, alpha=0.2)
    plt.colorbar()
    plt.axis("equal")
    plt.title("PCA on original data")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.show()
