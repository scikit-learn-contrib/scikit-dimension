import pytest
import numpy as np
import skdim.id_flex
from sklearn.datasets import make_swiss_roll

VERSIONS = ["FO", "maxgap", "Kaiser", "broken_stick", "LB"]


@pytest.fixture
def data():
    X = np.zeros((1000, 10))
    X[:, :5] = skdim.datasets.hyperBall(n=1000, d=5, radius=1, random_state=0)
    return X


@pytest.mark.parametrize("ver", VERSIONS)
def test_lpca_results(data, ver):
    pca = skdim.id_flex.lPCA(ver=ver, nbhd_type="eps", radius=0.5)
    assert pca.fit(data).dimension_ == pytest.approx(5, 0.5)


@pytest.mark.parametrize("ver", VERSIONS)
def test_on_swiss_roll(ver):
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    pca = skdim.id_flex.lPCA(ver=ver, nbhd_type="knn", n_neighbors=20)
    estim_dim = pca.fit_transform(swiss_roll_dat[0])
    assert 2.0 == pytest.approx(estim_dim, 0.5)


def test_when_eps_and_knn_almost_equivalent():
    pca = skdim.id_flex.lPCA
    rectangle = np.zeros((4, 4))
    rectangle[1:3, 1] = 2
    rectangle[:2, 0] = 1
    knn_estim = pca(nbhd_type="knn", n_neighbors=2).fit_transform(rectangle)
    eps_estim = pca(nbhd_type="eps", radius=2).fit_transform(rectangle)
    assert knn_estim == pytest.approx(eps_estim)
