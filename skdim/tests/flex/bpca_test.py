import pytest
import numpy as np
from skdim.id_flex import lBPCA
from sklearn.datasets import make_swiss_roll
from skdim.datasets import hyperBall

VERSIONS = ["FO", "maxgap", "Kaiser", "broken_stick"]


@pytest.fixture
def data():
    X = np.zeros((1000, 10))
    X[:, :5] = hyperBall(n=1000, d=5, radius=1, random_state=0)
    return X


#@pytest.mark.parametrize("ver", VERSIONS)
def test_lbpca_results(data):
    pca = lBPCA(nbhd_type="eps", radius=0.5)
    assert pca.fit(data).dimension_ == pytest.approx(5, 0.5)


#@pytest.mark.parametrize("ver", VERSIONS)
def test_on_swiss_roll():
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    pca = lBPCA(nbhd_type="knn", n_neighbors=20)
    estim_dim = pca.fit_transform(swiss_roll_dat[0])
    assert 2.0 == pytest.approx(estim_dim, 0.5)

