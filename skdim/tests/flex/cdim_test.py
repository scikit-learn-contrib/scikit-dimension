import pytest
import numpy as np
import skdim.id_flex
from sklearn.datasets import make_swiss_roll
import skdim.errors


def test_on_swiss_roll_knn():
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    cdim = skdim.id_flex.CDim(nbhd_type="knn", n_neighbors=10)
    estim_dim = cdim.fit_transform(swiss_roll_dat[0])
    # The expected value is taken from Yang et al
    assert 2.0 == pytest.approx(estim_dim, 0.02)


def test_on_swiss_roll_eps():
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    cdim = skdim.id_flex.CDim(nbhd_type="eps", radius=3.0)
    estim_dim = cdim.fit_transform(swiss_roll_dat[0])
    assert 2.0 == pytest.approx(estim_dim, 0.02)
