import pytest
import numpy as np
import skdim.id_flex
import geomle as gm
from geomle import DataGenerator

__K1 = 20
__K2 = 55
__AVG_STEPS = __K2 - __K1 + 1

@pytest.fixture
def data():
    X = np.zeros((100, 10))
    X[:, :5] = skdim.datasets.hyperBall(n=100, d=5, radius=1, random_state=0)
    return X

def test_wrong_k1_k2_values_are_not_accepted(data):
    with pytest.raises(ValueError):
        est = skdim.id_flex.GeoMle(average_steps=-1)
        est.fit(data)
    with pytest.raises(ValueError):
        est = skdim.id_flex.GeoMle(np.random.rand(100, 2), n_neighbors=-1)
        est.fit(data)
    with pytest.raises(ValueError):
        est = skdim.id_flex.GeoMle(np.random.rand(100, 2), n_neighbors=0)
        est.fit(data)

def test_non_positive_polynomial_degrees_are_not_accepted(data):
    with pytest.raises(ValueError):
        est = skdim.id_flex.GeoMle(interpolation_degree=-1)
        est.fit(data)
    with pytest.raises(ValueError):
        est = skdim.id_flex.GeoMle(interpolation_degree=0)
        est.fit(data)

def test_negative_bootstrap_num_is_not_accepted(data):
    with pytest.raises(ValueError):
        est = skdim.id_flex.GeoMle(bootstrap_num=-3)
        est.fit(data)

def test_throw_exception_when_dataset_is_too_small_for_k1_k2():
    X = np.random.rand(10, 2)
    GeoMle = skdim.id_flex.GeoMle(n_neighbors=11)
    with pytest.raises(ValueError):
        GeoMle.fit(X)
    GeoMle = skdim.id_flex.GeoMle(average_steps=3, n_neighbors=9)
    with pytest.raises(ValueError):
        GeoMle.fit(X)

def test_on_swiss_sphere():
    # Expected result comes from origginal paper, Table 3
    np.random.seed(121)
    acc = 0.0
    DG = DataGenerator()
    for i in range(10):
        data = DG.gen_data('Sphere', 1000, 15, 10)
        GeoMle = skdim.id_flex.GeoMle(average_steps=__AVG_STEPS, n_neighbors=__K1)
        GeoMle.fit(data)
        acc += GeoMle.transform()
    assert pytest.approx(acc/10, 0.1) == 10.2

def test_on_swiss_roll():
    # Expected result comes from origginal paper, Table 3
    np.random.seed(110)
    acc = 0.0
    DG = DataGenerator()
    for i in range(10):
        data = DG.gen_data('Roll', 1000, 3, 2)
        GeoMle = skdim.id_flex.GeoMle(average_steps=__AVG_STEPS, bootstrap_num=3, n_neighbors=__K1)
        GeoMle.fit(data)
        acc += GeoMle.transform()
    assert pytest.approx(acc / 10, 0.1) == 2.0

def test_is_like_original_GeoMle_on_sphere():
    np.random.seed(782)
    DG = DataGenerator()
    data = DG.gen_data('Sphere', 1000, 2, 1)
    GeoMle = skdim.id_flex.GeoMle(average_steps = __AVG_STEPS, n_neighbors=__K1)
    GeoMle.fit(data)
    assert pytest.approx(GeoMle.transform(), 0.05) == gm.geomle(data, __K1, __K2, nb_iter1=1, nb_iter2=20).mean()

def test_is_like_original_GeoMle_on_affine():
    np.random.seed(122)
    DG = DataGenerator()
    data = DG.gen_data('Affine', 1000, 7, 3)
    GeoMle = skdim.id_flex.GeoMle(average_steps = __AVG_STEPS, bootstrap_num=10, n_neighbors=__K1)
    GeoMle.fit(data)
    assert pytest.approx(GeoMle.transform(), 0.2) == gm.geomle(data, __K1, __K2, nb_iter1=1, nb_iter2=10).mean()

def test_is_like_original_GeoMle_on_spiral():
    np.random.seed(782)
    DG = DataGenerator()
    data = DG.gen_data('Spiral', 1000, 15, 1)
    GeoMle = skdim.id_flex.GeoMle(average_steps=__AVG_STEPS, bootstrap_num=20, n_neighbors=__K1)
    GeoMle.fit(data)
    assert pytest.approx(GeoMle.transform(), 0.2) == gm.geomle(data, __K1, __K2, nb_iter1=1, nb_iter2=20).mean()
