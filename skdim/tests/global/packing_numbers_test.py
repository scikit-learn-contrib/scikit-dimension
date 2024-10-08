import pytest
import numpy as np
import skdim
from skdim.id import PackingNumbers

@pytest.fixture
def data():
    X = np.zeros((100, 10))
    X[:, :5] = skdim.datasets.hyperBall(n=100, d=5, radius=1, random_state=0)
    return X

def test_wrong_radii_values_are_not_accepted(data):
    with pytest.raises(ValueError):
        g = PackingNumbers(r1=0)
        g.fit(data)
    with pytest.raises(ValueError):
        g =PackingNumbers(r1=-1)
        g.fit(data)
    with pytest.raises(ValueError):
        g =PackingNumbers(radius_multiplier=-0)
        g.fit(data)
    with pytest.raises(ValueError):
        g =PackingNumbers(r1=-1)
        g.fit(data)
    with pytest.raises(ValueError):
        g =PackingNumbers(r1=-5, radius_multiplier=0.1)
        g.fit(data)

def test_wrong_iter_number_value_is_not_accepted(data):
    with pytest.raises(ValueError):
        g = PackingNumbers(iter_number=0)
        g.fit(data)
    with pytest.raises(ValueError):
        g = PackingNumbers(iter_number=-1)
        g.fit(data)

def test_wrong_accuracy_value_is_not_accepted(data):
    with pytest.raises(ValueError):
        g = PackingNumbers(accuracy=0)
        g.fit(data)
    with pytest.raises(ValueError):
        g = PackingNumbers(accuracy=-1)
        g.fit(data)

def test_when_both_packing_numbers_equal():
    g = PackingNumbers(r1=1, radius_multiplier=2)
    data = np.array([[0, 0], [2, 3]])
    assert g.fit(data).dimension_ == 0.0

def test_when_packing_numbers_one_and_two():
    g = PackingNumbers(r1=1, radius_multiplier=2)
    data = np.array([[0, 0], [1, 1]])
    assert pytest.approx(g.fit(data).dimension_) == 1.0

def test_when_packing_numbers_three_and_six():
    g = PackingNumbers(r1=1.1, radius_multiplier=2.363636364)
    data = np.array([[0, 0], [0, 1], [0, 2.5], [10, 1], [10, 2], [10, 3.5], [20, 1], [20, 2], [20, 3.5]])
    assert pytest.approx(g.fit(data).dimension_) == np.log(2) / np.log(2.6 / 1.1)
