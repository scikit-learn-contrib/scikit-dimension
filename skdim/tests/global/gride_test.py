import pytest
import numpy as np
import skdim
from skdim.id import Gride
from sklearn.datasets import make_swiss_roll
from sklearn.exceptions import NotFittedError
from dadapy.data import Data

def test_wrong_n1_n2_values_are_not_accepted():
    with pytest.raises(ValueError):
        g = Gride(n1=0)
    with pytest.raises(ValueError):
        g =Gride(n1=-1)
    with pytest.raises(ValueError):
        g = Gride(n1 = 2, n2=1)
    with pytest.raises(ValueError):
        g = Gride(n1 = 0.5)
    with pytest.raises(ValueError):
        g = Gride(n1 = 2, n2=7.1)

def test_wrong_dimenstion_intervals_are_not_accepted():
    with pytest.raises(ValueError):
        g = Gride(d0=-1)
        g = Gride(d1=2, d2=1)

def test_multiple_single_scale_estim_eq_multiscale():
    np.random.seed(782)
    X = np.random.normal(0, 1, (100, 3))
   
    gride = Gride(range_max = 32)
    gride.fit(X)
    dim_evolution = gride.transform_multiscale()
    for i in range(4):
        g = Gride(n1=2**i).fit(X)
       
        assert pytest.approx(dim_evolution[i]) == g.transform()

def test_multiscale_not_computed_without_max_range():
    np.random.seed(3127)
    gride = Gride()
    X = np.random.normal(0, 1, (1000, 3))
    gride.fit(X)
    assert not hasattr(gride, 'dimension_array_')
    with pytest.raises(NotFittedError):
        gride.transform_multiscale()

def test_results_similar_to_dadapy_for_hyperspher():
    np.random.seed(782)
    X = np.random.normal(0, 1, (1000, 3))
    data = Data(X)
    grid = Gride(range_max=64)
    grid.fit(X)
    dim_evolution = grid.transform_multiscale()
    id_list, _, _ = data.return_id_scaling_gride(range_max=64)
    assert np.allclose(id_list, dim_evolution, atol=0.1)

def test_results_similar_to_dadapy_for_gaussian():
    np.random.seed(782)
    X = np.random.normal(0, 1, (1000, 3))
    data = Data(X)
    grid = Gride(range_max=64)
    grid.fit(X)
    dim_evolution = grid.transform_multiscale()
    id_list, _, _ = data.return_id_scaling_gride(range_max=64)
    assert np.allclose(id_list, dim_evolution, atol=0.1)

def test_avg_results_for_hypercube():
    # Values are taken from original paper, Figure 4
    MONTE_CARLO_SIMULATION_NUM = 50
    SAMPLE_SIZE = 10000
    N1 = 100
    real_dims = [2 * i for i in range(1, 6)]
    hypercubes = [np.random.uniform(low=0.0, high=1.0, size=(SAMPLE_SIZE, dim)) for dim in real_dims]
    estimates = []
    for i in range(len(real_dims)):
        acc = 0.0
        gride = Gride(n1=N1)
        for step in range(MONTE_CARLO_SIMULATION_NUM):
            hypercube = np.random.uniform(low=0.0, high=1.0, size=(SAMPLE_SIZE, real_dims[i]))
            acc += gride.fit_transform(hypercube)
        estimates.append(acc / MONTE_CARLO_SIMULATION_NUM)
    assert pytest.approx(estimates[0], 0.1) == 2.0
    assert pytest.approx(estimates[1], 0.1) == 3.5
    assert pytest.approx(estimates[2], 0.1) == 5.0
    assert pytest.approx(estimates[3], 0.2) == 6.45
    assert pytest.approx(estimates[4], 0.25) == 7.75