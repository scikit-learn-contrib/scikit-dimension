import pytest
import numpy as np
import skdim
from sklearn.datasets import make_swiss_roll


def __generate_distance_matrix(size, threshold):
    # Generate a random distance matrix with values between 0 and 1
    distance_matrix = np.random.rand(size, size)
    
    # Scale the values to fit the desired range (e.g., 0 to 100)
    distance_matrix = distance_matrix * 100
    
    # Ensure that each value is greater than the threshold
    distance_matrix = np.maximum(distance_matrix, threshold)
    
    # Fill diagonal with zeros (optional, depends on your use case)
    np.fill_diagonal(distance_matrix, 0)
    
    return distance_matrix

def test_on_swiss_roll():
    np.random.seed(782)
    swiss_roll_dat = make_swiss_roll(1000)
    mle = skdim.id_flex.MLE_basic(11)
    estim_dim = mle.fit_transform(swiss_roll_dat, n_neighbors=20)
    # The expected value is taken from Levina and Bickel's paper
    assert 2.1 == pytest.approx(estim_dim, 0.1)

def test_on_equal_distances():
    SIZE = 5
    distances = np.full((SIZE, SIZE), 0.75)
    for i in range(SIZE):
        distances[i,i] = 0.0
    mle = skdim.id_flex.MLE_basic()
    estim_dim = mle.fit_transform(distances, metric="precomputed", n_neighbors=3)
    assert  0.0 == pytest.approx(estim_dim)

def test_on_exponential_seq_of_distances():
    np.random.seed(356)
    SIZE = 5
    dist_matrix = __generate_distance_matrix(5, 0)
    for i in range(1, SIZE):
        dist_matrix[0, i] = np.exp(i)
        dist_matrix[i, 0] = np.exp(i)
    estim_dim = mle.fit_transform(dist_matrix, nbhd_type = 'knn', n_neighbors=2)
    assert 0.4 == pytest.approx(estim_dim)
    estim_dim = mle.fit_transform(dist_matrix, nbhd_type = 'knn', n_neighbors=3)
    assert 0.48 == pytest.approx(estim_dim)
    estim_dim = mle.fit_transform(dist_matrix, nbhd_type = 'knn', n_neighbors=3)
    assert 36.0 / 125.0  == pytest.approx(estim_dim)

def test_exception_is_raised_when_neighbourhoods_empty():
    np.random.seed(123)
    dist_matrix = __generate_distance_matrix(10, 12)
    mle = skdim.id_flex.MLE_basic()
    pytest.raises(ValueError, mle.fit_transform(dist_matrix, nhbd_type = 'eps', radius = 5))

def test_if_eps_and_knn_are_equivalent():
    mle = skdim.id_flex.MLE_basic()
    square = np.zeros((4,3))
    square[2:,0,0,0,0] = 1
    square[2,1,0,0,0] = 1
    square[1,1,0,0,0] = 1
    knn_estim = mle.fit_transform(square, nbhd_type = 'knn', n_neighbors=2)
    eps_estim = mle.fit_transform(square, nbhd_type = 'knn', radius=1)
    assert knn_estim == pytest.approx(eps_estim)
