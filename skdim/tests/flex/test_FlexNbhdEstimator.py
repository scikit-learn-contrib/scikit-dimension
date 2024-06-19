import pytest
import numpy as np
import skdim.id_flex
from inspect import getmembers, isclass

estimators = [o[1] for o in getmembers(skdim.id_flex) if isclass(o[1])]


@pytest.fixture
def data():
    X = np.zeros((100, 10))
    X[:, :5] = skdim.datasets.hyperBall(n=100, d=5, radius=1, random_state=0)
    return X

@pytest.mark.parametrize("estim_class", estimators)
def test_fit_unknown_neighborhood(estim_class, data):
    with pytest.raises(ValueError):
        estimator = estim_class(nbhd_type="wrong_unknown_type")
        estimator.fit_transform(data)

@pytest.mark.parametrize("estim_class", estimators)
def test_fit_unknown_metric(estim_class, data):
    with pytest.raises(ValueError):
        estimator = estim_class(metric="wrong_unknown_metric")
        estimator.fit_transform(data)

@pytest.mark.parametrize("estim_class", estimators)
def test_aggregation_unknown_comb(estim_class, data):
    with pytest.raises(ValueError):
        estimator = estim_class(comb="wrong_unknown_comb")
        estimator.fit_transform(data)

@pytest.mark.parametrize("estim_class", estimators)
def test_get_neigh_unknown_neighborhood(estim_class, data):
    
    with pytest.raises(ValueError):
        estimator = estim_class(nbhd_type="wrong_unknown_type")
        estimator.get_neigh(data)

@pytest.mark.parametrize("estim_class", estimators)
def test_get_neigh_unknown_metric(estim_class, data):
    with pytest.raises(ValueError):
        estimator = estim_class(metric="wrong_unknown_metric")
        estimator.get_neigh(data)

@pytest.mark.parametrize("estim_class", estimators)
def test_aggregation_computes_global_dim(estim_class, data):
    estimator = estim_class()
    estimator.fit_pw(data)
    estimator.aggr()
    assert hasattr(estimator, "dimension_")

@pytest.mark.parametrize("estim_class", estimators)
def test_fit_computes_global_dim(estim_class, data):
    estimator = estim_class()
    estimator.fit(data)
    assert hasattr(estimator, "dimension_")

@pytest.mark.parametrize("estim_class", estimators)
def test_fit_pw_computes_pw_dim(estim_class, data):
    estimator = estim_class()
    estimator.fit_pw(data)
    assert hasattr(estimator, "dimension_pw_")

@pytest.mark.parametrize("estim_class", estimators)
def test_fit_single_job_equal_to_fit(estim_class, data):
    estimator_single = estim_class()
    estimator_single.fit(data)
    estimator_multi = estim_class(n_jobs=4)
    estimator_multi.fit(data)
    assert pytest.approx(estimator_single.dimension_) == pytest.approx(estimator_multi.dimension_)

@pytest.mark.parametrize("estim_class", estimators)
def test_fit_pw_single_job_equal_to_fit_pw(estim_class, data):
    estimator_single = estim_class()
    estimator_single.fit_pw(data)
    estimator_multi = estim_class(n_jobs=4)
    estimator_multi.fit_pw(data)
    assert np.allclose(estimator_multi.dimension_pw_, estimator_single.dimension_pw_)

@pytest.mark.parametrize("estim_class", estimators)
def test_fit_transform_pw_is_fit_pw_plus_transform_pw(estim_class, data):
    estimator = estim_class()
    estimator.fit_pw(data)
    second_estimator = estim_class()
    assert np.allclose(second_estimator.fit_transform_pw(data), estimator.transform_pw(data))
