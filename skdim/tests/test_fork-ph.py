# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

import skdim
from skdim.id._PH import PH
from skdim.tests.test_all_params import EXPECTED_FAILED_CHECKS

FAST = dict(n_range_min=0.1, nsteps=10, subsamples=10, random_state=0)


# tolerance fixed before running: 20% relative error at n=500
@pytest.mark.parametrize("d", [2, 3, 5])
@pytest.mark.parametrize("shape", ["cube", "ball"])
def test_ph_ground_truth(d, shape):
    if shape == "cube":
        X = np.random.default_rng(d).uniform(size=(500, d))
    else:
        X = skdim.datasets.hyperBall(n=500, d=d, random_state=d)
    est = PH(**FAST).fit(X).dimension_
    assert abs(est - d) < 0.2 * d, est


def test_ph_reproducible_across_n_jobs():
    X = skdim.datasets.hyperBall(n=200, d=3, random_state=0)
    a = PH(**FAST, n_jobs=1).fit(X)
    b = PH(**FAST, n_jobs=2).fit(X)
    assert a.dimension_ == b.dimension_
    np.testing.assert_array_equal(a.y_, b.y_)
    assert PH(**FAST).fit(X).dimension_ == a.dimension_


def test_ph_params():
    X = skdim.datasets.hyperBall(n=100, d=3, random_state=0)
    dims = PH(**FAST, alpha=(0.5, 1.0, 1.5)).fit(X).dimension_
    assert dims.shape == (3,)
    assert np.isclose(dims[1], PH(**FAST, alpha=1.0).fit(X).dimension_)
    est = PH(n_range_min=10, n_range_max=100, range_type="num", nsteps=5, subsamples=2, random_state=0).fit(X)
    assert (est.nmin_, est.nmax_) == (10, 100)
    # metric is honoured
    assert PH(**FAST, metric="cityblock").fit(X).dimension_ != PH(**FAST).fit(X).dimension_
    clone(PH(alpha=[1.0, 2.0]))
    for kw in [
        dict(alpha=0),
        dict(alpha=(1.0, -1.0)),
        dict(nsteps=1),
        dict(subsamples=0),
        dict(range_type="x"),
        dict(n_range_min=1, n_range_max=50, range_type="num"),
        dict(n_range_min=10, n_range_max=1000, range_type="num"),
        dict(n_range_min=-0.1),
        dict(n_range_max=1.1),
        dict(n_range_min=0.5, n_range_max=0.4),
    ]:
        with pytest.raises(ValueError):
            PH(**kw).fit(X)


def test_ph_check_estimator():
    check_estimator(PH(nsteps=5, subsamples=2), expected_failed_checks=EXPECTED_FAILED_CHECKS)
