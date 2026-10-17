# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from skdim.datasets import hyperBall
from skdim.id import DANCo
from skdim.id._DANCo import _FIT_TABLE


def test_table_rows_reproduced_at_grid_cardinalities():
    t = np.load(_FIT_TABLE)
    est = DANCo(ver="DANCoFit", D=12)
    est._k, est._D = 10, 12
    for j in (0, 5, len(t["cardinalities"]) - 1):
        cal = est._fit_calibration(int(t["cardinalities"][j]))
        for key in ("dhat", "mu_nu", "mu_tau"):
            np.testing.assert_allclose([c[key] for c in cal["calibration_data"]], t[key][:12, j], rtol=1e-12)


@pytest.mark.parametrize("d", [2, 5, 10])
def test_ground_truth(d):
    # tolerance fixed before running: within 0.5 of the true dimension on 500 points
    X = np.c_[hyperBall(500, d, random_state=0), np.zeros((500, 3))]
    assert abs(DANCo(ver="DANCoFit").fit(X).dimension_ - d) < 0.5


def test_out_of_table_raises():
    X = np.c_[hyperBall(100, 3, random_state=0), np.zeros((100, 2))]
    with pytest.raises(ValueError, match="k=10"):
        DANCo(ver="DANCoFit", k=5).fit(X)
    with pytest.raises(ValueError, match="D <= 100"):
        DANCo(ver="DANCoFit", D=150).fit(X)
    with pytest.raises(ValueError, match="n_samples"):
        DANCo(ver="DANCoFit").fit(X[:15])
