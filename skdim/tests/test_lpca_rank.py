# SPDX-License-Identifier: BSD-3-Clause
import numpy as np
import pytest
from skdim.id._PCA import lPCA


def test_fan_does_not_mutate_input():
    ev = np.array([10.0, 5.0, 1.0, 0.1, 0.05])
    ev0 = ev.copy()
    lPCA(ver="Fan", fit_explained_variance=True).fit(ev)
    np.testing.assert_array_equal(ev, ev0)


def test_broken_stick_hand_example():
    # normalized [10,1,1,1]/13 = [.769,.077,.077,.077]; broken stick for p=4 is
    # b_k = (1/4) sum_{j>=k} 1/j = [.521,.271,.146,.063]: only the first exceeds -> 1
    est = lPCA(ver="broken_stick", fit_explained_variance=True).fit(np.array([10.0, 1, 1, 1]))
    assert est.dimension_ == 1
