# SPDX-License-Identifier: BSD-3-Clause
# Adapted from the MATLAB code of Amsaleg, Chelly, Houle, Kawarabayashi, Radovanovic and Treeratanajaru
# (https://github.com/radacha/tle; doi:10.1137/1.9781611975673.21, arXiv:2209.14475),
# originally released under CC BY-SA 4.0; see LICENSES_bundled.txt
import numpy as np
from .._commonfuncs import LocalEstimator


class MOM(LocalEstimator):
    """Intrinsic dimension estimation using the Method Of Moments algorithm. [Amsaleg2018]_ [IDRadovanović]_

    Parameters
    ----------
    None
    """

    def _fit(self, **kwargs):
        self.dimension_pw_ = self._mom(kwargs["dists"])

    def _mom(self, dists):
        # dists - nearest-neighbor distances (k x 1, or k x n), sorted
        w = dists[:, -1]
        m1 = np.sum(dists, axis=1) / dists.shape[1]
        ID = -m1 / (m1 - w)
        return ID
