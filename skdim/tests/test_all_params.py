#
# BSD 3-Clause License
#
# Copyright (c) 2020, Jonathan Bac
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
import pytest
import numpy as np
import skdim
import matplotlib.pyplot as plt
from inspect import getmembers, isclass
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_random_state
from sklearn.utils.testing import (
    assert_equal,
    assert_array_equal,
    assert_array_almost_equal,
    assert_raises,
    assert_in,
    assert_not_in,
    assert_no_warnings,
)


### test data generation
def test_datasets():
    skdim.datasets.hyperBall(100, 2)
    skdim.datasets.hyperSphere(100, 2)
    skdim.datasets.hyperTwinPeaks(100, 2)
    skdim.datasets.lineDiskBall(100)
    skdim.datasets.swissRoll3Sph(100, 100)
    skdim.datasets.BenchmarkManifolds(noise_type="uniform").generate(n=123, noise=0.1)
    skdim.datasets.BenchmarkManifolds(noise_type="normal").generate(
        name="M5b_Helix2d", n=456, dim=3, d=2, noise=0.3
    )


@pytest.fixture
def data():
    X = np.zeros((100, 10))
    X[:, :5] = skdim.datasets.hyperBall(n=100, d=5, radius=1, random_state=0)
    return X


# test all estimators pass check_estimator
local_class_list = [o[1] for o in getmembers(skdim.lid) if isclass(o[1])]
global_class_list = [o[1] for o in getmembers(skdim.gid) if isclass(o[1])]
estimators = local_class_list + global_class_list


@pytest.mark.parametrize("Estimator", estimators)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)


# test default and non-default parameters
def test_ess_params(data):
    x = skdim.lid.ESS().fit(data)
    x = skdim.lid.ESS(ver="b").fit(data)
    x = skdim.lid.ESS(d=2).fit(data)


def test_fisher_params(data, monkeypatch):
    monkeypatch.setattr(plt, "show", lambda: None)
    x = skdim.lid.FisherS().fit(data)
    x = skdim.lid.FisherS(conditional_number=2).fit(data)
    x = skdim.lid.FisherS(produce_plots=True).fit(data)
    x = skdim.lid.FisherS(project_on_sphere=False).fit(data)
    x = skdim.lid.FisherS(verbose=True).fit(data)
    x = skdim.lid.FisherS(limit_maxdim=True).fit(data)
    x = skdim.lid.FisherS().fit(data).point_inseparability_to_pointID()


def test_mind_ml_params(data):
    x = skdim.gid.MiND_ML()
    x = skdim.gid.MiND_ML(ver="MLi")
    x = skdim.gid.MiND_ML(ver="ML1")
    x = skdim.gid.MiND_ML(D=5)
    x = skdim.gid.MiND_ML(k=5)


def test_mom_params(data):
    x = skdim.lid.MOM()
    x = skdim.lid.MOM(k=5)


def test_lpca_params(data):
    x = skdim.lid.lPCA().fit(data)
    x = skdim.lid.lPCA(ver="fan").fit(data)
    x = skdim.lid.lPCA(ver="ratio").fit(data)
    x = skdim.lid.lPCA(ver="maxgap").fit(data)


def test_tle_params(data):
    x = skdim.lid.TLE().fit(data)
    x = skdim.lid.TLE(epsilon=0.01).fit(data)


def test_corrint_params(data):
    x = skdim.gid.CorrInt().fit(data)
    x = skdim.gid.CorrInt(k1=5, k2=15).fit(data)


def test_danco_params(data):
    x = skdim.gid.DANCo().fit(data)
    x = skdim.gid.DANCo(fractal=False).fit(data)
    x = skdim.gid.DANCo(D=5).fit(data)
    x = skdim.gid.DANCo(k=5).fit(data)
    x = skdim.gid.DANCo(ver="MIND_MLk").fit(data)
    x = skdim.gid.DANCo(ver="MIND_MLi").fit(data)


def test_knn_params(data):
    x = skdim.gid.KNN().fit(data)
    x = skdim.gid.KNN(k=5).fit(data)
    x = skdim.gid.KNN(ps=np.arange(30, 32)).fit(data)
    x = skdim.gid.KNN(M=2).fit(data)
    x = skdim.gid.KNN(gamma=3).fit(data)


def test_mada_params(data):
    x = skdim.lid.MADA().fit(data)
    x = skdim.lid.MADA(k=5).fit(data)
    x = skdim.lid.MADA(comb="average").fit(data)
    x = skdim.lid.MADA(comb="median").fit(data)
    x = skdim.lid.MADA(local=True).fit(data)


def test_mle_params(data):
    x = skdim.gid.MLE().fit(data)
    x = skdim.gid.MLE(k=5).fit(data)
    x = skdim.gid.MLE(n=20, sigma=0.1, dnoise="dnoiseGaussH").fit(data)
    x = skdim.gid.MLE(unbiased=True).fit(data)
    x = skdim.gid.MLE(K=10, neighborhood_based=False).fit(data)
    x = skdim.gid.MLE(neighborhood_aggregation="mean").fit(data)
    x = skdim.gid.MLE(neighborhood_aggregation="median").fit(data)


def test_twonn_params(data):
    # to trigger the "n_features>25 condition"
    test_high_dim = np.zeros((len(data), 30))
    test_high_dim[:, : data.shape[1]] = data
    x = skdim.gid.TwoNN().fit(test_high_dim)
    x = skdim.gid.TwoNN(discard_fraction=0.05).fit(data)


# test auxiliary functions
def test_get_estimators(data):
    assert skdim.get_estimators() == (
        {
            "ESS": skdim.lid._ESS.ESS,
            "FisherS": skdim.lid._FisherS.FisherS,
            "MADA": skdim.lid._MADA.MADA,
            "MOM": skdim.lid._MOM.MOM,
            "TLE": skdim.lid._TLE.TLE,
            "lPCA": skdim.lid._PCA.lPCA,
        },
        {
            "CorrInt": skdim.gid._CorrInt.CorrInt,
            "DANCo": skdim.gid._DANCo.DANCo,
            "KNN": skdim.gid._KNN.KNN,
            "MLE": skdim.gid._MLE.MLE,
            "MiND_ML": skdim.gid._MiND_ML.MiND_ML,
            "TwoNN": skdim.gid._TwoNN.TwoNN,
        },
    )


def test_aspointwise(data):
    x = skdim.asPointwise(data, skdim.lid.lPCA(), n_neighbors=50)
    x = skdim.asPointwise(data, skdim.lid.lPCA(), n_neighbors=50, n_jobs=2)
    assert len(x) == len(data)


def test_datasets():
    x = skdim.datasets.hyperSphere(n_points=100, n_dim=2, center=[], random_state=None)
    x = skdim.datasets.hyperTwinPeaks(
        n_points=100, n_dim=2, height=1, random_state=None
    )
    x = skdim.datasets.swissRoll3Sph(
        Ns=50, Nsph=50, a=1, b=2, nturn=1.5, h=4, random_state=None
    )
    x = skdim.datasets.swissRoll3Sph(
        Ns=0, Nsph=50, a=1, b=2, nturn=1.5, h=4, random_state=None
    )
    x = skdim.datasets.lineDiskBall(n_points=100, random_state=None)

