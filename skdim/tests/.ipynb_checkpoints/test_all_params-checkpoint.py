#
# MIT License
#
# Copyright (c) 2020 Jonathan Bac
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
import pytest
import numpy as np
import skdim
import matplotlib.pyplot as plt
from inspect import getmembers, isclass
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_random_state
from sklearn.utils.testing import (assert_equal,
                                   assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_raises,
                                   assert_in,
                                   assert_not_in,
                                   assert_no_warnings)



@pytest.fixture
def data():
    X = np.zeros((100,10))
    X[:,:5] = skdim.gendata.hyperBall(n_points = 100, n_dim = 5, radius = 1, random_state = 0)
    return X

#test all estimators pass check_estimator
local_class_list = [o[1]
                    for o in getmembers(skdim.local_id) if isclass(o[1])]
global_class_list = [o[1]
                     for o in getmembers(skdim.global_id) if isclass(o[1])]
estimators = local_class_list+global_class_list

@pytest.mark.parametrize("Estimator", estimators)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

#test default and non-default parameters 
def test_ess_params(data):
    x = skdim.local_id.ESS().fit(data)
    x = skdim.local_id.ESS(ver='b').fit(data)
    x = skdim.local_id.ESS(d=2).fit(data)
    
def test_fisher_params(data,monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    x = skdim.local_id.FisherS().fit(data)
    x = skdim.local_id.FisherS(conditional_number=2).fit(data)
    x = skdim.local_id.FisherS(produce_plots=True).fit(data) #travis ci macosx stuck when producing plots 
    x = skdim.local_id.FisherS(project_on_sphere=False).fit(data)
    x = skdim.local_id.FisherS(verbose=True).fit(data)
    x = skdim.local_id.FisherS(limit_maxdim=True).fit(data)
    x = skdim.local_id.FisherS().fit(data).point_inseparability_to_pointID()

def test_mind_ml_params(data):
    x = skdim.local_id.MiND_ML()
    x = skdim.local_id.MiND_ML(ver='MLi')
    x = skdim.local_id.MiND_ML(ver='ML1')
    x = skdim.local_id.MiND_ML(D=5)
    x = skdim.local_id.MiND_ML(k=5)
    
def test_mom_params(data):
    x = skdim.local_id.MOM()
    x = skdim.local_id.MOM(k=5)
    
def test_lpca_params(data):
    x = skdim.local_id.lPCA().fit(data)
    x = skdim.local_id.lPCA(ver='fan').fit(data)
    x = skdim.local_id.lPCA(ver='ratio').fit(data)
    x = skdim.local_id.lPCA(ver='maxgap').fit(data)
    
def test_tle_params(data):
    x = skdim.local_id.TLE().fit(data)
    x = skdim.local_id.TLE(epsilon=.01).fit(data)
    
def test_corrint_params(data):
    x = skdim.global_id.CorrInt().fit(data)
    x = skdim.global_id.CorrInt(k1=5,k2=15).fit(data)
    
def test_danco_params(data):
    x = skdim.global_id.DANCo().fit(data)
    x = skdim.global_id.DANCo(fractal=False).fit(data)
    x = skdim.global_id.DANCo(D=5).fit(data)
    x = skdim.global_id.DANCo(k=5).fit(data)
    x = skdim.global_id.DANCo(ver='MIND_MLk').fit(data)
    x = skdim.global_id.DANCo(ver='MIND_MLi').fit(data)


def test_knn_params(data):
    x = skdim.global_id.KNN().fit(data)
    x = skdim.global_id.KNN(k=5).fit(data)
    x = skdim.global_id.KNN(ps=np.arange(30,32)).fit(data)
    x = skdim.global_id.KNN(M=2).fit(data)
    x = skdim.global_id.KNN(gamma=3).fit(data)

def test_mada_params(data):
    x = skdim.global_id.Mada().fit(data)
    x = skdim.global_id.Mada(k=5).fit(data)
    x = skdim.global_id.Mada(comb='average').fit(data)
    x = skdim.global_id.Mada(comb='median').fit(data)
    x = skdim.global_id.Mada(local=True).fit(data)

def test_mle_params(data):
    x = skdim.global_id.MLE().fit(data)
    x = skdim.global_id.MLE(k=5).fit(data)
    x = skdim.global_id.MLE(n = 20, sigma=.1, dnoise='dnoiseGaussH').fit(data)
    x = skdim.global_id.MLE(unbiased=True).fit(data)
    x = skdim.global_id.MLE(K=10, neighborhood_based=False).fit(data)
    x = skdim.global_id.MLE(neighborhood_aggregation='mean').fit(data)
    x = skdim.global_id.MLE(neighborhood_aggregation='median').fit(data)
    
def test_twonn_params(data):
    #to trigger the "n_features>25 condition"
    test_high_dim = np.zeros((len(data),30))
    test_high_dim[:,:data.shape[1]] = data
    x = skdim.global_id.TwoNN().fit(test_high_dim)
    x = skdim.global_id.TwoNN(discard_fraction=0.05).fit(data)

#test auxiliary functions
def test_get_estimators(data):
    assert skdim.get_estimators() == ({'ESS': skdim.local_id._ESS.ESS,
                                 'FisherS': skdim.local_id._FisherS.FisherS,
                                 'MOM': skdim.local_id._MOM.MOM,
                                 'MiND_ML': skdim.local_id._MiND_ML.MiND_ML,
                                 'TLE': skdim.local_id._TLE.TLE,
                                 'lPCA': skdim.local_id._PCA.lPCA},
                                {'CorrInt': skdim.global_id._CorrInt.CorrInt,
                                 'DANCo': skdim.global_id._DANCo.DANCo,
                                 'KNN': skdim.global_id._KNN.KNN,
                                 'MLE': skdim.global_id._MLE.MLE,
                                 'Mada': skdim.global_id._Mada.Mada,
                                 'TwoNN': skdim.global_id._TwoNN.TwoNN})

def test_aspointwise(data):   
    x = skdim.asPointwise(data,skdim.local_id.lPCA(),n_neighbors=50)
    x = skdim.asPointwise(data,skdim.local_id.lPCA(),n_neighbors=50,n_jobs=2)
    assert len(x) == len(data)


def test_gendata():
    x = skdim.gendata.hyperSphere(n_points = 100 , n_dim = 2, center=[], random_state=None)
    x = skdim.gendata.hyperTwinPeaks(n_points = 100 , n_dim = 2, height = 1, random_state=None)
    x = skdim.gendata.swissRoll3Sph(Ns=50, Nsph=50, a = 1, b = 2, nturn = 1.5, h = 4, random_state=None)
    x = skdim.gendata.swissRoll3Sph(Ns=0, Nsph=50, a = 1, b = 2, nturn = 1.5, h = 4, random_state=None)
    x = skdim.gendata.lineDiskBall(n_points = 100, random_state=None)