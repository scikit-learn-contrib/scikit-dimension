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

import rpy2.robjects.packages as rpackages
import rpy2.robjects.numpy2ri
utils = rpackages.importr('utils')
utils.install_packages('intrinsicDimension')
utils.install_packages('ider')
intdimr = rpackages.importr('intrinsicDimension')
ider = intdimr = rpackages.importr('ider')
rpy2.robjects.numpy2ri.activate()

@pytest.fixture
def data():
    X = np.zeros((30,10))
    X[:,:5] = skdim.randball(n_points = 30, n_dim = 5, radius = 1, random_state = 0)
    return X

#test default and non-default parameters 
def test_ess_results(data):
    x=skdim.local_id.ESS().fit(data)
    x2=skdim.local_id.ESS(ver='b').fit(data)
    x3=skdim.local_id.ESS(d=2).fit(data)

    rx=intdimr.essLocalDimEst(data)
    rx2=intdimr.essLocalDimEst(data,ver='b')
    rx3=intdimr.essLocalDimEst(data,d=2)

    assert np.allclose(np.array([x.dimension_,x.essval_,x2.dimension_,x2.essval_,x3.dimension_,x3.essval_]),
                       np.array([rx[0],rx[1],rx2[0],rx2[1],rx3[0],rx3[1]]).T)

#def test_mind_ml_results(data):
#    x = skdim.local_id.MiND_ML()
#    x = skdim.local_id.MiND_ML(ver='MLi')
#    x = skdim.local_id.MiND_ML(ver='ML1')
#    x = skdim.local_id.MiND_ML(D=5)
#    x = skdim.local_id.MiND_ML(k=5)
    
#def test_mom_results(data):
#    x = skdim.local_id.MOM()
#    x = skdim.local_id.MOM(k=5)
    
def test_lpca_results(data):
    assert(skdim.local_id.lPCA().fit(data)             == intdimr.pcaLocalDimEst(data,ver='FO')[0][0],
           skdim.local_id.lPCA(ver='fan').fit(data)    == intdimr.pcaLocalDimEst(data,ver='fan')[0][0],
           skdim.local_id.lPCA(ver='maxgap').fit(data) == intdimr.pcaLocalDimEst(data,ver='maxgap')[0][0]
          )
    
#def test_tle_results(data):
#    x = skdim.local_id.TLE().fit(data)
#    x = skdim.local_id.TLE(epsilon=.01).fit(data)
    
def test_corrint_results(data):
    assert np.isclose(skdim.global_id.CorrInt().fit(data).dimension_,np.array(ider.corint(data,k1=10,k2=20)))
    
#def test_danco_results(data):
#    x = skdim.global_id.DANCo().fit(data)
#    x = skdim.global_id.DANCo(fractal=False).fit(data)
#    x = skdim.global_id.DANCo(D=5).fit(data)
#    x = skdim.global_id.DANCo(k=5).fit(data)
#    x = skdim.global_id.DANCo(ver='MIND_MLk').fit(data)
#    x = skdim.global_id.DANCo(ver='MIND_MLi').fit(data)

#def test_knn_results(data):
#    x = skdim.global_id.KNN().fit(data)
#    x = skdim.global_id.KNN(k=5).fit(data)
#    x = skdim.global_id.KNN(ps=np.arange(30,32)).fit(data)
#    x = skdim.global_id.KNN(M=2).fit(data)
#    x = skdim.global_id.KNN(gamma=3).fit(data)

def test_mada_results(data):
    assert all((np.allclose(skdim.global_id.Mada(local=True,k=20).fit(data).dimension_,np.array(ider.mada(data,local=True,k=20))),
                np.allclose(skdim.global_id.Mada(local=True,k=20,comb='median').fit(data).dimension_,np.array(ider.mada(data,local=True,k=20,comb='median'))),
                np.allclose(skdim.global_id.Mada(local=True,k=20,comb='vote').fit(data).dimension_,np.array(ider.mada(data,local=True,k=20,comb='vote')))
               ))

def test_mle_results(data):
    assert np.allclose(
        (skdim.global_id.MLE().fit(data).dimension_,
        skdim.global_id.MLE(k=5).fit(data).dimension_,
        skdim.global_id.MLE(n = 20, sigma=.1, dnoise='dnoiseGaussH').fit(data).dimension_,
        skdim.global_id.MLE(unbiased=True).fit(data).dimension_,
        skdim.global_id.MLE(K=10, neighborhood_based=False).fit(data).dimension_,
        skdim.global_id.MLE(neighborhood_aggregation='mean').fit(data).dimension_,
        ),
        (intdimr.maxLikGlobalDimEst(data,k=20)[0][0],
        intdimr.maxLikGlobalDimEst(data,k=5)[0][0],
        intdimr.maxLikGlobalDimEst(data,k=20,n = 20, sigma=.1, dnoise='dnoiseGaussH')[0][0],
        intdimr.maxLikGlobalDimEst(data,k=20,unbiased=True)[0][0],
        intdimr.maxLikGlobalDimEst(data,k=20,K=10, neighborhood_based=False)[0][0],
        intdimr.maxLikGlobalDimEst(data,k=20,neighborhood_aggregation='mean')[0][0]
        )
    )
    
#def test_twonn_results(data):
#    #to trigger the "n_features>25 condition"
#    test_high_dim = np.zeros((len(data),30))
#    test_high_dim[:,:data.shape[1]] = data
#    x = skdim.global_id.TwoNN().fit(test_high_dim)
#    x = skdim.global_id.TwoNN(discard_fraction=0.05).fit(data)
