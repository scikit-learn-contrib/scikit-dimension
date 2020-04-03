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

#import rpy2.robjects.packages as rpackages
#import rpy2.robjects.numpy2ri
#utils = rpackages.importr('utils')
#utils.install_packages('intrinsicDimension')
#utils.install_packages('ider')
#intdimr = rpackages.importr('intrinsicDimension')
#ider = intdimr = rpackages.importr('ider')
#rpy2.robjects.numpy2ri.activate()

@pytest.fixture
def data():
    X = np.zeros((30,10))
    X[:,:5] = skdim.gendata.hyperBall(n_points = 30, n_dim = 5, radius = 1, random_state = 0)
    return X

#test default and non-default parameters 
def test_ess_results(data):
    x=skdim.local_id.ESS().fit(data)
    x2=skdim.local_id.ESS(ver='b').fit(data)
    x3=skdim.local_id.ESS(d=2).fit(data)

#     rx=intdimr.essLocalDimEst(data)
#     rx2=intdimr.essLocalDimEst(data,ver='b')
#     rx3=intdimr.essLocalDimEst(data,d=2)
#     rx_results = np.array([rx[0],rx[1],rx2[0],rx2[1],rx3[0],rx3[1]]).T
    rx_results = np.array([[5.08681575, 0.88546915, 5.11274497, 0.371001, 5.06259738, 0.66653896]])

    assert np.allclose(np.array([x.dimension_,x.essval_,x2.dimension_,x2.essval_,x3.dimension_,x3.essval_]),rx_results)

def test_mind_ml_results(data):
#     all((skdim.local_id.MiND_ML(ver='MLk').fit(data).dimension_ == np.array(intdimr.dancoDimEst(data,k=20,D=10,ver='MIND_MLi')[0]),
#          skdim.local_id.MiND_ML(ver='MLi').fit(data).dimension_ == np.array(intdimr.dancoDimEst(data,k=10,D=5,ver='MIND_MLk')[0])))
    assert all((skdim.local_id.MiND_ML(ver='MLk').fit(data).dimension_ == 3.696083548872364,
                skdim.local_id.MiND_ML(ver='MLi').fit(data).dimension_ == 4))

#def test_mom_results(data):
#    x = skdim.local_id.MOM()
#    x = skdim.local_id.MOM(k=5)

def test_lpca_results(data):
#     assert all((skdim.local_id.lPCA().fit(data).dimension_         == intdimr.pcaLocalDimEst(data,ver='FO')[0][0],
#            skdim.local_id.lPCA(ver='fan').fit(data).dimension_    == intdimr.pcaLocalDimEst(data,ver='fan')[0][0],
#            skdim.local_id.lPCA(ver='maxgap').fit(data).dimension_ == intdimr.pcaLocalDimEst(data,ver='maxgap')[0][0]
#           ))
    assert all((skdim.local_id.lPCA().fit(data).dimension_        == 5,
           skdim.local_id.lPCA(ver='fan').fit(data).dimension_    == 3,
           skdim.local_id.lPCA(ver='maxgap').fit(data).dimension_ == 5
          ))
  
    
def test_corrint_results(data):
#     assert np.isclose(skdim.global_id.CorrInt().fit(data).dimension_,np.array(ider.corint(data,k1=10,k2=20)))
    assert np.isclose(skdim.global_id.CorrInt().fit(data).dimension_,2.9309171335492548)
    
def test_danco_results():
    x = np.zeros(9)
    for i in range(2,11):
        print(i,end='\r')
        X = skdim.gendata.hyperBall(n_points=100,n_dim=i,radius=1,random_state=0)
        x[i-2] = skdim.global_id.DANCo(D=11,fractal=False,random_state=0).fit(X).dimension_
        
    assert np.all(x == np.array([ 2,  3,  4,  5,  6,  7,  8,  9, 10]))
    
#def test_knn_results(data):
#    x = skdim.global_id.KNN().fit(data)
#    x = skdim.global_id.KNN(k=5).fit(data)
#    x = skdim.global_id.KNN(ps=np.arange(30,32)).fit(data)
#    x = skdim.global_id.KNN(M=2).fit(data)
#    x = skdim.global_id.KNN(gamma=3).fit(data)

def test_mada_results(data):
#     assert np.allclose(skdim.global_id.Mada(local=True,k=20).fit(data).dimension_,np.array(ider.mada(data,local=True,k=20)))
#            
    assert np.allclose(skdim.global_id.Mada(local=True,k=20).fit(data).dimension_,
                       np.array([2.73873109, 2.1619804 , 2.29577097, 3.13300883, 2.68340134,
                                 2.7744833 , 2.16936429, 2.63976628, 3.73803225, 2.68782875,
                                 3.3468158 , 3.3375878 , 2.58747076, 3.49101585, 2.84130906,
                                 3.68365681, 3.5694458 , 3.50015857, 2.78155042, 4.47995493,
                                 4.0897491 , 3.79781342, 2.62101409, 3.03665971, 2.84263582,
                                 3.82414291, 2.65283788, 3.46869821, 3.851816  , 3.13222988]))

def test_mle_results(data):
#     assert np.allclose(
#         (skdim.global_id.MLE().fit(data).dimension_,
#         skdim.global_id.MLE(k=5).fit(data).dimension_,
#         skdim.global_id.MLE(n = 20, sigma=.1, dnoise='dnoiseGaussH').fit(data).dimension_,
#         skdim.global_id.MLE(unbiased=True).fit(data).dimension_,
#         skdim.global_id.MLE(K=10, neighborhood_based=False).fit(data).dimension_,
#         skdim.global_id.MLE(neighborhood_aggregation='mean').fit(data).dimension_,
#         ),
#         (intdimr.maxLikGlobalDimEst(data,k=20)[0][0],
#         intdimr.maxLikGlobalDimEst(data,k=5)[0][0],
#         intdimr.maxLikGlobalDimEst(data,k=20,n = 20, sigma=.1, dnoise='dnoiseGaussH')[0][0],
#         intdimr.maxLikGlobalDimEst(data,k=20,unbiased=True)[0][0],
#         intdimr.maxLikGlobalDimEst(data,k=20,K=10, neighborhood_based=False)[0][0],
#         intdimr.maxLikGlobalDimEst(data,k=20,neighborhood_aggregation='mean')[0][0]
#         )
#     )
     assert np.allclose(
        (skdim.global_id.MLE().fit(data).dimension_,
        skdim.global_id.MLE(k=5).fit(data).dimension_,
        skdim.global_id.MLE(n = 20, sigma=.1, dnoise='dnoiseGaussH').fit(data).dimension_,
        skdim.global_id.MLE(unbiased=True).fit(data).dimension_,
        skdim.global_id.MLE(K=10, neighborhood_based=False).fit(data).dimension_,
        skdim.global_id.MLE(neighborhood_aggregation='mean').fit(data).dimension_,
        ),
        (3.341161830518472,
         4.350132038093711,
         3.2755228756927455,
         3.1653112078596046,
         5.312292926569348,
         3.4389424541367437)
    )   
    
def test_tle_results(data):
#     assert np.allclose(np.round(skdim.local_id.TLE().fit(data).dimension_,4),radovanovic_estimators_matlab(data,k=20)['id_tle'])
    assert np.allclose(np.round(skdim.local_id.TLE().fit(data).dimension_,4),
                        np.array([3.1975, 3.0745, 3.0447, 3.3422, 2.9231, 3.7637, 2.9517, 3.2285,
                                  3.8731, 2.9633, 3.2595, 4.1404, 3.1353, 3.671 , 3.6847, 3.9358,
                                  3.9902, 3.7741, 3.5728, 4.2938, 4.2238, 3.9431, 3.0891, 3.4052,
                                  4.0238, 4.7747, 3.9094, 4.9064, 4.0431, 3.4192])
               )
    
def test_twonn_results(data):
    #to trigger the "n_features>25 condition"
    test_high_dim = np.zeros((len(data),30))
    test_high_dim[:,:data.shape[1]] = data
    assert all((np.round(skdim.global_id.TwoNN().fit(test_high_dim).dimension_,5) == 4.05496,
                np.round(skdim.global_id.TwoNN(discard_fraction=0.05).fit(data).dimension_,5) == 4.11323))
