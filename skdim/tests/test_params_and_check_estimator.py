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
# utils.install_packages('intrinsicDimension')
# utils.install_packages('ider')
#intdimr = rpackages.importr('intrinsicDimension')
#ider = intdimr = rpackages.importr('ider')
# rpy2.robjects.numpy2ri.activate()

@pytest.fixture
def data():
    X = np.zeros((100,10))
    X[:,:5] = skdim.randball(n_points = 100, n_dim = 5, radius = 1, random_state = 0)
    return X

#test all estimators pass check_estimator
local_class_list = [o[1]
                    for o in getmembers(skdim.local_id) if isclass(o[1])]
global_class_list = [o[1]
                     for o in getmembers(skdim.global_id) if isclass(o[1])]
estimators = local_class_list+global_class_list

##test equality with R version#
# def test_ess(data):
#    res = intdimr.essLocalDimEst(data)
#    r_ess = dict(zip(res.names,[np.array(i) for i in res]))
#    assert (r_ess['dim.est'] == ess.dimension_ and r_ess['ess'] == ess.essval_)
# def test_corint(data):
#    corint = global_id.CorrInt(k1=10,k2=20).fit(data)
#    r_corint = np.array(ider.corint(data,k1=10,k2=20))
#    assert_equal(r_corint, corint.dimension_)
#
# def test_mada(data):
#    r_mada = np.array(ider.mada(data,local=True,k=20))
#    mada = global_id.Mada(local=True,k=20).fit(data)
#    assert_array_almost_equal(r_mada,mada.dimension_)
#
# def test_mle():
#    #test base params
#    assert np.isclose(        maxLikGlobalDimEst(data, k),
#                      intdimr.maxLikGlobalDimEst(data, k)[0][0])
#
#    #test dnoise+sigma
#    assert np.isclose(        maxLikGlobalDimEst(data, k, dnoise = 'dnoiseGaussH', sigma = 0.2),
#                      intdimr.maxLikGlobalDimEst(data, k, dnoise = 'dnoiseGaussH', sigma = 0.2)[0][0])
#
#    #test neighborhood_aggregation = 'mean'
#    assert np.isclose(        maxLikGlobalDimEst(data, k, neighborhood_aggregation = 'mean'),
#                      intdimr.maxLikGlobalDimEst(data, k, neighborhood_aggregation = 'mean')[0][0])
#
#    #test unbiased = True
#    assert np.isclose(        maxLikGlobalDimEst(data, k, unbiased = True),
#                      intdimr.maxLikGlobalDimEst(data, k, unbiased = True)[0][0])
#
#    #test neighborhood_based = 'False'
#    assert np.isclose(        maxLikGlobalDimEst(data, k, neighborhood_based = False),
#                      intdimr.maxLikGlobalDimEst(data, k, neighborhood_based = False)[0][0])
#
#