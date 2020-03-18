import pytest
import numpy as np
from skdim import local_id, global_id
from inspect import getmembers, isclass
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import (assert_equal,
                                   assert_array_equal,
                                   assert_array_almost_equal,
                                   assert_raises,
                                   assert_in,
                                   assert_not_in,
                                   assert_no_warnings)

@pytest.fixture
def data():
    return np.random.random((1000,10))

#test all estimators pass check_estimator
local_class_list = [o[1] for o in getmembers(local_id) if isclass(o[1])]
global_class_list = [o[1] for o in getmembers(global_id) if isclass(o[1])]
estimators = local_class_list+global_class_list
    
@pytest.mark.parametrize("Estimator", estimators)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

#test default and non-default parameters
def test_fisher_params(data):
    x = local_id.FisherS().fit(data)
    x = local_id.FisherS(ConditionalNumber=2).fit(data)
    x = local_id.FisherS(ProducePlots=True).fit(data)
    x = local_id.FisherS(ProjectOnSphere=False).fit(data)
    x = local_id.FisherS(ncomp=True).fit(data)
    x = local_id.FisherS(limit_maxdim=True).fit(data)
        
def test_lPCA_params(data):
    x = local_id.lPCA().fit(data)
    x = local_id.lPCA(ver='fan').fit(data)
    x = local_id.lPCA(ver='ratio').fit(data)
    x = local_id.lPCA(ver='maxgap').fit(data)
        
def test_ess_params(data):
    x = local_id.ESS().fit(data)
    x = local_id.ESS(ver='b').fit(data)
    x = local_id.ESS(d=2).fit(data)

#test equality with R version#
#def test_ess(data):
#    res = intdimr.essLocalDimEst(data)
#    r_ess = dict(zip(res.names,[np.array(i) for i in res]))
#    assert (r_ess['dim.est'] == ess.dimension_ and r_ess['ess'] == ess.essval_)

def test_corint(data): 
    corint = global_id.CorrInt(k1=10,k2=20).fit(data)
    r_corint = np.array(ider.corint(data,k1=10,k2=20))
    assert_equal(r_corint, corint.dimension_)
        
def test_mada(data): 
    r_mada = np.array(ider.mada(data,local=True,k=20))
    mada = global_id.Mada(local=True,k=20).fit(data)
    assert_array_almost_equal(r_mada,mada.dimension_)
    
def test_mle():
    #test base params
    assert np.isclose(        maxLikGlobalDimEst(data, k), 
                      intdimr.maxLikGlobalDimEst(data, k)[0][0])

    #test dnoise+sigma
    assert np.isclose(        maxLikGlobalDimEst(data, k, dnoise = 'dnoiseGaussH', sigma = 0.2), 
                      intdimr.maxLikGlobalDimEst(data, k, dnoise = 'dnoiseGaussH', sigma = 0.2)[0][0])

    #test neighborhood_aggregation = 'mean'
    assert np.isclose(        maxLikGlobalDimEst(data, k, neighborhood_aggregation = 'mean'), 
                      intdimr.maxLikGlobalDimEst(data, k, neighborhood_aggregation = 'mean')[0][0])

    #test unbiased = True
    assert np.isclose(        maxLikGlobalDimEst(data, k, unbiased = True), 
                      intdimr.maxLikGlobalDimEst(data, k, unbiased = True)[0][0])

    #test neighborhood_based = 'False'
    assert np.isclose(        maxLikGlobalDimEst(data, k, neighborhood_based = False), 
                      intdimr.maxLikGlobalDimEst(data, k, neighborhood_based = False)[0][0])