import pytest
import numpy as np
from skdim import local_id, global_id
from sklearn.datasets import load_iris
from inspect import getmembers, isfunction, isclass
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
    return load_iris(return_X_y=True)

#test all estimators pass check_estimator
local_class_list = [o[1] for o in getmembers(local_id) if isclass(o[1])]
global_class_list = [o[1] for o in getmembers(global_id) if isclass(o[1])]
estimators = local_class_list+global_class_list
    
@pytest.mark.parametrize("Estimator", estimators)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

#test all kwargs work with non-default parameters
def test_fisher_params(data):
    try:
        x = local_id.FisherS().fit(data)
        x = local_id.FisherS(ConditionalNumber=2).fit(data)
        x = local_id.FisherS(ProducePlots=True).fit(data)
        x = local_id.FisherS(ProjectOnSphere=False).fit(data)
        x = local_id.FisherS(ncomp=True).fit(data)
        x = local_id.FisherS(limit_maxdim=True).fit(data)
    except:
        raise AssertionError('test_fisher_params failed')
        
def test_ess_params(data):
    try:
        x = local_id.ESS().fit(data)
        x = local_id.ESS(ver='b').fit(data)
        x = local_id.ESS(d=2).fit(data)
    except:
        raise AssertionError('test_ess_params failed')

#test equality with R version#
def test_ess(data):
    res = intdimr.essLocalDimEst(data)
    r_ess = dict(zip(res.names,[np.array(i) for i in res]))
    assert (r_ess['dim.est'] == ess.dimension_ and r_ess['ess'] == ess.essval_)

def test_corint(data): 
    corint = global_id.CorrInt(k1=10,k2=20).fit(data)
    r_corint = np.array(ider.corint(data,k1=10,k2=20))
    assert_equal(r_corint, corint.dimension_)
        
def test_mada(data): 
    r_mada = np.array(ider.mada(data,local=True,k=20))
    mada = global_id.Mada(local=True,k=20).fit(data)
    assert_array_almost_equal(r_mada,mada.dimension_)   