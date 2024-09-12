import skdim.datasets
import pytest
import inspect

datasets_functions = [o for o in inspect.getmembers(skdim.datasets) 
                      if inspect.isfunction(o[1]) and o[0] not in ['product', 'solve_ivp', 'check_random_state','fetch_openml', 'check_random_state']]
datasets_classes = [o for o in inspect.getmembers(skdim.datasets) if inspect.isclass(o[1])]
print(datasets_functions)

@pytest.mark.parametrize("dsfunc", datasets_functions)

def test_dataset_default_func(dsfunc):
    f= dsfunc[1]
    fsig = inspect.signature(f)
    input_params = dict()

    for k,v in fsig.parameters.items():
        if k not in ['args', 'kwargs']:
            if v.default == inspect._empty: #only add if parameter is not specified as a default parameter
                if k in ['n', 'n_swiss', 'n_sphere']:
                    input_params[k] = 100
                elif k == 'd':
                    input_params[k] = 5
    X = f(**input_params)

@pytest.mark.parametrize("dscl", datasets_classes)

   
def test_dataset_default_class(dscl):
    cl = dscl[1]
    obj = cl()
    fsig = inspect.signature(cl.generate)
    input_params = dict()

    for k,v in fsig.parameters.items():
        if k not in ['args', 'kwargs']:
            if v.default == inspect._empty: #only add if parameter is not specified as a default parameter
                if k == 'n':
                    input_params[k] = 100
                elif k == 'd':
                    input_params[k] = 5

    X = obj.generate(**input_params)