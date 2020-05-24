import numpy as np
from inspect import isclass, signature, Parameter
import traceback
import random
import sklearn

def check_fit_exists(estimator):
    func_name = 'fit'
    # This will raise an exception if doesn't exist
    func_callable = callable(getattr(estimator, f'{func_name}'))
    
    if not func_callable:
        raise Exception(f"{func_name} not callable")
        
    return True

def check_fit_arg_names(estimator):
    
    param_dict = signature(estimator.fit).parameters
    
    if 'X' not in param_dict and 'y' not in param_dict:
        raise Exception("fit does not have parameters 'X' and 'y'")
    
    if list(param_dict.keys())[:2] != ["X", "y"]:
        raise Exception("fit parameter order incorrect, first two params must be X, y")
    
    return True

def check_fit_returns_self(estimator):
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    val = estimator.fit(X, y)
    
    if val != estimator:
        raise Exception(f"fit does not return self")
        
    return True

def check_predict_exists(estimator):
    func_name = 'predict'
    # This will raise an exception if doesn't exist
    func_callable = callable(getattr(estimator, f'{func_name}'))
    
    if not func_callable:
        raise Exception(f"{func_name} not callable")
        
    return True

def check_predict_arg_names(estimator):
    
    param_dict = signature(estimator.predict).parameters
    
    if 'X' not in param_dict:
        raise Exception("predict does not have parameter 'X'")
    
    if len(param_dict) > 1:
        raise Exception("predict must only have one parameter 'X'")
    
    return True

def check_predict_return(estimator):
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    val = estimator.fit(X, y)
    
    preds = estimator.predict(X)
    
    if type(preds) != np.ndarray:
        raise Exception("predict did not return a numpy.ndarray")
        
    return True


def check_score_exists(estimator):
    func_name = 'score'
    # This will raise an exception if doesn't exist
    func_callable = callable(getattr(estimator, f'{func_name}'))
    
    if not func_callable:
        raise Exception(f"{func_name} not callable")
        
    return True

def check_score_arg_names(estimator):
    
    param_dict = signature(estimator.score).parameters
    
    if 'X' not in param_dict and 'y' not in param_dict:
        raise Exception("score does not have parameters 'X' and 'y'")
    
    if list(param_dict.keys())[:2] != ["X", "y"]:
        raise Exception("score parameter order incorrect, first two params must be X, y")

    return True

def check_score_return(estimator):
    
    X = np.random.randn(100, 10)
    y = np.random.randint(0, 2, 100)
    val = estimator.fit(X, y)
    
    score = estimator.score(X, y)

    if type(score) not in [float, np.float32, np.float64]:
        raise Exception("score did not return a float")
        
    return True

def check_init_exists(estimator):
    
    try:
        parent_class = estimator.__init__.__objclass__
        raise Exception("__init__ not defined")
    except AttributeError as e:
        pass
    
    return True

def check_no_attributes_set_in_init(estimator):
    
    from sklearn.utils.estimator_checks import check_no_attributes_set_in_init
    
    check_no_attributes_set_in_init("test", estimator)
    
    return True


def check_get_params_exists(estimator):
    
    func_name = 'get_params'
    # This will raise an exception if doesn't exist
    func_callable = callable(getattr(estimator, f'{func_name}'))
    
    if not func_callable:
        raise Exception(f"{func_name} not callable")
        
    return True

def check_get_params_return_type(estimator):
    
    params = estimator.get_params()
    if type(params) is not dict:
        raise Exception("get_params did not return a dict")
        
    return True

def check_get_params_return_value(estimator):
    
    param_dict = signature(estimator.__init__).parameters
    init_keys_set = set(param_dict.keys())
    
    get_params_keys_set = set(estimator.get_params().keys())
    
    if init_keys_set != get_params_keys_set:
        raise Exception("get_params returned a dictionary which did not match __init__ parameters")

    return True

def check_set_params_exists(estimator):
    
    func_name = 'set_params'
    # This will raise an exception if doesn't exist
    func_callable = callable(getattr(estimator, f'{func_name}'))
    
    if not func_callable:
        raise Exception(f"{func_name} not callable")
        
    return True

def check_set_params_arg_type(estimator):
    
    set_params_arg_dict = signature(estimator.set_params).parameters
    
    if 'params' not in set_params_arg_dict:
        raise Exception(f"set_params does not have a parameter called 'params'")
        
    if len(set_params_arg_dict) > 1:
        raise Exception(f"set_params must only have 1 parameter called 'params'")
        
    if set_params_arg_dict['params'].kind != Parameter.VAR_KEYWORD:
        raise Exception(f"set_params parameter called 'params' must be a **kwargs")
        
    return True

def check_set_params_returns_self(estimator):
    
    val = estimator.set_params()
    
    if val != estimator:
        raise Exception(f"set_params does not return self")
    
    return True


def check_base_estimator(estimator):
    
    if not issubclass(type(estimator), sklearn.base.BaseEstimator):
        raise Exception("estimator is not subclass of sklearn.base.BaseEstimator")
    
    return True

def check_classifier_mixin(estimator):
    
    if not issubclass(type(estimator), sklearn.base.ClassifierMixin):
        raise Exception("estimator is not subclass of sklearn.base.ClassifierMixin")
    
    return True

def check_fit_only_2d(estimator):
    
    X = np.random.randn(100)
    y = np.random.randint(0, 2, 100)
    
    try:
        val = estimator.fit(X, y)
    except:
        pass
    else:
        raise Exception("estimator's fit function accepted 1d X, must accept 2d only")
    
    return True

def check_fit_equal_n(estimator):
    X = np.random.randn(100, 1)
    y = np.random.randint(0, 2, 10)
    
    try:
        val = estimator.fit(X, y)
    except:
        pass
    else:
        raise Exception("estimator's fit function accepted X and y of different lengths")
    
    return True

def check_fit_y_no_nan(estimator):
    X = np.random.randn(100)
    y = np.full(100, np.inf)
    
    try:
        val = estimator.fit(X, y)
    except:
        pass
    else:
        raise Exception("estimator's fit function accepted y containining NaN")
    
    return True

def check_estimator_unfitted(estimator):
    # attributes that get set during fit() must end with an underscore
    
    X = np.random.randn(100)
    
    try:
        val = estimator.predict(X)
    except sklearn.exceptions.NotFittedError as e:
        pass
    else:
        raise Exception("predict did not raise NotFittedError when unfitted")
    
    return True
    

def check_estimator_adaboost(estimator_cls):
    """
    Runs a list of checks for rough compatibility with sklearn.
    
    Parameters
    ----------
    estimator_cls : class
        The class to check
    
    """
    
    check_list = [
        # BASIC
        check_fit_exists, # fit function defined
        check_fit_arg_names, #the first args are X and y
        check_fit_returns_self, # fit function returns self
        check_predict_exists, # predict function defined
        check_predict_arg_names, # the only arg is X
        check_predict_return, # predict returns an ndarray
        check_score_exists, # score function defined
        check_score_arg_names, # the first args of score are X and y
        check_score_return, # score fucntion returns a float
        
        # ADVANCED
        check_init_exists, # init function defined
        check_no_attributes_set_in_init, # init only sets values that are arguments
        check_get_params_exists, # get_params function defined
        check_get_params_return_type, # get_params returns a dict
        check_get_params_return_value, # keys in dictionary returned by get_params matches __init__ args
        check_set_params_exists, # set_params function defined
        check_set_params_arg_type, # set_params has single arg called params, which is a **kwargs
        check_set_params_returns_self, # set_params returns self
        check_base_estimator, # inherits from sklearn.base.BaseEstimator
        check_classifier_mixin, # inherits from sklearn.base.ClassifierMixin
        check_fit_only_2d, # fit only accepts 2D X
        check_fit_equal_n, # fit only accepts X and y of equal length
        check_fit_y_no_nan, # fit only accepts y with no NaN
        check_estimator_unfitted, # predict only works after calling fit
    ]
    
    for check in check_list:
        
        check_name = check.__name__
        
        try:
            check(estimator_cls())
            print(f"Passed: {check_name}")
        except Exception as e:
            print(f"Failed: {check_name}")
            print("".join(traceback.TracebackException.from_exception(e).format()))


