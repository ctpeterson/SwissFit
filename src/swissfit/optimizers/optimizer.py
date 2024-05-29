from scipy.optimize import Bounds as _Bounds # SciPy bounds object
import gvar as _gvar # For manipulating Gaussian variables
import numpy as _numpy # Number crunching

# Mock "pool" class that makes abstracting pool in code easier
class MockSerialPool(object):
    def __init__(self): return None
    def map(self, fcn, x): return list(map(fcn, x))

# Parent class for optimizers
class Optimizer(object):
    def __init__(self,
                 optimizer_arguments = None,
                 bounds = None,
                 pool = None):
        self._args = optimizer_arguments
        self._pool = MockSerialPool() if pool is None else pool
        self._bounds = bounds
        self.method = 'MAP'
        self.local_optimizer = 'none'
        return None

    def set_jac(self, jac): self._jac = jac
    def set_fcn(self, fcn): self._fcn = fcn
    def set_hess(self, hess): self._hess = hess
    
    def create_bounds(self,
                      p = None,
                      standard_deviations = 1.,
                      result_type = 'seq',
                      extra_args = {},
                      lb = None, ub = None
                      ):
        """
        Create bounds from GVar variables. 

        - "result_type" can be a dictionary ('dict'), sequence ('seq') or an instance 
        of the SciPy Bounds class ('scipy_bounds').

        - "standard_deviations" the number of standard deviations that each 
        variable is allowed to sit between. "standard_devs "can be a single 
        number or a list. 

        - "extra_args" passed initialization of SciPy Bounds instance
        """
        if (lb is None) or (ub is None):
            # Set sequence of "integer standard deviations"
            if not hasattr(standard_deviations, '__len__'):
                sdev = standard_deviations; standard_deviations = [sdev for pv in p];
            standard_deviations = _numpy.array(standard_deviations)
            lb = _gvar.mean(p) - standard_deviations * _gvar.sdev(p)
            ub = _gvar.mean(p) + standard_deviations * _gvar.sdev(p)
        match result_type:
            case 'seq': self._bounds = list(zip(lb, ub))
            case 'scipy_bounds': self._bounds = _Bounds(
                lb = lb, ub = ub, **extra_args
            )
            case _: print('Invalid option for result_type.',
                          'Must be:', "'seq',", 'or', "'scipy_bounds'")

    def __str__(self):
        return 3 * ' ' + 'SwissFit optimizer object'
