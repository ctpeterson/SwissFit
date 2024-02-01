from scipy import optimize as _optimize # SciPy optimize
import numpy as _numpy # Number crunching
from .optimizer import Optimizer as _Optimizer # Optimizer parent class

""" Least squares wrapper """
# Filter for appropriate SciPy least squares arguments
_lsq_args = ['jac', 'method', 'ftol', 'xtol',
             'gtol', 'x_scale', 'loss', 'f_scale',
             'diff_step', 'tr_solver', 'tr_options',
             'jac_sparsity', 'max_nfev', 'verbose',
             'args', 'kwargs']

# Wrapper of SciPy's least squares
def scipy_least_squares(fcn, x0, **kwargs):
    try: # Exception ensures program does not unnecessarily crash if fit goes haywire
        # Do least squares fit
        fit = _optimize.least_squares(
            fcn, x0, **{key: kwargs[key] for key in kwargs.keys() if key in _lsq_args}
        );
        
        # Add cost to OptimizeResult object as 'fun'
        fit.fun = fit.cost

        # Return OptimizeResult
        return fit
    except ValueError:
        # Warn user that there was a value error
        print('Warning! ValueError in local optimization.')
        
        # Return null OptimizeResult
        return _optimize.OptimizeResult(
            x = x0,
            fun = _numpy.inf,
            nfev = 1, njev = 1,
            success = False,
            message = 'ValueError'
        )
        
""" Least squares class """
class SciPyLeastSquares(_Optimizer):
    def __init__(self,
                 fitter = None,
                 fcn = None,
                 jac = None,
                 tolerance_schedule = None,
                 optimizer_arguments = {}
                 ):
        # Initialize optimizer object
        super().__init__(
            fcn = fcn if fitter is None else fitter.calculate_residual,
            jac = jac if fitter is None else fitter.calculate_jacobian,
            optimizer_arguments = optimizer_arguments
        )

        # Take care of defaults if fitter specified
        if fitter is not None:
            fitter.local_optimizer = self.scipy_least_squares
            fitter.local_optimizer_tag = 'scipy_least_squares'
            
    # Run SciPy least squares optimization on call
    def __call__(self, p0):
        return scipy_least_squares(
            self._fcn, p0,
            jac = self._jac,
            **self._args
        )

    # Wrapper method (alternative to call - discards kwargs)
    def scipy_least_squares(self, fcn, x0, **kwargs):
        self._args['jac'] = self._jac
        for key in kwargs.keys():
            if key not in self._args.keys():
                self._args[key] = kwargs[key]
        return scipy_least_squares(fcn, x0, **self._args)
        
