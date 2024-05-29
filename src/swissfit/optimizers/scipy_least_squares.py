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
    def __init__(self, optimizer_arguments = {}):
        super().__init__(optimizer_arguments = optimizer_arguments)
        self.tag = 'scipy_least_squares'
 
    # Run SciPy least squares optimization on call
    def __call__(self, p0, fcn, jac):
        self.fit = scipy_least_squares(fcn, p0, jac = jac, **self._args)
        return self.fit

    # Wrapper method (alternative to call)
    def scipy_least_squares(self, fcn, x0, **kwargs):
        self._args['jac'] = self._jac
        for key in kwargs.keys():
            if key not in self._args.keys():
                self._args[key] = kwargs[key]
        self.fit = scipy_least_squares(fcn, x0, **self._args)
        return self.fit

    # Returns information about fit as string
    def __str__(self):
        out = 3 * ' ' + 'algorithm = SciPy least squares\n'
        for key, item in self.fit.items():
            unwanteds = [
                'x', 'lowest_optimization_result', 'jac',
                'cost', 'grad', 'active_mask'
            ]
            if all(unwanted not in key for unwanted in unwanteds):
                out += 3 * ' ' + key + ' = ' + str(item) + '\n'
        return out
        
