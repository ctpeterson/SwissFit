from scipy import optimize as _optimize # SciPy optimize
from .optimizer import Optimizer as _Optimizer # Optimizer parent class

# SciPy minimize wrapper class
class SciPyMinimize(_Optimizer):
    def __init__(self, optimizer_arguments = {}):
        super().__init__(optimizer_arguments = optimizer_arguments)
        self.tag = 'scipy_minimize'

    # Run SciPy minimize optimization class
    def __call__(self, p0, fcn, grad):
        if 'jac' not in self._args.keys():
            self.fit = _optimize.minimize(fcn, p0, jac = grad, **self._args)
        else: self.fit = _optimize.minimize(fcn, p0, **self._args)
        return self.fit

    # Wrapper method (alternative to call)
    def scipy_minimize(self, fcn, x0, **kwargs):
        self._args['jac'] = self._jac
        for key in kwargs.keys():
            if key not in self._args.keys():
                self._args[key] = kwargs[key]
        self.fit = _optimize.minimize(fcn, x0, **self._args)
        return self.fit

    # Returns information about fit as string
    def __str__(self):
        out = 3 * ' ' + 'algorithm = SciPy minimize\n'
        for key, item in self.fit.items():
            unwanteds = [
                'x', 'lowest_optimization_result', 'jac',
                'cost', 'grad', 'active_mask', 'hess_inv'
            ]
            if all(unwanted not in key for unwanted in unwanteds):
                out += 3 * ' ' + key + ' = ' + str(item) + '\n'
        return out
