from scipy import optimize as _optimize # SciPy optimize
from .optimizer import Optimizer as _Optimizer # Optimizer parent class

class LocalOptimizer(_Optimizer):
    def __init__(self,
                 fcn = None,
                 jac = None,
                 hess = None,
                 optimizer_arguments = {},
                 pool = None
                 ):
        super().__init__(fcn = fcn,
                         jac = jac,
                         hess = hess,
                         optimizer_arguments = optimizer_arguments,
                         pool = pool)
    def __call__(self, p0):
        if self._hess is not None: return _optimize.minimize(
                self._fcn,
                p0,
                jac = self._jac,
                hess = self._hess,
                **self._args
        )
        else: return _optimize.minimize(
                self._fcn,
                p0,
                jac = self._jac,
                **self._args
        )
