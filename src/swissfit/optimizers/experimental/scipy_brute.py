from scipy import optimize as _optimize # SciPy optimize
from .optimizer import Optimizer as _Optimizer # Optimizer parent class
from scipy.optimize import OptimizeResult as _OptimizeResult # For mocking SciPy OptimizeResult

""" Brute class """
class Brute(_Optimizer):
    def __init__(self,
                 fcn = None,
                 bounds = None,
                 optimizer_arguments = {}
                 ):
        super().__init__(
            fcn = fcn, optimizer_arguments = optimizer_arguments,
            bounds = bounds
        )

    def __call__(self, p0): return _OptimizeResult(
            x = _optimize.brute(
                func = self._fcn,
                ranges = self._bounds,
                **self._args
            )
    )
