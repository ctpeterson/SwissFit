from scipy import optimize as _optimize # SciPy optimize
from .optimizer import Optimizer as _Optimizer # Optimizer parent class

""" SciPy dual simulated annealing class """
class SciPyDualSimulatedAnnealing(_Optimizer):
    def __init__(self,
                 fcn = None,
                 bounds = None,
                 optimizer_arguments = {}
                 ):
        super().__init__(fcn = fcn, optimizer_arguments = optimizer_arguments, bounds = bounds)
    def __call__(self, p0):
        return _optimize.dual_annealing(
            self._fcn, self._bounds,
            x0 = p0, **self._args
        )
