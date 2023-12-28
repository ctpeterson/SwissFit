from scipy import optimize as _optimize # SciPy optimize
import numpy as _numpy # Usual number crunching
from .optimizer import Optimizer as _Optimizer # Optimizer parent class

""" Custom basin hopping functions  """
# Take step routine enforcing positiviy
def take_step_biased(x, indices = [],
                     array_size = None,
                     maximum_tries = 1000,
                     stepsize_schedule = None,
                     restart_probability = 0.,
                     restart_function = None,
                     args = ()
                     ):
    # Step size schedule
    if stepsize_schedule is not None: take_step_biased.stepsize = stepsize_schedule()

    # Random restart probability
    if (restart_probability != 0.) and (restart_function is not None):
        if _numpy.random.uniform(0., 1.) < restart_probability:
            return restart_function(*args)
    
    # Make hops until hop preserves positivity constraint
    try_iteration = 0
    while True:
        # Draw random vector
        dx = _numpy.array([
            _numpy.random.uniform(-1., 1.) for dxv in range(array_size)
        ]); dx *= take_step_biased.stepsize / _numpy.linalg.norm(dx); try_iteration += 1;

        # Positivity constraint check
        if all(x[ind] + dx[ind] for ind in indices): break
        elif try_iteration > maximum_tries: break

    # Return perturbed coordinates
    return x + dx
take_step_biased.stepsize = 0.5 # Default step size

# Stochastic tunneling accept/reject
def accept_test_STUN(f_new, x_new, f_old, x_old, rng = None):
    if fit.calculate_chi2(x_new) < accept_test_STUN.f_low: accept_test_STUN.f_low = f_new
    fnew, fold = [*map(
        lambda x: 1. - _numpy.exp(-accept_test_STUN.gamma * (x - accept_test_STUN.f_low)),
        [f_new, f_old]
    )]; acc = _numpy.exp(-max(0., fnew - fold) / accept_test_STUN.T);
    return "force accept" if rng.uniform(0., 1.) < acc else False
accept_test_STUN.T = 1.
accept_test_STUN.f_low = _numpy.inf
accept_test_STUN.gamma = 1.

""" Basin hopping class """
# Scipy basin hopping class
class BasinHopping(_Optimizer):
    """
    Notes:
      - I *highly* recommend turning the tolerance for the convergence criterion of the
        local optimization algorithm down when using basin hopping. In many cases,
        having a high tolerance is absolutely unnecessary at best and computationally 
        prohibitive at worst.
    """
    def __init__(self,
                 fcn = None,
                 optimizer_arguments = {}
                 ):
        super().__init__(fcn = fcn, optimizer_arguments = optimizer_arguments)
    def __call__(self, p0):
        self.fit = _optimize.basinhopping(self._fcn, p0, **self._args)
        self._prepare_out_string(); return self.fit;

    def _prepare_out_string(self):
        self._out = 3 * ' ' + 'algorithm = SciPy basin hopping\n'
        for key, item in self.fit.items():
            if all(unwanted not in key for unwanted in ['x', 'lowest_optimization_result']):
                self._out += 3 * ' ' + key + ' = ' + str(item) + '\n'
    
    def __str__(self): return self._out
        
