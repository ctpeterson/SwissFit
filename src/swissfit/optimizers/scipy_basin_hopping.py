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
    def __init__(self, optimizer_arguments = {}, local_estimator = None):
        super().__init__(optimizer_arguments = optimizer_arguments)
        self.tag = 'scipy_basin_hopping'
        if local_estimator is not None: self.set_local_optimizer(local_estimator)

    # Sets local optimizer according to specifications from SwissFit estimator object
    def set_local_optimizer(self, estimator):
        self.local_estimator = estimator
        if self.local_estimator.tag == 'scipy_least_squares':
            self._args['minimizer_kwargs'] = {'method': self.local_estimator.scipy_least_squares}

    # Run SciPy basin hopping on call
    def __call__(self, p0, fcn, jac):
        self.local_estimator.set_jac(jac)
        self.fit = _optimize.basinhopping(fcn, p0, **self._args)
        return self.fit

    # Wrapper method (alternative to call - discards kwargs)
    def basin_hopping(self, func, x0, **kwargs):
        for key in kwargs.keys():
            if key not in self._args.keys(): self._args[key] = kwargs[key]
        return _optimize.basinhopping(func, x0, **self._args)

    # Returns information about fit as string
    def __str__(self):
        out = 3 * ' ' + 'algorithm = SciPy basin hopping\n'
        for key, item in self.fit.items():
            if all(unwanted not in key for unwanted in ['x', 'lowest_optimization_result']):
                out += 3 * ' ' + key + ' = ' + str(item) + '\n'
        return out
