from scipy.optimize import OptimizeResult as _OptimizeResult # For mocking SciPy OptimizeResult
import numpy as _numpy # Vectorized number crunching
from functools import partial as _partial # For partial evaluation of function
from ..other_tools import progress as _progress # For monitoring progress
from .optimizer import Optimizer as _Optimizer # Optimizer parent class

"""
SciPy-style particle swarm optimization

- My future plan is to migrate this from pure Python code to a Nim wrapper
"""

""" Implementation of particle swarm optimization """
# Base particle swarm optimization
def _particle_swarm(fcn,
                    bounds,
                    x0,
                    pool,
                    fkwargs = {},
                    w = 0.8,
                    c1 = 1.,
                    c2 = 1.,
                    population_size = 10,
                    maxiter = 1000,
                    initializer = None,
                    callback = None,
                    intervener = None,
                    verbose = False,
                    seed = None,
                    **kwargs
                    ):
    # Prepare random number generator
    rng = _numpy.random.default_rng(seed = seed)
    
    # Initialize population and temporary particle swarm variables
    x = _numpy.zeros((population_size, len(x0)))
    v = _numpy.zeros((population_size, len(x0)))
    f = _numpy.zeros(population_size)
    if initializer is None:
        if hasattr(bounds, 'keep_feasible'): bounds = list(zip(bounds.lb, bounds.ub))
        def _init_x(member): return [
                rng.uniform(bounds[x0i][0], bounds[x0i][-1])
                for x0i, x0v in enumerate(x0)
        ]
        def _init_v(member): return [
                rng.uniform(
                    -abs(bounds[x0i][-1] - bounds[x0i][0]),
                     abs(bounds[x0i][-1] - bounds[x0i][0])
                )
                for x0i, x0v in enumerate(x0)
        ]
        x = _numpy.array(list(map(_init_x, range(_numpy.shape(x)[0]))))
        v = _numpy.array(list(map(_init_v, range(_numpy.shape(x)[0]))))
    else: x = initializer(x0, x)
    best_x = x; best_f = _numpy.array(pool.map(_partial(fcn, **fkwargs), best_x));
    optimal_x = best_x[best_f.argmin()]; optimal_f = best_f[best_f.argmin()];
    
    # Particle swarm optimization loop
    for iteration in range(0, maxiter):
        # Callback & print
        if callback and callback(iteration, x, best_f, optimal_f): break
        if verbose: _progress.update_progress((iteration + 1) / maxiter)

        # Update coordinates & velocities
        r1, r2 = rng.uniform(0.,1.), rng.uniform(0.,1.)
        v = w*v + c1 * r1 * (best_x - x) + c2 * r2 * (optimal_x - x)
        x += v;

        # See if user wants to do something w/ coordinates after move
        if intervener: x, v = intervener(pool, rng, x, v, iteration)
        f = _numpy.array(pool.map(_partial(fcn, **fkwargs), x))
        
        # Save & update solutions
        best_x[f <= best_f] = x[f <= best_f]; best_f[f <= best_f] = f[f <= best_f];
        optimal_x = best_x[best_f.argmin()]; optimal_f = best_f[best_f.argmin()];

    # Return result of particle swarm
    return _OptimizeResult(
        x = optimal_x,
        fun = optimal_f,
        nit = (iteration + 1) * population_size,
        nfev = (iteration + 1) * population_size,
        success = True
    )
    
""" Particle swarm class """
class ParticleSwarm(_Optimizer):
    def __init__(self,
                 fcn = None,
                 bounds = None,
                 optimizer_arguments = {},
                 pool = None
                 ):
        super().__init__(
            fcn = fcn, optimizer_arguments = optimizer_arguments,
            pool = pool, bounds = bounds)
    def __call__(self, p0): return _particle_swarm(
            self._fcn, self._bounds,
            p0, self._pool, **self._args)
