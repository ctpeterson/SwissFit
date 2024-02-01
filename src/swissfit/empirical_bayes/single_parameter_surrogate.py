import numpy as _numpy # Usual number crunching
from scipy import optimize as _optimize # SciPy optimization
import gvar as _gvar # For cubic spline
from functools import partial as _partial # For partial evaluation of function
import sys as _sys # For exiting if bounds are not right
from scipy.optimize import OptimizeResult as _OptimizeResult # For mocking up SciPy OptimizeResult
from .empirical_bayes import EmpiricalBayes as _EmpiricalBayes # Empirical Bayes parent class

""" Convenience functions """

# Visualize result for 'fcn' & its interpolation over set of points
def _visualize_sample_points(fcn,
                             spline,
                             lb, ub,
                             n_points = 10,
                             pool = None,
                             p0 = None,
                             show = True,
                             save = False,
                             fig_name = ''
                             ):
    # Import plotting tools
    import matplotlib.pyplot as _plt
    
    # Get set of points
    fcn_in = _numpy.linspace(lb, ub, n_points) if p0 is None else p0
    fcn_out = pool.map(fcn, fcn_in)

    # Interpolate over set of points
    interpolation = spline(fcn_in, fcn_out)

    # Plot data points used in interpolation
    _plt.scatter(fcn_in, fcn_out, label = 'fcn', color = 'k')

    # Get result from interpolation & plot
    xvs = _numpy.linspace(lb, ub, n_points * n_points)
    yvs = [interpolation(xx) for xx in xvs]
    _plt.plot(xvs, yvs, label = 'interpolation', color = 'maroon')

    # Finish figure
    _plt.xlabel('x'); _plt.ylabel('fcn(x)');
    _plt.grid('on'); _plt.legend();
    
    # Show & save figure
    if show: _plt.show()
    if save: _plt.savefig(fig_name)
    
    # Close figure
    _plt.close()
    
""" Single-parameters surrogate optimization """

# Main function for optimization by bisection
def _surrogate_bisection_optimization(fcn,
                         spline,
                         lb, ub,
                         n_points = 10,
                         pool = None,
                         p0 = None,
                         tol = 1e-8,
                         callback = None,
                         max_iter = 1000,
                         generate_new_points = None
                         ):
    # Create first set of points
    fcn_in = _numpy.linspace(lb, ub, n_points) if p0 is None else p0
    fcn_out = pool.map(fcn, fcn_in)

    # Interpolate over initial set of points
    interpolation = spline(fcn_in, fcn_out)

    # Do optimization
    iteration, converged, new_mid = 0, False, 0.5 * lb + 0.5 * ub
    while True:
        # Get dfa, dfb & dfc
        df_lb, df_ub = [*map(interpolation.D, [lb, ub])]
        mid = new_mid; df_mid = interpolation.D(mid);
        
        # Bisection test
        if df_ub * df_mid < 0.: lb = mid
        elif df_lb * df_mid < 0.: ub = mid
        elif df_lb * df_ub > 0: _sys.exit('df_lb/dx & df_ub/dx must have different signs.')

        # Test if converged & print information out
        iteration += 1; new_mid = 0.5 * lb + 0.5 * ub; f_mid = interpolation(mid);
        if callback and callback(iteration, lb, ub, mid, f_mid):
            converged = True; break;
        if _numpy.abs(mid - new_mid) <= tol:
            converged = True; break;
        elif iteration > max_iter: break
        
        # Generate new set of points
        if generate_new_points is None: fcn_in = _numpy.linspace(lb, ub, n_points)
        else: fcn_in = generate_new_points(
                pool, interpolation, lb, new_mid, ub, mid, f_mid)
        fcn_out = pool.map(fcn, fcn_in)

        # Interpolate over new points
        interpolation = spline(fcn_in, fcn_out)

    # Return result of fit
    return _OptimizeResult(
        x = mid,
        fun = f_mid,
        jac = interpolation.D(mid),
        nit = iteration,
        nfev = n_points * iteration,
        success = converged
    )

# Main function for optimization by optimization of surrogate
def _surrogate_curve_optimization(fcn,
                                  spline,
                                  lb, ub,
                                  n_points = 10,
                                  pool = None,
                                  x0 = None,
                                  p0 = None,
                                  scipy_optimize_arguments = {}
                                  ):
    # Create first set of points for interpolation
    fcn_in = _numpy.linspace(lb, ub, n_points) if p0 is None else p0
    fcn_out = pool.map(fcn, fcn_in)

    # Interpolate over set of points
    interpolation = spline(fcn_in, fcn_out)

    # Run optimization over interpolation & return
    return _optimize.minimize(
        interpolation,
        0.5 * (lb + ub) if x0 is None else x0,
        jac = interpolation.D,
        bounds = [[lb, ub]],
        **scipy_optimize_arguments
    )
    
# Single-parameter surrogate-based optimization
class SingleParameterSurrogate(_EmpiricalBayes):
    def __init__(self,
                 fcn = None,
                 lb = None,
                 ub = None,
                 pool = None,
                 n_points = None,
                 spline_algorithm = 'steffen',
                 arguments = {},
                 optimization_method = 'direct' # 'direct' or 'bisection'
                 ):
        super().__init__(fcn = fcn, arguments = arguments, pool = pool)
        self._lb, self._ub = lb, ub
        self._spline = _partial(_gvar.cspline.CSpline, alg = spline_algorithm)
        self._n_points = n_points
        self._optimization_method = optimization_method
        
    # Runs optimization
    def __call__(self, p0 = None):
        if self._optimization_method == 'bisection':
            return _surrogate_bisection_optimization(
                self._fcn,
                self._spline,
                self._lb, self._ub,
                pool = self._pool,
                p0 = p0, n_points = self._n_points,
                **self._args       
            )
        elif self._optimization_method == 'direct':
            return _surrogate_curve_optimization(
                self._fcn,
                self._spline,
                self._lb, self._ub,
                pool = self._pool,
                p0 = p0, n_points = self._n_points,
                **self._args       
            )
            
    # For visualizing inital set of data points
    def visualize_sample_points(self,
                                p0 = None,
                                show = True,
                                save = False,
                                fig_name = ''):
        _visualize_sample_points(
            self._fcn,
            self._spline,
            self._lb, self._ub,
            pool = self._pool,
            p0 = p0,
            n_points = self._n_points,
            show = show, save = save,
            fig_name = fig_name
        )
