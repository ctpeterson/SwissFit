import gvar as gv
import numpy as np
from swissfit import fit

def compare_hessians(p0):
    # Set fit object as global parameter
    global fit_object

    # Direct calculation of Hessian
    hessian1 = fit_object.calculate_hessian(p0)

    # Indirect calculation of "exact" Hessian
    fit_object(None, p0 = p0, estimation_method = 'none', approximate_parameter_covariance = False)
    hessian2 = fit_object.hessian

    # Indirect calculation of approximate Hessian
    fit_object(None, p0 = p0, estimation_method = 'none', approximate_parameter_covariance = True)
    hessian3 = fit_object.hessian

    # Print result
    print(25 * '-.')
    print('x =', p0, '\n')
    print('direct: \n', hessian1, '\n')
    print('"exact" indirect:\n', hessian2, '\n')
    print('approximate indirect:\n', hessian3, '\n')
    print(
        'exact:\n',
        np.array(
            [
                [12. * p0[0]**2. + 2. * p0[-1]**2., 4. * p0[0] * p0[-1]],
                [4. * p0[0] * p0[-1],               2. * (1. + p0[0]**2.)]
            ]
        )
    )
    print(25 * '-.' + '\n')
    
if __name__ == '__main__':
    """
    Test of Hessian calculations for f(x, y) = x^4 + y^2 + x^2*y^2
    """
    
    # Create fit object
    fit_object = fit.SwissFit(
        data = {'y': [
            gv.gvar('0(1)'),
            gv.gvar('0(1)'),
            gv.gvar('0(1)')
        ]},
        p0 = {'I': {'p': [0., 0.]}},
        fit_fcn = lambda p: np.array([
            p['I']['p'][0]**2., # x^4
            p['I']['p'][-1], # y^2
            p['I']['p'][0] * p['I']['p'][-1], # x^2*y^2
        ])
    )

    # Print spacing
    print('\n')
    
    # Check at (0, 0)
    compare_hessians([0., 0.])

    # Check at (2, 2)
    compare_hessians([2., 2.])

    # Check at (100, 100)
    compare_hessians([100., 100.])

    # Check at (1e6, 1e6)
    compare_hessians([1e6, 1e6])

    # Print spacing
    print('')
