""" Imports """
# External imports
import numpy as np
import gvar as gv
import lsqfit as lsqfit
from functools import partial

# Local imports
import example_tools as et

# SwissFit imports
import swissfit as swissfit
from swissfit.machine_learning import radial_basis
from swissfit.optimizers import scipy_least_squares

""" Fit functions """
def log(x, cutoff = 1e6):
    if x > 0: return gv.log(x)
    else: return gv.gvar(cutoff, '0') if isinstance(x, gv.GVar) else cutoff

def fit_fcn(b, l, p):
    params = p['I'] if 'I' in p.keys() else p
    return np.ravel(
        neural_net.out(
            (b * params['c'][0] - 1.) * l**params['c'][1],
            params
        )
    )

""" Fits """
def swissfit_fit():
    # Create fitter
    fit = swissfit.fit.SwissFit(
        udata = data,
        uprior = {'I': {
            'c': [gv.gvar('2.0(2.0)'), gv.gvar('1.0(1.0)')],
            'lyr1.center': [gv.gvar(0., 1e12), gv.gvar(0., 1e12)],
            'lyr1.bandwidth': [gv.gvar(0., 1e12), gv.gvar(0., 1e12)],
            'lyr2.weight': [gv.gvar(0., 1.785), gv.gvar(0., 1.785)],
            'lyr2.bias': [gv.gvar(0., 1e12)]
        }},
        p0 = {'I': p0},
        fit_fcn = fit_fcn
    )

    # Create estimator
    trust_region_reflective = scipy_least_squares.SciPyLeastSquares(
        fcn = fit.calculate_residual, jac = fit.calculate_jacobian,
    )

    # Do fit
    fit(trust_region_reflective)

    # Print fit
    print(fit)

    # Return fit
    return fit

def lsqfit_fit():
    # Get input data
    b, l = np.transpose(data['x'])
    
    # Do fit
    fit = lsqfit.nonlinear_fit(
        data = data['y'],
        fcn = lambda p: fit_fcn(b, l, p),
        p0 = p0,
        prior = {
            'c': [gv.gvar('2.0(2.0)'), gv.gvar('1.0(1.0)')],
            'lyr1.center': [gv.gvar(0., 1e12), gv.gvar(0., 1e12)],
            'lyr1.bandwidth': [gv.gvar(0., 1e12), gv.gvar(0., 1e12)],
            'lyr2.weight': [gv.gvar(0., 1.785), gv.gvar(0., 1.785)],
            'lyr2.bias': [gv.gvar(0., 1e12)]
        }
    )

    # Print result of fit
    print(fit)

    # Return fit
    return fit

def calculate_logml(fit):
    data_cov = gv.evalcov(fit.data['y'])
    prm_cov = gv.evalcov([prv for key in fit.p['I'].keys() for prv in fit.p['I'][key]])
    
    prior_cov = gv.evalcov([
        gv.gvar('2.0(2.0)'), gv.gvar('1.0(1.0)'),
        gv.gvar(0., 1e12), gv.gvar(0., 1e12),
        gv.gvar(0., 1e12), gv.gvar(0., 1e12),
        gv.gvar(0., 1.785), gv.gvar(0., 1.785),
        gv.gvar(0., 1e12)
    ])

    logdet_data_cov = np.linalg.slogdet(data_cov)[-1]
    logdet_prior_cov = np.linalg.slogdet(prior_cov)[-1]
    logdet_prm_cov = np.linalg.slogdet(prm_cov)[-1]

    logml = fit.chi2 - logdet_prm_cov
    logml += logdet_data_cov + logdet_prior_cov
    logml += fit.dof * gv.log(2. * np.pi)
    
    return -0.5 * logml

""" Run benchmark """
if __name__ == '__main__':
    # Create neural network
    neural_net = radial_basis.RadialBasisNeuralNetwork(
        {
            'lyr1': { # First layer
                'in': 1, 'out': 2, # Dimension of input & output
                'activation': 'exp', # Activation function
            },
            'lyr2': { # Output layer
                'in': 2, 'out': 1,
                'activation': 'linear',
            }
        }
    )
    
    # Get data
    data = et.ising_data()

    # Get starting values
    p0 = gv.mean(
        gv.load('./example_data/ising_u_exp_1p785_ridge_regression.bin')['I']
    )

    # Run swissfit
    swissfit_result = swissfit_fit()
    
    # Run lsqfit
    lsqfit_result = lsqfit_fit()

    # Print comparison of fit parameters
    for key, item in swissfit_result.p['I'].items():
        for item_ind, item_val in enumerate(item):
            print(key, item_ind, item_val, lsqfit_result.p[key][item_ind])
    print('chi2 comparison:', swissfit_result.chi2, lsqfit_result.chi2)
    print(
        'logGBF comparison:',
        swissfit_result.logml,
        lsqfit_result.logGBF,
        calculate_logml(swissfit_result)
    )
