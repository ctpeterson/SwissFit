""" SwissFit imports """
from swissfit import fit # SwissFit fitter
from swissfit.optimizers import scipy_least_squares # SciPy's trust region reflective

""" Other imports """
import gvar as gvar # Peter Lepage's GVar library
import numpy as np # NumPy

# Sine fit function
def sin(x, p):
    return p['c'][0] * gvar.sin(p['c'][-1] * x)

def create_dataset(a, b, error):
    # Actual parameters of the sine function
    real_fit_parameters = {'c': [a, b]}

    # Real dataset
    np.random.seed(0) # Seed random number generator
    data = {} # Dictionary to hold data

    # Input data
    data['x'] = np.linspace(0., 2. * np.pi / b, 20)

    # Output data
    data['y'] = [
        gvar.gvar(
            np.random.normal(sin(xx, real_fit_parameters), error), # Random mean
            error # Error on mean
        )
        for xx in data['x']
    ]

    # Return dataset
    return data

if __name__ == '__main__':
    """ Data and priors """
    # Dictionary of "actual" fit parameters
    a, b, error = 2.0, 0.5, 0.1

    # Artificial dataset
    data = create_dataset(a, b, error)
    
    # Create priors
    prior = {'c': [gvar.gvar('1.5(1.5)'), gvar.gvar('0.75(0.75)')]}
    
    """ Fit objects """
    # Create fit object
    fitter = fit.SwissFit(
        data = data,
        prior = prior,
        fit_fcn = sin,
    )
    
    # Create optimizer object
    optimizer = scipy_least_squares.SciPyLeastSquares(fitter = fitter)
    
    """ Do fit """
    # Do fit by passing optimizer through fitter
    fitter(optimizer)
    
    # Print out result of fit
    print(fitter)
    
    # Save fit parameters
    fit_parameters = fitter.p
    
    # Calculate f(0.5, f(1.0)
    fa = sin(0.5, fit_parameters)
    fb = sin(1.0, fit_parameters)

    # Print f(0.5) & f(1.0)
    print('f(0.5) f(1.0):', fa, fb)
    
    # Print covariance matrix of (fa, fb)
    print('covariance of f(0.5) & f(1.0):\n', gvar.evalcov([fa, fb]))
