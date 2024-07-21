""" SwissFit imports """
from swissfit import fit as swissfit # SwissFit fitter
from swissfit.monte_carlo import vegas as vegas_lepage # Peter Lepage's "Vegas"

""" Other imports """
import gvar as gvar # Peter Lepage's GVar library
import lsqfit as lsqfit # Peter Lepage's Lsqfit library
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
    """ Create data and priors """
    # Dictionary of "actual" fit parameters
    a, b, error = 2.0, 0.5, 0.1

    # Artificial dataset
    data = create_dataset(a, b, error)
    
    # Create priors
    prior = {'c': [gvar.gvar('1.5(1.5)'), gvar.gvar('0.75(0.75)')]}
    
    """ Do fit with SwissFit """
    # Create fit object
    fitter = swissfit.SwissFit(data = data, prior = prior, fit_fcn = sin)

    # Create estimator for MCMC with Vegas++
    estimator = vegas_lepage.VegasLepage()

    # Do fit by MCMC estimation
    fitter(estimator)

    # Print result of fit
    print(fitter)
    
