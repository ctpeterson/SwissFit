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
    optimizer = scipy_least_squares.SciPyLeastSquares()
    
    """ Do fit """
    # Do fit by passing optimizer through fitter
    fitter(optimizer)
    
    # Print out result of fit
    print(fitter)

    # Get fit parameters
    fit_parameters = fitter.p

    """ Play around with fit parameters """
    # Calculate f(0.5, f(1.0)
    fa = sin(0.5, fit_parameters)
    fb = sin(1.0, fit_parameters)

    # Print f(0.5) & f(1.0)
    print('f(0.5) f(1.0):', fa, fb)
    
    # Print covariance matrix of (fa, fb)
    print('covariance of f(0.5) & f(1.0):\n', gvar.evalcov([fa, fb]))

    # Show that correlation w/ data is also calculated & produced as an output
    print(
        "Is p['c'][0] correlated with y[0]?",
        not gvar.uncorrelated(fit_parameters['c'][0], data['y'][0])
    )
    print('And here\'s the covariance:\n', gvar.evalcov([fit_parameters['c'][0], data['y'][0]]))

    """ Plot data """
    # Import Matplotlib
    import matplotlib.pyplot as plt

    # Plot fit data
    plt.errorbar(
        data['x'], 
        gvar.mean(data['y']), 
        gvar.sdev(data['y']), 
        color = 'k', markerfacecolor = 'none',
        markeredgecolor = 'k',
        capsize = 6., fmt = 'o'
    )

    # Get result of fit function
    x = np.linspace(data['x'][0], data['x'][-1], 100)
    y = sin(x, fit_parameters)
    
    # Plot error of fit function from fit as a colored band
    plt.fill_between(
        x,
        gvar.mean(y) - gvar.sdev(y),
        gvar.mean(y) + gvar.sdev(y),
        color = 'maroon', alpha = 0.5
    )
    
    # x/y label
    plt.xlabel('x', fontsize = 20.)
    plt.ylabel('$a\\sin(bx)$', fontsize = 20.)

    # Show fit parameters
    plt.text(
        7.25, 0.75,
        '$a=' + str(fit_parameters['c'][0]) + '$, \n $b=' + str(fit_parameters['c'][-1]) + '$',
        fontsize = 15.
    )
    
    # Grid
    plt.grid('on')

    # Show figure
    plt.show()
