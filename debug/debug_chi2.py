""" External packages """
import numpy as np
import gvar as gv
import matplotlib.pyplot as plt

""" Imports from this package & local path """
import example_tools as et
import swissfit as swissfit
from swissfit.machine_learning import pure_python_neural_network as pnn
from swissfit.numerical_tools import linalg

""" Model function """
def fit_fcn(b, l, p):
    """ fit_function(b, l) = NN( (b/bc - 1) * L^{1/nu} ) """
    global nn
    if not hasattr(b, '__len__'):return np.ravel(
            nn.out([(b * gv.abs(p['I']['c'][0]) - 1.) * l**gv.abs(p['I']['c'][-1])], p['I'])
    )
    else: return np.ravel(
            nn.out((b * gv.abs(p['I']['c'][0]) - 1.) * l**gv.abs(p['I']['c'][-1]), p['I'])
    )

""" Function that calculates the chi^2 of the prior """
def log(x, cutoff = 1e6):
    if x > 0: return gv.log(x)
    else: return cutoff

# Define prior function
def prior_fcn(p):
    """
    This function calculates the log prior for each level of priors. 
    This will typically be
    
    logPrior = sum_{level}sum_{p_level} * [cov_level^{-1/2} * (p_level - <p_level>)]^2,

    where "level" is the level of the prior if a hierarchical model is being used. 
    The covariance matrix in our case is diagonal, though it does not have to be diagonal.
    To force the positivity of some parameters without potential issues with numerical
    instability, some priors are modified as

    cov_level^{-1/2} * (p_level - <p_level>) 
    ---> step(p_level) * cov_level^{-1/2} * (p_level - <p_level>),

    where step(p_level) is a smooth step function that is large when p_level < 0. The
    resulting prior is essentially a truncated Gaussian. It is therefore a totally
    valid choice for the prior.
    """
    
    """ Setup """
    # Have function recognize prior as a global variable
    global prior, topo, prior_icovinv

    """ Level I priors """
    # Prior on network parameters (log prior to force positivity)
    diff = [log(c) - gv.mean(log(prior['I']['c'][ci])) for ci, c in enumerate(p['I']['c'])]
    
    # Priors on neural network weights
    diff += [w - gv.mean(prior['I'][lyr + '.weight'][wind])
             for lyr in topo.keys() for wind, w in enumerate(p['I'][lyr + '.weight'])]
    
    # Return difference and covariance to calculate residual
    return {
        'icovroot': prior_icovsqrt,
        'diff': np.array(diff)
    }

# Explicit calculation of chi^2
def calculate_chi2(data, p):
    # Initialize chi2
    chi2 = 0.

    # Likelihood contribution
    for yi, yv in enumerate(data['y']):
        b, l = data['x'][yi]
        chi2v = (gv.mean(fit_fcn(b, l, p))[0] - gv.mean(yv))**2. / gv.sdev(yv)**2.
        chi2 += chi2v
    """
        plt.errorbar(
            gv.mean((b * gv.abs(p['I']['c'][0]) - 1.) * l**gv.abs(p['I']['c'][-1])),
            gv.mean((fit_fcn(b, l, p)[0] - gv.mean(yv)) / gv.sdev(yv)),
            gv.sdev((fit_fcn(b, l, p)[0] - gv.mean(yv)) / gv.sdev(yv)),
            capsize = 4., fmt = 'none'
        )
    plt.show()
    plt.close()
    """
 
    # Prior contribution
    prior_info = prior_fcn(p)
    prior_resid = gv.mean(np.matmul(prior_info['icovroot'], prior_info['diff']))
    
    # Return result
    return np.dot(prior_resid, prior_resid), chi2
        
if __name__ == '__main__':
    """ Set things up """
    # Choose FSS observable: 'u' for Binder cumulant & 'sus' for susceptibility
    obs = 'u'
    
    # Get data
    data = et.get_data(obs)

    # Define neural network topology as dictionary
    topo = {
        'lyr1': { # First layer
            'in': 1, 'out': 2, # Dimension of input & output
            'act': 'gelu', # Activation function
        },
        'lyr2': { # Second layer
            'in': 2, 'out': 2,
            'act': 'gelu',
        },
        'lyr3': { # Third layer
            'in': 2, 'out': 1,
            'act': 'linear',
        }
    }

    # Create neural network object
    nn = pnn.FeedforwardNeuralNetwork(topo)

    # Get priors & starting values for parameters
    prior, p0 = et.create_prior_and_p0(obs, topo, data, nn, 0)

    # Calculate inverse square root of covariance matrix
    prior_icovsqrt = linalg.cov_inv_SVD(
        gv.evalcov([prv for key in prior['I'] for prv in prior['I'][key]])
    )

    # Modify covariance of prior to accomodate log priors
    prior_icovsqrt[0, 0] = 1. / gv.sdev(log(prior['I']['c'][0]))
    prior_icovsqrt[1, 1] = 1. / gv.sdev(log(prior['I']['c'][1]))

    # Create fit object
    fit = swissfit.fit.SwissFit(
        data = data,
        prior = prior,
        p0 = p0,
        fit_fcn = fit_fcn,
        prior_fcn = {'I': prior_fcn}
    )

    """ Test chi2 """
    # Data array for Binder cumulant fit
    p = [
        2.26934097,  0.99786267,  0.1364224 , -0.46050109, -0.65872389,
        0.25681669,  4.53892139,  0.29327318,  0.01292595,  2.20402788,
        -0.27232019, -0.09147871, -0.86424891, -0.25088638,  0.80985857
    ] # From SGD run

    # Pass array through fit object without performing fit
    fit(None, x0 = p, fit = False)

    # Calculate chi^2 from fit object
    chi2_from_obj = fit.calculate_chi2(p)

    # Do explicit calculation of chi2
    chi2p, chi2l = calculate_chi2(data, fit.p)

    # Compare
    print('object/explicit/chi2p/chi2l:', chi2_from_obj, chi2p + chi2l, chi2p, chi2l)
