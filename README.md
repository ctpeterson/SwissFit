# SwissFit

<p align="center">
  <img src="https://github.com/ctpeterson/SwissFit/blob/main/SwissFit_logo.png">
</p>

SwissFit is a general-purpose library for fitting models to data with Gaussian-distributed noise. The design of this library is inspired by Peter Lepage's [lsqfit](https://github.com/gplepage/lsqfit) and operates in a similar manner. As such, it builds on top of the [GVar](https://github.com/gplepage/gvar) library and extensively utilizes the powerful numerical tools of [SciPy](https://scipy.org/) and [Scikit-learn](https://scikit-learn.org/stable/). SwissFit is readily deployable; however, it is under active development.

If you are here because you looked at "Constrained curve fitting for semi-parametric models with radial basis function networks" by Curtis Taylor Peterson and Anna Hasenfratz ([arXiv:2402.04175](https://arxiv.org/abs/2402.04175)), I have provided `Jupyter` notebooks that reproduce our results from that paper under the `examples` folder.

## Features

`SwissFit` currently supports the following.

  - [lsqfit](https://github.com/gplepage/lsqfit)-style least squares fitting (`examples/simple_fit.ipynb`), including priors. Priors can be transformed to represent some constraints. Quality of fit and model selection criteria directly available from fit.
  - Fully integrated with [GVar](https://github.com/gplepage/gvar), which allows fit parameters to be propagated into a secondary analysis with full automatic error propagation
  - Support for integrating radial basis function networks (`examples/simple_radial_basis_function_fit.ipynb`) and feedforward neural networks (example notebook coming soon) in least-squares model function
  - Optimization with [SciPy](https://scipy.org/)'s trust region reflective local optimization algorithm (`examples/simple_fit.ipynb`) and/or [SciPy](https://scipy.org/)'s basin hopping global optimization algorithm (`examples/simple_radial_basis_function_fit.ipynb`)
  - Basic support for surrogate-based empirical Bayes ([arXiv:2402.04175](https://arxiv.org/abs/2402.04175); see any of the notebooks under `examples` that reproduce the results from that paper).

The following are planned or already in the works for `SwissFit`

  - Optimization with [SciPy](https://scipy.org/) `minimize` for local optimization and [SciPy](https://scipy.org/)'s various global optimization algorithms
  - Optimization with stochastic gradient descent, specifically Adam and its Nesterov-accelerated counterpart
  - Empirical Bayes via [Scikit-learn](https://scikit-learn.org/stable/)'s Bayesian optimization module
  - Model parameter estimation by direct sampling of posterior distributions
  - Extended support for hierarchical Bayesian modelling

`SwissFit` is currently in beta. Help us get to a v1.0.0 release by providing feedback and letting me know if you run into problems! Thank you for considering to use `SwissFit` for whatever problem that you are trying to tackle!

## Requirements

  - `Python>=3.10`
  - [Numpy](https://github.com/numpy/numpy)
  - [SciPy](https://scipy.org/)
  - [Scikit-learn](https://scikit-learn.org/stable/)
  - [GVar](https://github.com/gplepage/gvar)
  - [Matplotlib](https://github.com/matplotlib/matplotlib)

All versions of the above libraries should at least be compatible with `Python>=3.10`. Library dependencies are automatically installed.

## Installation

SwissFit will be uploaded to PyPI for simple installation in the near future. For now, install SwissFit as follows. First, clone this repository into whatever folder that you wish. Then `cd` into your cloned directory for SwissFit and install by running `setup.py` as
```
python3 setup.py install
```
That's all. The `setup.py` script will install SwissFit for you, along with all of SwissFit's dependences; namely, [Numpy](https://github.com/numpy/numpy), [SciPy](https://scipy.org/), [Scikit-learn](https://scikit-learn.org/stable/), [GVar](https://github.com/gplepage/gvar), and [Matplotlib](https://github.com/matplotlib/matplotlib).

## Basic example usage

Let's get familiar with SwissFit by fitting a simple sine function. The full example code can be found under `examples/simple_fit.py` or `examples/simple_fit.ipynb`. Choose the sine function to be
$$f(x) = a\sin(bx),$$
with $a=2.0$ and $b=0.5$. First, let's import everything that we'll need.
```
""" SwissFit imports """
from swissfit import fit # SwissFit fitter
from swissfit.optimizers import scipy_least_squares # SciPy's trust region reflective

""" Other imports """
import gvar as gvar # Peter Lepage's GVar library
import numpy as np # NumPy
```
To extract the parameters of the sine function from data, we need to define a fit function; let's do so:
```
def sin(x, p):
    return p['c'][0] * gvar.sin(p['c'][-1] * x)
```
SwissFit operates around Python dictionaries. Therefore, you'll see that the fit parameters are encoded by a Python dictionary in our fit function. Now we need data. Let's create a function that generates an artificial dataset for us to fit to.
```
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
```
This function takes in the values for $a$, $b$ and the error that we want our artificial dataset to possess. It returns a dictionary with inputs `data['x']` in $[0,2\pi/b]$ and outputs `data['y']` that are uncorrelated [GVar](https://github.com/gplepage/gvar) variables. Note that SwissFit is fully capable of handling correlated [GVar](https://github.com/gplepage/gvar) variables. This dictionary of inputs is what we will feed into SwissFit. Before we create our SwissFit object, let's generate our artificial dataset and define our priors.
```
# Artificial dataset
data = create_dataset(
  2.0, # a
  0.5, # b
  0.1  # error
)
    
# Create priors
prior = {'c': [gvar.gvar('1.5(1.5)'), gvar.gvar('0.75(0.75)')]}
```
Again, SwissFit operates around Python dictionaries. Therefore, you see that both our dataset and priors are defined as Python dictionaries. We're now ready to create our SwissFit object.
```
fitter = fit.SwissFit(
    data = data,
    prior = prior,
    fit_fcn = sin,
)
```
To fit to data, we also need to create an optimizer object. We do so by passing the SwissFit object through the optimizer object's constructor.
```
optimizer = scipy_least_squares.SciPyLeastSquares(fitter = fitter)
```
Now we are ready to fit. It is as simple as passing the SwissFit optimizer object through the call method of the SwissFit object
```
fitter(optimizer)
```
Now that we have done our fit, we can print the output and save our (correlated) fit parameters.
```
print(optimizer)
fit_parameters = fitter.p
```
The output of print is:
```
SwissFit: 🧀
   chi2/dof [dof] = 1.04 [20]   Q = 0.41   (Bayes) 
   chi2/dof [dof] = 1.15 [18]   Q = 0.3   (freq.) 
   AIC [k] = 24.63 [2]   logML = 7.511*

Parameters*:
     c
             1                  2.007(33)   [1.5(1.5)]
             2                 0.4990(21)   [0.75(75)]

Estimator:
   SwissFit optimizer object
*Laplace approximation
```
We can also grab many quality of fit & information criteria directly from `fitter` as follows.
```
print(
    'chi2_data:', fitter.chi2_data,
    '\nchi2_prior:', fitter.chi2_prior,
    '\nchi2:', fitter.chi2,
    '\ndof (Bayes):', fitter.dof,
    '\ndof (freq.):', fitter.frequentist_dof,
    '\np-value:', fitter.Q,
    '\nmarginal likelihood:', fitter.logml,
    '\nAkaike information criterion:', fitter.aic
)
```
The output of the above print statement is
```
chi2_data: 20.628697369539452 
chi2_prior: 0.22612697167343043 
chi2: 20.854824341212883 
dof (Bayes): 20 
dof (freq.): 18 
p-value: 0.40572144469143007 
marginal likelihood: 7.511209597426163 
Akaike information criterion: 24.628697369539452
```
Because the output of `fitter.p` are correlated [GVar](https://github.com/gplepage/gvar) variables, we can pass these parameters through any function that we want and get an output with Gaussian errors fully propagated through. For example, we could calculate `f(0.5)` and `f(1.0)`, along with the their covariance
```
# Calculate f(0.5, f(1.0)
fa = sin(0.5, fit_parameters)
fb = sin(1.0, fit_parameters)

# Print f(0.5) & f(1.0)
print('f(0.5) f(1.0):', fa, fb)
    
# Print covariance matrix of (fa, fb)
print('covariance of f(0.5) & f(1.0):\n', gvar.evalcov([fa, fb]))
```
We could do the same thing for any other derived quantity. That's the power of automatic error propagation by automatic differentiation! The output of the above block of code is:
```
f(0.5) f(1.0): 0.4955(85) 0.960(16)
covariance of f(0.5) & f(1.0):
 [[7.29612481e-05 1.40652271e-04]
 [1.40652271e-04 2.71200285e-04]]
```
Okay, that's all fine an dandy, but how to we visualize the result of our fit? This is no longer a exercise in using `SwissFit` - we now simply manipulate the [GVar](https://github.com/gplepage/gvar) variables that we get from our fit. To produce the plot above, we use [Matplotlib](https://github.com/matplotlib/matplotlib).
```
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
```
This produces the following figure.
<p align="center">
  <img src="https://github.com/ctpeterson/SwissFit/blob/main/simple_fit.png">
</p>
More realistic examples can be found under the `examples` folder. 

## Basic fit with a radial basis function network

Let's try and interpolate over the sine function from `examples/simple_fit.ipynb` with a radial basis function network (RBFN). The sine function is
$$f(x) = a\sin(bx),$$
with $a=2.0$ and $b=0.5$. First, let's import everything we need.
```
""" SwissFit imports """
from swissfit import fit # SwissFit fitter
from swissfit.optimizers import scipy_basin_hopping # Basin hopping global optimizer
from swissfit.optimizers import scipy_least_squares # Trust region reflective local optimizer
from swissfit.machine_learning import radial_basis # Module for radial basis function network

""" Other imports """
import gvar as gvar # Peter Lepage's GVar library
import numpy as np # NumPy
```
So that we have something to fit to, let's create an artificial dataset. We do so in the next block.
```
# Parameters of the sine function & the error
a, b, error = 2.0, 0.5, 0.1

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
        np.random.normal(a * np.sin(b * xx), error), # Random mean
        error # Error on mean
    )
    for xx in data['x']
]
```
Next, let's create a radial basis function network. We do so by first specifying the topology of the RBFN. The following RBFN will have two nodes in its hidden layer.
```
network_topology = {
    'lyr1': { # Hidden layer
        'in': 1, 'out': 2, # Dimension of input & output
        'activation': 'exp', # Exponential activation
    },
    'lyr2': { # Output layer
        'in': 2, 'out': 1,
        'activation': 'linear' # Linear activation
    }
}
```
In `SwissFit`, we create a RBFN by passing the above dictionary to a `RadialBasisFunctionNeuralNetwork` object constructor as follows.
```
# Create radial basis function network
neural_network = radial_basis.RadialBasisNeuralNetwork(network_topology)

# Initialize radial basis function network parameters
p0 = neural_network.initialize_parameters(initialization = 'zero', p0 = {})
```
That's it! Now let's define our fit function using the instance of the `RadialBasisFunctioNeuralNetwork` class that we just created.
```
def fit_fcn(x, p):
    return np.ravel(neural_network.out(x, p))
```
Now that we have our fit function, let's go ahead and fit. Because the loss landscape is much more complicated than it is in examples/simple_fit.ipynb, we will use a global optimizer. Everything else is exactly the same.
```
# Basin hopping parameters
niter_success = 200 # Number of iterations with same best fit parameters for basin hopping to "converge"
niter = 10000 # Upper bound on total number of basin hopping iterations
T = 1. # Temperature hyperparameter for basin hopping

# Create SwissFit fit object
fitter = fit.SwissFit(
    udata = data, # Fit data; "data = data" is also acceptable - "udata" means "uncorrelated"
    p0 = p0, # Starting values for parameters,
    fit_fcn = fit_fcn, # Fit function
)

# Create trust region reflective local optimizer from SciPy - fitter will save reference to local_optimizer for basin hopping
local_optimizer = scipy_least_squares.SciPyLeastSquares(fitter = fitter)

# Basin hopping global optimizer object instantiation
global_optimizer = scipy_basin_hopping.BasinHopping(
    fitter = fitter, # Fit function is the "calculate_residual" method of fitter object
    optimizer_arguments = {
        'niter_success': niter_success,
        'niter': niter,
        'T': T
    }
)
```
Let's fit!
```
# Do fit
fitter(global_optimizer)

# Print result of fit
print(fitter)

# Save fit parameters
fit_parameters = fitter.p
```
The output of `print(fitter)` is
```
SwissFit: 🧀
   chi2/dof [dof] = 2.06 [13]   Q = 0.01   (Bayes) 
   chi2/dof [dof] = 2.06 [13]   Q = 0.01   (freq.) 
   AIC [k] = 40.78 [7]   logML = -1.626*

Parameters*:
     lyr1.center
             1                  9.357(70)   [n/a]
             2                  3.124(64)   [n/a]
     lyr1.bandwidth
             1                 -0.186(28)   [n/a]
             2                  0.169(23)   [n/a]
     lyr2.weight
             1                  -2.01(17)   [n/a]
             2                   2.18(17)   [n/a]
     lyr2.bias
             1                  -0.03(17)   [n/a]

Estimator:
   algorithm = SciPy basin hopping
   minimization_failures = 5
   nfev = 13464
   njev = 11988
   fun = 13.39159977365442
   message = ['success condition satisfied']
   nit = 362
   success = True

*Laplace approximation
```
Let's visualize what our fit looks like.
```
# Import Matplotlib
import matplotlib.pyplot as plt

# Plot fit data
plt.errorbar(
    data['x'], 
    gvar.mean(data['y']), 
    gvar.sdev(data['y']), 
    color = 'k', markerfacecolor = 'none',
    markeredgecolor = 'k',
    capsize = 6., fmt = 'o',
    label = 'data'
)

# Get result of fit function
x = np.linspace(data['x'][0], data['x'][-1], 100)
y = fit_fcn(x, fit_parameters)

# Plot error of fit function from fit as a colored band
plt.fill_between(
    x,
    gvar.mean(y) - gvar.sdev(y),
    gvar.mean(y) + gvar.sdev(y),
    color = 'maroon', alpha = 0.5,
    label = 'RBFN'
)

# x/y label
plt.xlabel('x', fontsize = 20.)
plt.ylabel('$a\\sin(bx)$', fontsize = 20.)

# Show legend
plt.legend()

# Grid
plt.grid('on')
```
The output of the code above is
<p align="center">
  <img src="https://github.com/ctpeterson/SwissFit/blob/main/simple_rbfn_fit.png">
</p>
