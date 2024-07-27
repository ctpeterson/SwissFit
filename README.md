# SwissFit
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org)
[![License: GPL v2](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ctpeterson/SwissFit?tab=MIT-1-ov-file)
[![Documentation Status](https://readthedocs.org/projects/swissfit/badge/?version=latest)](https://swissfit.readthedocs.io/en/latest/?badge=latest)

<p align="center">
  <img src="https://github.com/ctpeterson/SwissFit/blob/main/SwissFit_logo.png">
</p>

`SwissFit` is a general-purpose library for fitting models to data with Gaussian-distributed noise. The design of this library is inspired by Peter Lepage's [lsqfit](https://github.com/gplepage/lsqfit) and operates in a similar manner. As such, it builds on top of the [GVar](https://github.com/gplepage/gvar) library and extensively utilizes the powerful numerical tools of [SciPy](https://scipy.org/) and [Vegas](https://github.com/gplepage/vegas) to extract model parameters and their associated statistical uncertainties from maximum a posteriori estimation (i.e., traditional least squares fitting with Gaussian priors) or Markov chain Monte Carlo sampling. 

The original intent of `SwissFit` was to provide a library for fitting simple datasets with either feedforard or radial basis function neural networks; however, it as grown out of its initial purpose and now serves as an independent scientific library focused on fitting general models to data. As such, `SwissFit` supports fitting with both feedfoward and radial basis function neural networks. Generic fits using maximum a posteriori estimation (via minimization of an augmented $\chi^2$) can use any of [SciPy](https://scipy.org/)'s local optimization algorithms from "[least squares](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)", [SciPy](https://scipy.org/)'s local optimization algorthms from "[minimize](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html)", and [SciPy](https://scipy.org/)'s implementation of the [basin hopping](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.basinhopping.html) global optimization algorithm. `SwissFit` also ships with basic tools for one-dimensional empirical Bayes; however, multi-dimensional empirical Bayes via Bayesian optimization is in the works. Also in the works is support for generic Bayesian model averaging. 

The current version of `SwissFit` is in beta status; as such, please consider it an "early release". If you have any suggestions for bug fixes or additional features, I would love to hear them, though I can only guarantee that bug fixes will be implemented.

If you are here because you looked at "Constrained curve fitting for semi-parametric models with radial basis function networks" by Curtis Taylor Peterson and Anna Hasenfratz ([arXiv:2402.04175](https://arxiv.org/abs/2402.04175)), I have provided `Jupyter` notebooks that reproduce our results from that paper under the `examples` folder. These examples will work with v0.1 of SwissFit, which is downloadable under the "releases" tab.

## Acknowledgement

If you use `SwissFit`, please consider citing this repository (see "cite the repository" on the right) or [arXiv:2402.04175](https://arxiv.org/abs/2402.04175). If you use the `VegasLepage` module to perform your fits using Markov chain Monte Carlo estimation, please acknowledge [Vegas](https://github.com/gplepage/vegas). 

Additionally, if you use our XY model data under `examples/v0p1/example_data/clockinf` for your research, please acknowledge [USQCD](https://www.usqcd.org/) resources by adding the following statement to your acknowledgements.

"*The data used in this work was generated using the computing and long-term storage facilities of the USQCD Collaboration, which are funded by the Office of Science of the U.S. Department of Energy.*"

If you use any other spin model data in this repository (2- & 3-state Potts, along with 4-state clock), please acknowledge [SpinMonteCarlo.jl](https://github.com/yomichi/SpinMonteCarlo.jl).

## Features

`SwissFit` currently supports the following.

  - [lsqfit](https://github.com/gplepage/lsqfit)-style least squares fitting, including priors. Priors can be transformed to represent constraints. Quality of fit and model selection criteria directly available from fit. See the example below and under `examples/map_fit.py`.
  - Fully integrated with [GVar](https://github.com/gplepage/gvar), which allows fit parameters to be propagated into a secondary analysis with full automatic error propagation
  - Bayesian model averaging in the flavor of [PRD103(2021)114502](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.114502) and [PRD109(2024)014510](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.014510). All model-averaged parameters returned by SwissFit are correlated! See the example below and under `examples/model_average_correlation_function.py`.
  - Markov Chain Monte Carlo (MCMC-based) parameter estimation via Peter Lepage's [Vegas](https://github.com/gplepage/vegas) library. Similar to the MCMC estimation already available in [lsqfit](https://github.com/gplepage/lsqfit). See the example below and under `examples/mcmc_fit.py`.
  - Support for integrating radial basis function networks and feedforward neural networks in least-squares model function
  - Optimization with [SciPy](https://scipy.org/)'s least_squares optimization methods (trust region reflective, Levenberg-Marquardt, dogbox), [SciPy](https://scipy.org/)'s "minimize" local optimization methods (BFGS, Nelder-Mead, conjugate gradient, etc.), and [SciPy](https://scipy.org/)'s basin hopping global optimization algorithm
  - Basic support for surrogate-based empirical Bayes ([arXiv:2402.04175](https://arxiv.org/abs/2402.04175))

The following are planned or already in the works for `SwissFit`

  - Proper documentation
  - Optimization with stochastic gradient descent, specifically Adam and its Nesterov-accelerated counterpart
  - Options for other Markov Chain Monte Carlo algorithms, such as Hamiltonian Monte Carlo
  - Empirical Bayes via [Scikit-learn](https://scikit-learn.org/stable/)'s Bayesian optimization module

`SwissFit` is currently in beta. Help us get to a v1.0.0 release by providing feedback and letting me know if you run into problems! Thank you for considering to use `SwissFit` for whatever problem that you are trying to tackle!

## Requirements

  - `Python>=3.10`
  - [Numpy](https://github.com/numpy/numpy)
  - [SciPy](https://scipy.org/)
  - [Scikit-learn](https://scikit-learn.org/stable/)
  - [GVar](https://github.com/gplepage/gvar)
  - [Vegas](https://github.com/gplepage/vegas)
  - [Matplotlib](https://github.com/matplotlib/matplotlib)

All versions of the above libraries should at least be compatible with `Python>=3.10`. Library dependencies are automatically installed.

## Installation

SwissFit will be uploaded to PyPI for simple installation sometime in the near future. For now, install SwissFit as follows. First, clone this repository into whatever folder that you wish. Then `cd` into your cloned directory for SwissFit and install by running `setup.py` as
```
python3 setup.py install
```
That's all. The `setup.py` script will install SwissFit for you, along with all of SwissFit's dependences; namely, [Numpy](https://github.com/numpy/numpy), [SciPy](https://scipy.org/), [Scikit-learn](https://scikit-learn.org/stable/), [GVar](https://github.com/gplepage/gvar), [Vegas](https://github.com/gplepage/vegas), and [Matplotlib](https://github.com/matplotlib/matplotlib).

## Basic example usage

Let's get familiar with SwissFit by fitting a simple sine function. The full example code can be found under `examples/simple_fit.py` or `examples/simple_fit.ipynb`. Choose the sine function to be
$$f(x) = a\sin(bx),$$
with $a=2.0$ and $b=0.5$. First, let's import everything that we'll need.
```
""" SwissFit imports """
from swissfit import fit # SwissFit fitter
from swissfit.optimizers import scipy_least_squares # SciPy's least squares methods

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
To fit to data, we also need to create an optimizer object:
```
optimizer = scipy_least_squares.SciPyLeastSquares()
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
   AIC [k] = 24.85 [2]   logML = 7.511*

Parameters*:
     c
             1                  2.007(33)   [1.5(1.5)]
             2                 0.4990(21)   [0.75(75)]

Estimator:
   algorithm = SciPy least squares
   fun = 10.427412170606441
   optimality = 0.0017098526837675543
   nfev = 16
   njev = 14
   status = 2
   message = `ftol` termination condition is satisfied.
   success = True
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
Because the output of `fitter.p` are correlated [GVar](https://github.com/gplepage/gvar) variables, we can pass these parameters through any function that we want and get an output with Gaussian errors fully propagated through. For example, we could calculate `f(0.5)` and `f(1.0)`, along with their covariance
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

## Markov Chain Monte Carlo estimation

The example above estimates the mean and covariance of fit parameters via maximum a posteriori estimation (MAP). The MAP estimate for the mean can be poor because what we are calculating is really the posterior mode. Alternatively, we can estimate the mean of the fit parameters by sampling directly from the posterior. SwissFit has budding support for this kind of parameter estimation, which currently only supports sampling via the Vegas algorithm (see [arXiv:2009.05112](https://arxiv.org/abs/2009.05112) for details). The infrastructure for sampling with Vegas is provided by Peter Lepage's [Vegas](https://github.com/gplepage/vegas) library. MCMC estimation with SwissFit is simple and follows essentially the same steps as the MAP estimation example above. Simply replace
```
from swissfit.optimizers import scipy_least_squares

...

optimizer = scipy_least_squares.SciPyLeastSquares()
fitter(optimizer)
```
with
```
from swissfit.monte_carlo import vegas as vegas_lepage

...

estimator = vegas_lepage.VegasLepage()
fitter(estimator)
```
Everything else is the same, including printing out information about the fit. The printout should look like the following. Note, however, that SwissFit does not currently calculate the correlation between the fit parameters and the underlying dataset for MCMC-based estimates of the fit parameters. I hope to alleviate this deficit in the future.
```
SwissFit: 🧀
   chi2/dof [dof] = 1.04 [20]   Q = 0.41   (Bayes) 
   chi2/dof [dof] = 1.15 [18]   Q = 0.3   (freq.) 
   AIC [k] = 24.86 [2]   logML = 7.5247(23)

Parameters:
     c
             1                  2.006(33)   [1.5(1.5)]
             2                 0.4990(21)   [0.75(75)]

Estimator:
   algorithm = Peter Lepage's Vegas++
   nitn = 10 (adapt) 
   nitn = 10 (MCMC)
```


## Simple Bayesian model averaging: subset selection

`SwissFit` supports model averaging in the form of "Bayesian model averaging". See [PRD103(2021)114502](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.114502) and [PRD109(2024)014510](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.014510) for details. Let's go through the example outlined in Section IV.A of [PRD103(2021)114502](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.114502) using `SwissFit`. The code for generating the data in this example can be found in the [companion code](https://github.com/jwsitison/improved_model_avg_paper) for [PRD109(2024)014510](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.014510). For completeness, it is reproduced below.
```
import gvar as gv
import numpy as np

def create_synthetic_data(
        a0=2.0,a1=10.4,
        e0=0.8,e1=1.16,
        NT=32,
        rho=0.6,
        var=0.09,
        nsamp=1000
    ): 
    # arXiv:2008.01069 
    # jwsitison/improved_model_avg_paper/improved_model_averaging/synth_data.py
    def F(nt): return a0*np.exp(-e0*nt)+a1*np.exp(-e1*nt)
    def corr_fcn(nt,ntp): return rho**np.abs(nt-ntp)

    Fnt = np.fromfunction(F,(NT,))
    corr = np.fromfunction(corr_fcn, (NT,NT))
    eta = gv.raniter(gv.correlate([gv.gvar(0.,np.sqrt(var))]*NT,corr))
    
    dataset = [Fnt*(1.+next(eta)) for _ in range(nsamp)]
    return [*range(NT)], gv.dataset.avg_data(dataset)

xdata,ydata = create_synthetic_data()
```
Now that we have our synthetic dataset, let's define a model function for it.
```
def model(nt,p):
    return p['a0'][0]*gv.exp(-p['e0']*nt)
```
We want to extract a model-averaged estimate for `a0` and `e0` from fits to subsets of the data created by `create_synthetic_data()`. The following code utilizes `SwissFit` to extract an estimate of `a0` and `e0` over the subsets explored in [PRD103(2021)114502](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.114502). Each fit is performed as in the latter two examples and they are collected in an array called `fits`.
```
from swissfit import fit as fitter
from swissfit.optimizers import scipy_least_squares

tmin_max = 28
p0 = {'a0': [0.1], 'e0': [0.1]}
optimizer = scipy_least_squares.SciPyLeastSquares()

fits = [
  fitter.SwissFit(
    data = {'x': xdata[nt:], 'y': ydata[nt:]},
    p0 = p0,
    fit_fcn = model
  )(optimizer)
  for nt in range(0,tmin_max+1)
]
```
To model average, we simply create a `BayesianModelAveraging` object by passing the above array of `SwissFit` fits and the entire dataset (`ydata`) into its constuctor.
```
from swissfit.model_averaging import model_averaging
model_average = model_averaging.BayesianModelAveraging(models=fits, ydata=ydata)
```
To get the result of the model average, simply call `model_average.p`, as we do with regular `SwissFit` objects.
```
model_average_result = model_average.p
a0,e0 = model_average_result['a0'][0],model_average_result['e0'][0]
print('a0:',a0)
print('e0:',e0)
```
The above code should yield the following result.
```
a0: 2.084(39)
e0: 0.80141(78)
```
Note that the `BayesianModelAveraging` class calculates the full model-averaged covariance matrix. Hence, the fit parameters in `model_average_result` dictionary above are fully correlated [GVar](https://github.com/gplepage/gvar) variables. For more details, see the example code under `examples/model_average_correlation_function.py`.
