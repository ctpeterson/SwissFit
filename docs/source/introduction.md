## SwissFit
[![Python](https://img.shields.io/badge/Python-3.10+-brightgreen.svg)](https://www.python.org)
[![License: GPL v2](https://img.shields.io/badge/license-MIT-blue)](https://github.com/ctpeterson/SwissFit?tab=MIT-1-ov-file)
[![Documentation Status](https://readthedocs.org/projects/swissfit/badge/?version=latest)](https://swissfit.readthedocs.io/en/latest/?badge=latest)

`SwissFit` is a general-purpose library for fitting models to data with Gaussian-distributed noise. The design of `SwissFit` is inspired by Peter Lepage's [lsqfit](https://github.com/gplepage/lsqfit) and operates in a similar manner. As such, it builds on top of the [GVar](https://github.com/gplepage/gvar) library and extensively utilizes the powerful numerical tools of [SciPy](https://scipy.org/) and [Vegas](https://github.com/gplepage/vegas) to estimate model parameters and their associated uncertainties. 

`SwissFit` currently supports the following.

  `-` [lsqfit](https://github.com/gplepage/lsqfit)-style least squares fitting, including priors. Priors can be transformed to represent constraints. Quality of fit and model selection criteria directly available from fit. 

  `-` Fully integrated with [GVar](https://github.com/gplepage/gvar), which allows fit parameters to be propagated into a secondary analysis with full automatic error propagation.

  `-` Bayesian model averaging in the flavor of [PRD103(2021)114502](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.103.114502) and [PRD109(2024)014510](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.014510). 

  `-` Markov Chain Monte Carlo (MCMC-based) parameter estimation via Peter Lepage's [Vegas](https://github.com/gplepage/vegas) library. Similar to the MCMC estimation already available in [lsqfit](https://github.com/gplepage/lsqfit). 

  `-` Support for integrating radial basis function networks and feedforward neural networks in least-squares model function

  `-` Optimization with [SciPy](https://scipy.org/)'s least_squares optimization methods (trust region reflective, Levenberg-Marquardt, dogbox), [SciPy](https://scipy.org/)'s "minimize" local optimization methods (BFGS, Nelder-Mead, conjugate gradient, etc.), and [SciPy](https://scipy.org/)'s basin hopping global optimization algorithm

  `-` Basic support for surrogate-based empirical Bayes in the flavor of [arXiv:2402.04175](https://arxiv.org/abs/2402.04175)

The following are planned or already in the works for `SwissFit`

  `-` Optimization with stochastic gradient descent, specifically Adam and its Nesterov-accelerated counterpart

  `-` Options for other Markov Chain Monte Carlo algorithms, such as Hamiltonian Monte Carlo

  `-` Empirical Bayes via [Scikit-learn](https://scikit-learn.org/stable/)'s Bayesian optimization module

  `-` Support for some form of non-parameteric regression

  `-` Much more! 

`SwissFit` is currently in beta. Help us get to a v1.0.0 release by providing feedback and letting me know if you run into problems! Thank you for considering to use `SwissFit` for whatever problem that you are trying to tackle!

## Requirements and installation

To install `SwissFit`, open up a terminal and enter the following commands.
```
# Upgrade pip - optional, but recommended
pip3 install --upgrade pip

# Install SwissFit
pip3 install swissfit
```
That's all! Upon installation, the following packages will also been installed. Note that `SwissFit` requires `Python>=3.10` to run.

