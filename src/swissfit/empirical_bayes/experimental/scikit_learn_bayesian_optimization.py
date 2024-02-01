from skopt import gp_minimize as _gp_minimize # Gaussian process optimizer from Scikit-learn
from .empirical_bayes import EmpiricalBayes as _EmpiricalBayes # Empirical Bayes parent class

# Class for Bayesian optimization with Scikit-learn
def ScikitLearnBayesianOptimization(_EmpiricalBayes):
    def __init__(self,
                 fcn = None,
                 bounds = None,
                 arguments = None):
        super().__init__(fcn = fcn, arguments = arguments, pool = None)
        self._bounds = bounds

    def __call__(self): return _gp_minimize(self._fcn, self._bounds, **self._args)
