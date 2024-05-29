import numpy as _numpy # Usual number crunching
import scipy as _scipy # Also usual number crunching
import gvar as _gvar # For converting MC data to GVar variables

# Monte Carlo base class
class MonteCarlo(object):
    def __init__(self,
                 mode = None,
                 monte_carlo_arguments = None
                 ):
        self._args = monte_carlo_arguments
        self._mode = mode
        self.method = 'MCMC'
        self.local_estimator = 'none'

    """
    Multivariate autocorrelation time:

    Calculated from multivariate effective sample size, as described 
    in arXiv:1512.07713. The effective sample size is

    ESS = n * (|Sigma| / |sigma|)^(1/p),

    where n is the number of samples, Sigma is the covariance matrix
    on the full dataset, sigma is the covariance matrix on the asmpytotically
    optimal batch of size n^(1/3), and p is the number of parameters that
    are being estimated from the distribution. The multivariate autocorrelation
    time is estimated as

    tau = n / ESS.

    Note that this is a bit different than the quantity-dependent autocorrelation
    time that you may be used to. This autocorrelation time does not capture the 
    autocorrelation time of individual observables, which may be vastly different.
    """
    def _integrated_autocorrelation_time(self):
        # Set parameters
        batch_size = self.nsamples**(1./3.) # <--- asymptotically optimal batch size
        nbatches = int(round(self.nsamples / batch_size))

        # Covariance matrix of original samples
        cov = _numpy.sum([
            _numpy.outer(resid, resid) for resid in self.samples - _numpy.mean(self.samples)
        ], axis = 0) / (self.nsamples - 1.)
        covdet = _numpy.linalg.det(cov)
            
        # Split data into batches
        batches = _numpy.array_split(self.samples, nbatches)
        batch_sizes = [len(batch) for batch in batches]
        mean_batch_size = _numpy.mean(batch_sizes)
        batch_means = [_numpy.sum(batch, axis = 0) / batch_sizes[batch_ind]
                       for batch_ind, batch in enumerate(batches)]
        batch_resids = _numpy.array(batch_means) - _numpy.mean(batch_means, axis = 0)
            
        # Calculate covariance
        batch_cov = _numpy.sum([
            len(batches[batch_ind]) * _numpy.outer(batch_resid, batch_resid)
            for batch_ind, batch_resid in enumerate(batch_resids)
        ], axis = 0) / (len(batches) - 1.)
        batch_covdet = _numpy.linalg.det(batch_cov)
            
        # Calculate effective sample size
        self.ess = len(batches) * mean_batch_size * (covdet / batch_covdet)**(1. / self._nparams)

        # Return autocorrelation time
        return self.nsamples / self.ess

    """
    Process Monte Carlo (MC) data:

    Calculates mean & covariance of MC data. Covariance includes 
    effect from autocorrelation via a calculation of the "multivariate
    autocorrelation time". 
    """
    def process_mc_data(self, return_x = False):
        # Convenience parameters
        self.nsamples, self._nparams = _numpy.shape(self.samples)

        # Calculate autocorrelation time from multivariate effective sample size
        self.integrated_time = self._integrated_autocorrelation_time()

        # Get mean & covariance of parameters
        gvar_variables = _gvar.dataset.avg_data(self.samples)

        # Create GVar variables for fit paramters including autocorrelation time
        if self._mode == 'simulation':
            x = _gvar.gvar(
                _gvar.mean(gvar_variables),
                self.integrated_time * _gvar.evalcov(gvar_variables)
            )
        elif self._mode == 'regression':
            x = _gvar.gvar(
                _gvar.mean(gvar_variables),
                self.ess * _gvar.evalcov(gvar_variables)
            )

        # Return result of fit parameters
        if return_x: return x
        else: self.x = x

    """
    Get fit parameters when obj.x called
    """

    # Process MC data
    def _getx(self): return self.process_mc_data(return_x = True)

    # Calling obj.x will invoke _getx
    x = property(_getx)
