""" SwissFit

Description of SwissFit...
"""

# Import global & local modules
import typing
from .numerical_tools import linalg as _linalg # Useful linear algebra tools
import gvar as _gvar # For automatic propagation of correlated errors
import numpy as _numpy # For vectorized numerical operations
from scipy.special import gammaincc as _gammaincc # For calculating p-value of fit chi^2
from functools import partial as _partial # For partial evaluation of functions
from scipy.linalg import block_diag as _block_diag # For getting block diagonal matrix
from scipy.linalg import det as _det # For determinants
from scipy.optimize import approx_fprime as _jac # For calculating Hessian
from functools import reduce as _reduce # For cleaner matrix calculations

# Dummy estimator object for None
class NoneEstimator(object):
    def __init__(self): self.method = 'none'

# SwissFit errors
class SwissFitException(Exception):
    def __init__(self, tag): self.tag = tag
    def __str__(self): return self.tag

# Parent fitter class
class _Fitter(object):
    def __init__(self, tag):
        self.tag = tag

    def _create_tracker(self, p): 
        self._trackerp = _numpy.array([*map(_gvar.gvar, p)])
        
    def call(self,
             estimator,
             p0 = None,
             p = None,
             approx_cov = True
             ):
        self._estimator = estimator
        self._approx_cov = approx_cov
        if p0 is not None: self.pflat = p0
        if self._estimator is not None:
            if self._estimator.method == 'MAP':
                match self._estimator.tag:
                    case 'scipy_least_squares':
                        self.fit = self._estimator(
                            self.pflat, self.calculate_residual, self.calculate_jacobian
                        )
                    case 'scipy_minimize':
                        self.fit = self._estimator(
                            self.pflat, self.calculate_chi2, self.calculate_gradient
                        )
                    case 'scipy_basin_hopping':
                        self.fit = self._estimator(
                            self.pflat,
                            {
                                'scipy_least_squares': self.calculate_residual,
                                'scipy_minimize': self.calculate_chi2
                            }[self._estimator.local_optimizer.tag],
                            {
                                'scipy_least_squares': self.calculate_jacobian,
                                'scipy_minimize': self.calculate_gradient
                            }[self._estimator.local_optimizer.tag]
                        )
                self.pmean = self.fit.x
            elif self._estimator.method == 'MCMC':
                match self._estimator.tag:
                    case 'vegas_peter_lepage':
                        self.fit = self._estimator(
                            self.prior if (p is None) and (self.tag == 'SwissFit') else p,
                            self.calculate_pdf
                        )
                        self.pmean = _gvar.mean(
                            [p for pkey, ps in self._estimator.p.items() for p in ps]
                        )
        else:
            self._estimator = NoneEstimator()
            self.pmean = self.p0 if p0 is None else p0
    
# Parent SwissFit class
class SwissFit(_Fitter):
    """Fit with SwissFit
    
    SwissFit objects can be used to fit data by maximum a posterior (MAP) estimation 
    or direct sampling from the posterior.

    Attributes
    ----------
    data : dict
        Dictionary of the form {'y': y_data} or {'x': x_data, 'y': y_data}.
        Each element of data['x'] must be a list/array with "n_args - 1" entries,
        where "n_args" is the number of positional arguments of 'fit_fcn'.
        For example, if fit_fcn has three positional arguments, then data['x']
        might look like data['x'] = [[a,b], [c,d], ...]. The entries of each
        element of data['x'] must be floats. data['y'] is a one-dimensional
        array of GVar variables.
    prior : dict
        Dictionary of lists/arrays containing priors. For example, 
        prior = {'a': prior_a, 'b': prior_b, ...} where
        prior_a = [aa, ab, ...] and prior_b = [ba, bb, ...]. Entries
        of lists/arrays must be GVar variables
    p0 : dict
        Dictionary of lists/arrays of starting values for fit parameters. 
        For example, p0 = {'a': p0_a, 'b': p0_b, ...} where p0_a = [aa, ab, ...] 
        and p0_b = [ba, bb, ...]. Entries of lists/arrays must be floats.
    pmean : dict
        Mean of fit parameters as dictionary
    pflat : Numpy array
        Flat array of fit parameter means
    fit : SciPy Optimizer
        SciPy Optimizer object for getting fit parameters

    Methods
    -------
    map_keys
    data_residual
    prior_residual
    calculate_residual
    calculate_chi2
    calculate_jacobian
    calculate_gradient
    calculate_hessian
    calculate_pdf
    
    """
    def __init__(self,
                 data = None, # Input/output data - {'x': xdata, 'y': ydata}
                 udata = None, # Uncorrelated input/output data - same as 'data'

                 prior = None, # Priors - dictionary
                 uprior = None, # Uncorrelated priors - dictionary

                 p0 = None, # Starting values for fit parameters

                 fit_fcn = None, # Function to be fit
                 prior_fcn = None, # Custom function for priors (optional)

                 prior_transformation_fcn = {}, # Custom transformation of priors (optional)

                 data_svdcut = None, # Apply SVD cut to data (optional)
                 prior_svdcut = None, # Apply SVD cut to prior (optional)
                 ):
        """SwissFit constructor method
        
        Parameters
        ----------
        data : dict
            Dictionary of the form {'y': y_data} or {'x': x_data, 'y': y_data}.
            Each element of data['x'] must be a list/array with "n_args - 1" entries,
            where "n_args" is the number of positional arguments of 'fit_fcn'.
            For example, if fit_fcn has three positional arguments, then data['x']
            might look like data['x'] = [[a,b], [c,d], ...]. The entries of each
            element of data['x'] must be floats. data['y'] is a one-dimensional
            array of GVar variables.
        udata : dict, alternative to 'data'
            Same as 'data', but with uncorrelated GVar variables
        prior : dict
            Dictionary of lists/arrays containing priors. For example, 
            prior = {'a': prior_a, 'b': prior_b, ...} where
            prior_a = [aa, ab, ...] and prior_b = [ba, bb, ...]. Entries
            of lists/arrays must be GVar variables
        uprior : dict, alternative to 'prior'
            Same as 'prior', but with uncorrelated GVar variables
        p0 : dict
            Dictionary of lists/arrays of starting values for fit parameters. 
            For example, p0 = {'a': p0_a, 'b': p0_b, ...} where p0_a = [aa, ab, ...] 
            and p0_b = [ba, bb, ...]. Entries of lists/arrays must be floats. If 'prior'
            contains dictionary entries that 'p0' does not, they will be filled in 
            with the mean values of those 'prior' entries. 
        fit_fcn : function
            Model function for data. Must be of the form fcn(a, b, ..., p), where 
            'a, b, ...' are positional arguments for input lists/arrays a, b, ...
            ordered according to data['x'] and 'p' is a dictionary of fit parameters
            specified by 'prior' and 'p0'.
        prior_fcn : None or function, optional
            Optional function to replace calculation of prior contribution to chi^2.
        prior_transformation_fcn : None or dict, optional
            Dictionary of functions to transformation priors with. For example, if
            you want log priors for the priors in prior['a'], then 
            prior_transformation_fcn would contain a function like
            prior_transformation_fcn['a'] = lambda x: gvar.log(x).
        data_svdcut : None or float, optional
            Cut to impose on singular values of data covariance matrix. If None, 
            then no cut is imposed.
        prior_svdcut : None or float, optional
            Same as data_svdcut, but for prior parameters
        
        """

        # Initialize parent class for fitter
        super().__init__("SwissFit")
        
        # Save fit functions
        if fit_fcn is not None: self.fit_fcn, self.prior_fcn = fit_fcn, prior_fcn
        else: raise SwissFitException("'fit_fcn' must be specified")
        if prior_transformation_fcn is not None:
            self.prior_transformation_fcn = prior_transformation_fcn
        else: self.prior_transformation_fcn = {}
            
        # Check if data specified. If so, check if it is correlated.
        if (data is None) and (udata is None):
            raise SwissFitException("'data' or 'udata' must be specified")
        if udata is None: self.data, self._correlated_data = data, True
        else: self.data, self._correlated_data = udata, False
        
        # Check if prior specified. If so, check if it is correlated.
        if (prior is None) and (uprior is None): prior, self._prior_specified = {}, False 
        else: self._prior_specified = True
        if uprior is None: self.prior, self._correlated_prior = prior, True
        else: self.prior, self._correlated_prior = uprior, False

        # Check if starting values for fit specified
        if (p0 is None) and (prior is None) and (uprior is None):
            raise SwissFitException("at least 'p0', 'prior', or 'uprior' must be specified")
        else: self.p0 = p0 if p0 is not None else {}

        # Check if SVD cuts specified
        self.data_svdcut, self.prior_svdcut = data_svdcut, prior_svdcut
        
        # Prepare data & priors
        self._prepare_data_and_prior()

        # Prepare fit functions
        self._prepare_fit_fcns()

    def map_keys(self, p, return_parameters = False):
        """Convert list to dictionary
        
        Takes in a list of fit parameters "p" and converts them into
        a dictionary that can be understood by the user. Dictionary 
        keys are specified & ordered according to self.prior and self.p0.

        Regardless of whether return_parameters is True, resulting
        dictionary is saved as private _pdict attribute.

        Args
        ----
        p : list
            Fit parameters as one-dimensional list/array
        return_parameters : bool, optional
            Specify whether or not dictionary of fit parameters
            should be returned to user.

        Returns
        -------
        pdict : dictionary
            Dictionary of fit parameters. Returned if 
            return_parameters is True.

        """
        for key in self.p0.keys():
            self._pdict[key] = p[self._plengths[key][0]:self._plengths[key][-1]]
        if return_parameters: return self._pdict

    def __call__(self,
             estimator,
             p0 = None,
             p = None,
             approximate_parameter_covariance = True
             ):
        self.call(estimator, p0 = p0, p = p, approx_cov = approximate_parameter_covariance)
        return self
        
    def _prepare_data_and_prior(self):
        # Check to see if 'x' and/or 'y' specified in 'data'
        self._x_specified = True if 'x' in self.data.keys() else False
        if self._x_specified:
            _ndim, _xlen = _numpy.array(self.data['x']).ndim, _numpy.array(self.data['x']).shape[-1]
            _x = self.data['x'] if _ndim != 1 else _numpy.reshape(self.data['x'], (_xlen, 1))
            self._x = _numpy.transpose(_x) if _numpy.array(_x).ndim != 1 else _x
        if 'y' not in self.data.keys():
            raise SwissFitException("'y' data must be specified in 'data' or 'udata'")
        if any(not isinstance(data, _gvar.GVar) for data in self.data['y']):
            raise SwissFitException("all members of data['y'] must be GVars")
        
        # Get inverse square root of covariance matrix & diagonal of covariance matrix for data
        self._data_icovsqrt = _linalg.cov_inv_SVD(
            _gvar.evalcov(self.data['y']),
            square_root = True,
            svdcut = self.data_svdcut
        )
        self._data_mean = _gvar.mean(self.data['y']) # Mean of data
        self._data_sdev = _gvar.sdev(self.data['y']) # Square root of diagonal covariance

        # Get statistical information from priors & check for consistency w/ p0
        if self._prior_specified:
            self.prior_flat = _numpy.array([
                self.prior_transformation_fcn[key](prior)
                if key in self.prior_transformation_fcn.keys() else prior
                for key, prior_list in self.prior.items() for prior in prior_list
            ])

            # Get inverse square root of covariance matrix 
            self._prior_icovsqrt = _linalg.cov_inv_SVD(
                _gvar.evalcov(self.prior_flat),
                square_root = True,
                svdcut = self.prior_svdcut
            )

            # Get diagonal entries of covariance matrix (square root)
            self._prior_sdev = _gvar.sdev(self.prior_flat)

            # Save prior mean
            self._prior_mean = _gvar.mean(self.prior_flat)

            # Fill in missing p0 entries with mean of prior entries
            for key, prior in self.prior.items():
                if key not in self.p0.keys(): self.p0[key] = _gvar.mean(prior)

            # Force ordering on p0 keys to be consistent with ordering on prior keys
            temp_p0 = {}
            for key in self.prior.keys():
                temp_p0[key] = self.p0[key]
            for key in self.p0.keys():
                if key not in temp_p0.keys():
                    temp_p0[key] = self.p0[key]
            self.p0 = temp_p0

        # Create dictionary of dictionary array lengths
        self.pflat, self._plengths, _length = [], {}, 0
        for key, items in self.p0.items():
            for item in items: self.pflat.append(item)
            self._plengths[key] = [_length, _length + len(items)]
            _length += len(items)
        self.pmean = _numpy.copy(self.pflat)
            
        # Prepare dictionary of parameters
        self._pdict = {
            key: self.pflat[self._plengths[key][0]:self._plengths[key][-1]]
            for key in self.p0.keys()
        }

        # Initialize tracker data structure
        self._trackerp = _numpy.array([*map(_gvar.gvar, self.pflat)])

        # Calculate prefactor for PDF
        self._logpdf_prefac = (2. * _numpy.pi)**len(self.data['y'])
        if self._prior_specified:
            self._logpdf_prefac *= (2. * _numpy.pi)**len(self.prior_flat)
        self._logpdf_prefac *= _det(_gvar.evalcov(self.data['y']))
        if self._prior_specified:
            self._logpdf_prefac *= _det(_gvar.evalcov(self.prior_flat))
        self._logpdf_prefac = 1. / _numpy.sqrt(self._logpdf_prefac)
        self._logpdf_prefac = _numpy.log(self._logpdf_prefac)
        
    def _prepare_fit_fcns(self):
        if self._correlated_data: self.data_residual = self._correlated_data_residual
        else: self.data_residual = self._uncorrelated_data_residual
        if self._prior_specified:
            if self._correlated_prior: self.prior_residual = self._correlated_prior_residual
            else: self.prior_residual = self._uncorrelated_prior_residual
        
    def _return_fit_result(self, p):
        if self._x_specified: return self.fit_fcn(*self._x, p)
        else: return self.fit_fcn(p)

    def _correlated_data_residual(self, p):
        return self._data_icovsqrt @ (self._return_fit_result(p) - self._data_mean)

    def _uncorrelated_data_residual(self, p):
        return (self._return_fit_result(p) - self._data_mean) / self._data_sdev
    
    def _return_prior_result(self, p):
        return _numpy.array([
            self.prior_transformation_fcn[pkey](p)
            if pkey in self.prior_transformation_fcn.keys() else p
            for pkey, plist in p.items() for p in plist
            if pkey in self.prior.keys()
        ])
        
    def _correlated_prior_residual(self, p):
        return self._prior_icovsqrt @ (self._return_prior_result(p) - self._prior_mean)

    def _uncorrelated_prior_residual(self, p):
        return (self._return_prior_result(p) - self._prior_mean) / self._prior_sdev

    def calculate_residual(self, p, return_residual = True):
        """Calculate residual of augmented chi^2

        Calculates residual "residual", which enters the augmented
        chi^2 "chi_aug^2" as

        chi_aug^2 = sum_i residual_i^2.

        In other words, in terms of the full covariance matrix of 
        the data/priors "cov" and the difference of the fit results 
        between the data/priors "delta", the residual is

        residual = cov^{-1/2} * delta.

        Args
        ----
        p : list/array or dict
            Fit parameters. Will be converted to dictionary if
            specified as a list/array.
        return_residual : bool, optional
            Specify if user wishes to have residual returned explicitly.

        Returns
        ------
        residual : Numpy array
            Numpy array of residuals. Only returned if
            return_residual is True.

        """
        if (not isinstance(p, dict)) and (not isinstance(p, _gvar.BufferDict)):
            p = self.map_keys(p, return_parameters = True)
        residual = self.data_residual(p)
        if self._prior_specified:
            residual = _numpy.concatenate([residual, self.prior_residual(p)])
        if return_residual: return residual
        else: self.residual = residual
    
    def calculate_chi2(self, p, return_chi2 = True):
        """Calculate augmented chi^2
        
        Calculate augmented chi^2 "chi_aug^2" from 
        residual "residual" as
        
        chi_aug^2 = sum_i residual_i^2.

        Args
        ---
        p : list/array or dict
            Fit parameters. Will be converted to dictionary if
            specified as a list/array.
        return_chi2 : bool, optional
            Specify if user wishes to have chi^2 returned explicitly.

        Returns
        ------
        chi2 : GVar or float
            Augmented chi^2

        """
        self.calculate_residual(p, return_residual = False)
        if return_chi2: return _numpy.dot(self.residual, self.residual)
        else: self.chi2 = _numpy.dot(self.residual, self.residual)

    def _jacobian(self):
        return _numpy.array([
            residual.deriv(self._trackerp) for residual in self.calculate_residual(self._trackerp)
        ])

    def calculate_jacobian(self, p, return_jacobian = True):
        """Calculate "Jacobian"

        Calculates "Jacobian" of fit result (fit function & prior) 
        for each "data" point (data & prior) in fit parameters. 
        In other words,

        J_ij = ∂f_i/∂p_j,

        where f_i can be either the fit function or prior result and
        p_j a fit parameter.

        Args
        ---
        p : list/array or dict
            Fit parameters. Will be converted to dictionary if
            specified as a list/array.
        return_jacobian : bool, optional
            Specify if user wishes to have Jacobian returned explicitly.

        Returns
        ------
        jacobian : Numpy array
           Array of Jacobian values J_ij = ∂f_i/∂p_j

        """
        self._create_tracker(p)
        if return_jacobian: return self._jacobian()
        else: self.jacobian = self._jacobian()

    def _gradient(self): return self.calculate_chi2(self._trackerp).deriv(self._trackerp)

    def calculate_gradient(self, p, return_gradient = True):
        """Calculate gradient of chi^2
        
        Calculates the gradient of the augmented chi^2 "chi_aug^2"
        in the fit parameters. In other words,

        g_j = chi^2_aug/∂p_j,

        where p_j is a fit parameter.

        Args
        ---
        p : list/array or dict
            Fit parameters. Will be converted to dictionary if
            specified as a list/array.
        return_gradient : bool, optional
            Specify if user wishes to have gradient returned explicitly.

        Returns
        -------
        gradient : Numpy array
            Augmented chi^2 gradient

        """
        self._create_tracker(p)
        if return_gradient: return self._gradient()
        else: self.gradient = self._gradient()

    def calculate_hessian(self, p, return_hessian = True, approximate_hessian = True):
        """Calculate Hessian matrix
        
        Calculates Hessian matrix of augmented chi^2 in fit parameters. 
        Calculated either approximately

        H_ij = sum_k J_ik * J_kj,

        where J_ik is the "Jacobian" matrix from "calculate_jacobian", or
        "exactly" using finite differences

        H_ij = ∂^2 chi^2_aug / ∂p_i ∂p_j,

        where p_i,p_j are fit parameters. The approximate calculation is
        what one would use for, say, a Gauss-Newton method. It is standard.
        However, it can be inaccurate. If the approximate Hessian looks 
        weird (one can tell from the estimated statistical errors of 
        the fit paramteres), then chances are that you are not actually
        at a minimum or the estimate is poor.

        Args
        ---
        p : list/array or dict
            Fit parameters. Will be converted to dictionary if
            specified as a list/array.
        return_hessian : bool, optional
            Specify if user wishes to have Hessian returned explicitly.

        Returns
        ------
        Hessian : Numpy array
            Result of Hessian calculation

        """
        if approximate_hessian:
            self.calculate_jacobian(p, return_jacobian = False)
            self.hessian = _numpy.transpose(self.jacobian) @ self.jacobian
        else: self.hessian = 0.5 * _jac(p, self.calculate_gradient)
        if return_hessian: return self.hessian

    def calculate_pdf(self, p, return_pdf = True, **kwargs):
        """Calculate probability distribution function
        
        Calculate probability distribution function for the posterior 
        distribution "posterior"

        posterior ~ likelihood * prior ~ exp(-chi_aug^2 / 2),

        where chi_aug^2 is the "augmented chi^2". The distribution is
        not normalized.

        Args
        ----
        p : list/array or dict
            Fit parameters. Will be converted to dictionary if
            specified as a list/array.
        return_hessian : bool, optional
            Specify if user wishes to have PDF returned explicitly.

        Returns
        -------
        PDF : float
            Posterior PDF at a particular value of the fit parameters p

        """
        self.pdf = _gvar.exp(-0.5 * self.calculate_chi2(p) + self._logpdf_prefac)
        if return_pdf: return self.pdf
        
    # Functions for getting fit results & assigning them to properties

    # chi2 of the data
    def _chi2_data(self):
        residual = self.data_residual(self.map_keys(self.pmean, return_parameters = True))
        return _numpy.dot(residual, residual)
    
    # chi2 of the prior
    def _chi2_prior(self):
        residual = self.prior_residual(self.map_keys(self.pmean, return_parameters = True))
        return _numpy.dot(residual, residual)
        
    # Augmented chi2
    def _chi2(self): return self.calculate_chi2(self.pmean, return_chi2 = True)

    # "Frequentist dof"
    def _frequentist_dof(self): return len(self.data['y']) - len(self.pmean)
    
    # Bayesian count on # of degrees of freedom
    def _dof(self):
        if self._prior_specified: return self._frequentist_dof() + len(self.prior_flat)
        else: return self._frequentist_dof()

    # Bayesian p-value
    def _Q(self): return _gammaincc(0.5 * self._dof(), 0.5 * self._chi2())

    # log(ML), where ML = "marginal likelihood"
    def _logml(self):
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            # Create buffer
            if not self._prior_specified: buf = (_numpy.array(self.data['y']).flat[:])
            else:
                buf = (_numpy.array(self.data['y']).flat, _numpy.array(self.prior_flat).flat)
                buf = (_numpy.concatenate(buf))

            # Calculate log of marginal likelihood (Laplace approx.) & return
            logml = self._chi2()
            logml += self._dof() * _numpy.log(2. * _numpy.pi)
            logml += _linalg.logdet(self.calculate_hessian(self.pmean, return_hessian = True))
            logml += _linalg.logdet(_gvar.evalcov(buf))
            logml *= -0.5
            return logml
        elif self._estimator.method == 'MCMC': # Direct MCMC estimate
            match self._estimator.tag:
                case 'vegas_peter_lepage':
                    return _numpy.log(self._estimator.p.pdfnorm)

    def _aic(self): return self._chi2() + 2. * len(self.pmean)
    
    def _cov(self):
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            return _linalg.pinv(self.calculate_hessian(
                self.pmean, return_hessian = True, approximate_hessian = self._approx_cov
            ))
        elif (self._estimator.method == 'MCMC'):
            return _gvar.evalcov([p for pkey, ps in self._estimator.p.items() for p in ps])
    
    def _p(self): # arXiv:1406.2279 & github.com/gplepage/lsqfit
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            # Create buffer
            if not self._prior_specified: buf = (_numpy.array(self.data['y']).flat[:])
            else:
                buf = (_numpy.array(self.data['y']).flat, _numpy.array(self.prior_flat).flat)
                buf = (_numpy.concatenate(buf))
                
            # Calculate dp/dy ****
            dfdp = _numpy.transpose(self.calculate_jacobian(self.pmean)) # df/dp
            dcov = _linalg.cov_inv_SVD(_gvar.evalcov(buf), square_root = True) # Data covariance
            dpdy = self._cov() @ dfdp @ dcov

            # Collect parameters & return
            p = []
            for pv,drv in zip(self.pmean, dpdy):
                p.append(_gvar.gvar(pv, _gvar.wsum_der(drv, buf), buf[0].cov))
            return self.map_keys(p, return_parameters = True).copy()
        elif self._estimator.method == 'MCMC':
            match self._estimator.tag:
                case 'vegas_peter_lepage': return self._estimator.p
    
    # Define calls to these functions as SwissFit properties
    chi2_data = property(_chi2_data)
    chi2_prior = property(_chi2_prior)
    chi2 = property(_chi2)
    Q = property(_Q)
    dof = property(_dof)
    frequentist_dof = property(_frequentist_dof)
    logml = property(_logml)
    aic = property(_aic)
    cov = property(_cov)
    p = property(_p)
    
    # Printout when called as string
    def __str__(self):
        # Fit title
        out = ''
        lbr = 3 * ' '
        out = '\nSwissFit: ' + '\U0001f9c0\n'

        dof = self._dof()
        if dof != 0:
            chi2 = self._chi2()
            Q = self._Q()
            out += lbr + 'chi2/dof [dof] = ' + str(round(chi2/dof, 2))
            out += ' [' + str(self.dof) + ']'
            out += lbr + 'Q = ' + str(round(Q, 2)) + lbr + '(Bayes) \n'
                
            # Frequentist chi2/dof
            freq_dof = self._frequentist_dof()
            freq_chi2 = self._chi2_data()
            if freq_dof > 0:
                out += lbr + 'chi2/dof [dof] = ' + str(round(freq_chi2/freq_dof, 2))
                out += ' [' + str(freq_dof) + ']'
                out += lbr + 'Q = ' + str(round(
                    _gammaincc(0.5 * freq_dof, 0.5 * freq_chi2), 2
                )) + lbr + '(freq.) \n'

        # AIC & marginal likelihood
        out += lbr + 'AIC [k] = ' + str(round(self._aic(), 2))
        out += '* [' + str(len(self.pmean)) + ']'
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            out += lbr + 'logML = ' + str(round(self._logml(), 3)) + '**\n'
        elif (self._estimator.method == 'MCMC'):
            out += lbr + 'logML = ' + str(self._logml()) + '\n'
            
        # Get ready to show parameters
        out += '\n' + 'Parameters'
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            out += '**'
        out += ':\n'

        # Fit parameters
        pcounter = 0
        cov = self._cov()
        for pname in self.p0.keys():
            out += 5 * ' ' + pname + '\n'
            for pind, pval in enumerate(self.p0[pname]):
                mean = self.pmean[pcounter]
                sig = _numpy.sqrt(cov[pcounter][pcounter])
                pvalg = str(_gvar.gvar(mean, sig))
                out += 13 * ' '
                if pname in self.prior.keys():
                    prg = str(self.prior[pname][pind])
                    out += '%-10s   %15s   [%8s]' % (str(pind + 1), pvalg, prg)
                else:
                    prg = '0(inf)'
                    out += '%-10s   %15s   [n/a]' % (str(pind + 1), pvalg)
                out += '\n'
                pcounter += 1

        # Fit footer
        if hasattr(self._estimator, '__str__'):
            out += '\n' + 'Estimator:\n'
            out += str(self._estimator)
        out += '\n' + '*chi^2 + 2 X "# parameters"\n'
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            out += '**Laplace approximation\n'

        # Return out string
        return out
