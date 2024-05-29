# Import global & local modules
from .numerical_tools import linalg as _linalg # Useful linear algebra tools
import gvar as _gvar # For automatic propagation of correlated errors
import numpy as _numpy # For vectorized numerical operations
from scipy.special import gammaincc as _gammaincc # For calculating p-value of fit chi^2
from functools import partial as _partial # For partial evaluation of functions
from scipy.linalg import block_diag as _block_diag # For getting block diagonal matrix
from scipy.optimize import approx_fprime as _jac # For calculating Hessian
from functools import reduce as _reduce # For cleaner matrix calculations

# First version of SwissFit class
class SwissFitV1(object):
    def __init__(self,
                 # Lsqfit-style arguments
                 data = {'x': {}, 'y': {}}, # Input/output data
                 udata = None, # Uncorrelated input/output data
                 prior = None, # Priors
                 uprior = None, # Uncorrelated priors
                 p0 = {}, # Starting values for fit
                 fit_fcn = None, # Function to be fit
                 prior_fcn = None, # Function for priors

                 # Function for passing priors through
                 prior_transformation_fcn = {},
                 
                 # Parameters for data processing
                 x_normalization = None,
                 y_normalization = None,

                 # Function for calculating log of the prior
                 calculate_logPrior = None,
                 
                 # Parameters for gradient clipping
                 clip_gradient = False, # If True, gradient clipping will be applied
                 gradient_tolerance = 1e6, # Maximum size of gradient if clip_gradient is True
                 custom_gradient_clipping = None, # Optional user-provided grad. clip. function

                 # Default SVD cut on data
                 data_svdcut = None
                 ):
        """ Save input specifications """
        # Function & prior function
        self.fit_fcn = fit_fcn; self.prior_fcn = prior_fcn;

        # Check if prior has been specified
        if (prior is None) and (uprior is None):
            prior = {}; self._prior_specified = False; 
        else: self._prior_specified = True
        self.prior_specified = self._prior_specified
        
        # Set function for calculating log of prior
        self.calculate_logPrior = calculate_logPrior
        if self.calculate_logPrior is None: self.logPr = 0.

        # Check if prior and/or data specified as "uncorrelated"
        if udata is None: self._correlated_data = True
        else: self._correlated_data = False
        if uprior is None: self._correlated_prior = True
        else: self._correlated_prior = False
        
        # Data, prior, and starting value
        self.data = data if udata is None else udata
        self.prior = prior if uprior is None else uprior
        self.p0 = p0

        # Optional gradient clipping; if clip_grad = False, then is the identity
        self._grad_clip = {
            True: lambda x: gradient_tolerance * x / _numpy.linalg.norm(x)
            if any(_numpy.abs(xx) > gradient_tolerance for xx in x) else x,
            False: lambda x: x
        }[clip_gradient] if custom_gradient_clipping is None else custom_gradient_clipping

        # Prepare data, priors & starting parameters for fit
        self._prep_data(
            x_normalization,
            y_normalization,
            data_svdcut,
            prior_transformation_fcn
        )

    """ 
    Tools & setup:

    - _create_pgv (protected): Creates GVar variable for the mean of each fit parameter
      so that gradient and Jacobian can be calculated by automatic differentiation
    - _prep_data (protected): Prepares data and priors for any fit object that they're used for
    - map_keys (public): Converts array of parameters to dictionary of parameters so that
      they can be easily access by user's provided functions
    """
    # Create GVar instance of variables for backprop
    def _create_pgv(self, p):
        self._pgv = _numpy.array([*map(_gvar.gvar, p)])

    # Normalize input/output data
    def _normalize(self, data, normalization):
        if normalization is None:
            if _numpy.array(data).ndim != 1: return _numpy.transpose(data)
            else: return data
        if _numpy.array(data).ndim != 1: data_series = _numpy.transpose(data)
        else: data_series = [data]
        new_data_series = []
        for dt_ind, dt in enumerate(data_series):
            match normalization[dt_ind]:
                case 'min': dt /= min(dt)
                case 'max': dt /= max(dt)
                case 'max/min':
                    minimum, maximum = min(dt), max(dt)
                    dt = (2. * dt - maximum - minimum ) / (maximum - minimum)
                case 'max-min': dt /= (max(dt) - min(dt))
                case 'standardize':
                    mean = sum(dt) / len(dt)
                    sdev = _gvar.sqrt(sum((dt - mean)**2.) / (len(dt) - 1.))
                    dt = (dt - mean) / sdev
                case None: dt = dt
                case _: dt = dt
            new_data_series.append(dt)
        if _numpy.array(data).ndim != 1: return new_data_series
        else: return new_data_series[0]
        
    # Prepare data
    def _prep_data(self, x_normalization, y_normalization,
                   svdcut, prior_transformation_fcn):
        """ Preprocess input/output data """
        # Preprocess inputs
        if 'x' in self.data.keys():
            ndim, xlen = _numpy.array(self.data['x']).ndim, _numpy.array(self.data['x']).shape[-1]
            self._x = self._normalize(
                self.data['x'] if ndim != 1 else _numpy.reshape(self.data['x'], (xlen, 1)),
                x_normalization
            )
        if 'x' in self.data.keys(): self._x_specified = True
        else: self._x_specified = False
            
        # Preprocess outputs
        y = self._normalize(self.data['y'], y_normalization); self._ymean = _gvar.mean(y);
        if not self._correlated_data: self._ysdev = _gvar.sdev(y)
        
        # Get covariance of data & diagonal entries of covariance matrix
        self._icovsqrt = _linalg.cov_inv_SVD(
            _gvar.evalcov(y), square_root = True, svdcut = svdcut
        ); self._data_sdev = _gvar.sdev(y);

        """ Check if priors are hierarchical """
        if 'I' not in self.prior.keys():
            self.prior = {'I': self.prior}
            self._hierarchical = False
        else: self._hierarchical = True
        if 'I' not in self.p0.keys(): self.p0 = {'I': self.p0}
        
        """ Set prior function up """
        # Fix if priors are not hierarchical
        if not self._hierarchical:
            prior_transformation_fcn['I'] = prior_transformation_fcn
        
        # Transform transformation function
        for level in self.prior.keys():
            if level not in prior_transformation_fcn.keys():
                prior_transformation_fcn[level] = {}
        
        # Transformation of priors
        self._prior_transformation_fcn = {}
        for level in self.prior.keys():
            self._prior_transformation_fcn[level] = {}
            for key in self.prior[level].keys():
                if key in prior_transformation_fcn[level].keys():
                    self._prior_transformation_fcn[level][key] = prior_transformation_fcn[level][key]
                else: self._prior_transformation_fcn[level][key] = lambda x: x

        """ Preprocess priors & starting values for fit parameters """
        # Correct prior & p0 if not specified in standard format
        if 'x' in self.data.keys():
            if not self._hierarchical:
                self._wrap_fcn = lambda *args, p: self.fit_fcn(*args, p = p['I'])
            else: self._wrap_fcn = lambda *args, p: self.fit_fcn(*args, p = p)
        else:
            if not self._hierarchical: self._wrap_fcn = lambda p: self.fit_fcn(p['I'])
            else: self._wrap_fcn = lambda p: self.fit_fcn(p)
            
        # Process prior & p0
        self._p0 = []; self._p = {}; self._lngths = {};
        self._prior = {}; self._prior_flat = []; size = 0;
        for level in self.prior.keys():
            if level not in self.p0.keys(): self.p0[level] = {}
            self._lngths[level] = {}; self._p[level] = {};
            self._prior[level] = {}
            for p0_key in self.p0[level].keys():
                self._p[level][p0_key] = self.p0[level][p0_key]
                self._lngths[level][p0_key] = [size, size + len(self.p0[level][p0_key])]
                size += len(self.p0[level][p0_key])
                for p0 in self.p0[level][p0_key]: self._p0.append(p0)
                self.p0[level][p0_key] = _numpy.array(self.p0[level][p0_key])
            for prior_key in self.prior[level].keys():
                if prior_key not in self.p0[level].keys():
                    self.p0[level][prior_key] = _gvar.mean(self.prior[level][prior_key])
                    self._p[level][prior_key] = self.p0[level][prior_key]
                    self._lngths[level][prior_key] = [size, size + len(self.p0[level][prior_key])]
                    size += len(self.p0[level][prior_key])
                    for p0 in self.p0[level][prior_key]: self._p0.append(p0)
                    self.p0[level][prior_key] = _numpy.array(self.p0[level][prior_key])
                self._prior[level][prior_key] = self.prior[level][prior_key]
                for prv in self._prior[level][prior_key]:
                    self._prior_flat.append(
                        self._prior_transformation_fcn[level][prior_key](prv)
                    )

        # Get Gaussian properties of variables
        if self.prior_fcn is None:
            self._prior_icovsqrt = _linalg.cov_inv_SVD(
                _gvar.evalcov(self._prior_flat),
                square_root = True
            ); self.prior_fcn = {'I': self._prior_function};
        self._prior_mean = _gvar.mean(self._prior_flat)
        self._iprior_sdev = 1. / _gvar.sdev(self._prior_flat)
        self.prior_flat = self._prior_flat
        
        """ Create space for quantities used throughout fitting process """
        # Create space for quantities that may be saved during use of object
        self.residual = _numpy.zeros(len(self._p0)); self.gradient = _numpy.zeros(len(self._p0));
        self.chi2 = 0.; self.jacobian = _numpy.zeros((len(self._ymean), len(self._p0)));
        self._pgv = [_gvar.gvar(p0, 0.) for p0 in self._p0]
        
        # Save pmean
        self.pmean = self._p0

    # Map keys
    def map_keys(self, p, return_parameters = False):
        for lvl in self.p0.keys():
            for ky in self.p0[lvl].keys():
                self._p[lvl][ky] = p[self._lngths[lvl][ky][0]:self._lngths[lvl][ky][-1]]
        if return_parameters: return self._p

    """ Prior function """
    def _prior_function(self, p):
        return {
            'icovroot': self._prior_icovsqrt if self._correlated_prior else self._iprior_sdev,
            'diff': _numpy.array(
                [
                    self._prior_transformation_fcn[self._level][key](prv)
                    for key in p['I'].keys() for prv in p['I'][key]
                    if key in self.prior['I'].keys()
                ]
            ) - self._prior_mean
        }
                
    """ 
    Calculate chi^2:

    1.) diff = <fit/parameters> - <data/priors>
    2.) residual = cov^{-1/2} * diff
    3.) chi^2 = residual^T * residual = diff^T * cov^{-1} * diff

    - Square root of inverse covariance matrix to be understood in terms of
      the singular value decomposition of the covariance matrix. See the notes
      at the top of swissfit.numerical_tools.linalg.cov_inv_SVD for more details.
    - The covariance matrix here is to be understood as block diagonal in the
      chi^2 of the data and the chi^2 of the priors for each hierarchical level
      of the priors
    """

    # Residual from data
    def data_residual(self, resid = None):
        # Get result from model
        if self._x_specified: self._ymodel = self._wrap_fcn(*self._x, p = self._p)
        else: self._ymodel = self._wrap_fcn(self._p)

        # Inverse covariance matrix contribution
        resid = [] if resid is None else resid
        if self._correlated_data:
            resid += list(_numpy.matmul(self._icovsqrt, self._ymodel - self._ymean))
        else: resid += list( (self._ymodel - self._ymean) / self._ysdev )
        
        # Return residual from data
        return resid
        
    # Residual from prior
    def prior_residual(self, resid = None):
        # Get priors for each level of model
        resid = [] if resid is None else resid
        if self._prior_specified:
            for level in self.prior.keys():
                # Get output from prior function
                self._level = level
                prior = self.prior_fcn[level](self._p)

                # Get residual from prior
                if self._correlated_prior:
                    resid += list(_numpy.matmul(prior['icovroot'], prior['diff']))
                else: resid += list(prior['diff'] * prior['icovroot'])
                
        # Return residual from prior
        return resid
    
    # Residuals in uncorrelated case
    def calculate_residual(self, p, return_residual = True):
        # Check if keys need to be mapped
        if not isinstance(p, dict): self.map_keys(p)
        
        # Calculate residuals from data & prior
        resid = self.prior_residual(resid = self.data_residual())
                
        # Return residual
        if return_residual: return resid
        else: self.residual = resid
        
    # Calculate augmented chi^2
    def calculate_chi2(self, p, return_chi2 = True):
        # Check if keys need to be mapped
        if not isinstance(p, dict): p = self.map_keys(p, return_parameters = True)
        
        # Calculate residual
        self.calculate_residual(p, return_residual = False)

        # Include log of prior if requested
        if self.calculate_logPrior is not None:
            self.logPr = self.calculate_logPrior(p)

        # Return/save chi^2
        if return_chi2: return _numpy.dot(self.residual, self.residual) + self.logPr
        else: self.chi2 = _numpy.dot(self.residual, self.residual) + self.logPr

    """ 
    Calculate derivatives:

    Calculates various derivatives that are used in fits & assessing both
    quality of fit & model selection criteria. Calculation of derivatives
    supported by GVar's automatic differentiation capabilities.

    + We define the "Jacobian" J_ij as J_ij = ∂f_i/∂p_j, where f_i is the
      "ith" observation of the data and p_j is the "jth" model parameter. 
      Calling this a "Jacobian" might confuse some folks because 
      it's not what many of us are used to calling a Jacobian, but 
      this is standard language.
    + The gradient is calculated ∂chi^2/∂p_j, where "chi^2" is 
      the augmented chi^2. 

    - calculate_jacobian, _jacobian (public, private): "calculate_jacobian"
      prepares calculation of ∂f_i/∂p_j by creating GVar variables out
      of p_j means, so that numerical operations can be cached for for 
      automatic differentiation
      Then, "calculate_jacobian" calls "_jacobian" to calculate the Jacobian.
    - calculate_gradient, _gradient (public, private): Same as "calculate_jacobian"
      and "_jacobian" pair, but calculates the gradient of the chi^2 with 
      respect to p_j; that is, ∂chi^2/∂p_j
    """
    
    # "Jacobian" (deriv. of each output w.r.t. parameters - uses autodiff.)
    def _jacobian(self): return _numpy.array([
            self._grad_clip(fwd.deriv(self._pgv))
            for fwd in self.calculate_residual(self._pgv)
    ])

    # Prepare & calculate jacobian (∂f_i/∂p_j)
    def calculate_jacobian(self, p, return_jacobian = True):
        self._create_pgv(p)
        if return_jacobian: return self._jacobian()
        else: self.jacobian = self._jacobian()
        
    # Gradient of chi^2 with respect to parameters
    def _gradient(self):
        return self._grad_clip(
            self.calculate_chi2(self._pgv).deriv(self._pgv)
        )
    
    # Prepare & calculate gradient (∂chi^2/∂p_j)
    def calculate_gradient(self, p, return_gradient = True):
        self._create_pgv(p)
        if return_gradient: return self._gradient()
        else: self.gradient = self._gradient()

    # Approximate calculation of Hessian (J*J^T ~ [H_ij] = [∂^2chi^2/∂p_i∂p_j])
    def calculate_hessian(self, p, return_hessian = True, approximate_hessian = True):
        # Calculate Hessian
        if approximate_hessian:
            self.calculate_jacobian(p, return_jacobian = False)
            self.hessian = _numpy.matmul(_numpy.transpose(self.jacobian), self.jacobian)
        else: self.hessian = 0.5 * _jac(p, self.calculate_gradient)

        # Return Hessian
        if return_hessian: return self.hessian

    """
    Get fit parameters:

    Fetch fit prediction & transform into GVar variables with appropriate
    correlations & errors as predicted from the Laplace approximation.
    Based on information from [arXiv:1406.2279] and the Lsqfit source code
    (see "_getp" in https://github.com/gplepage/lsqfit)
    """
        
    # Called when "SwissFit.p" called to grab fit parameters
    def _getp(self):
        # Check estimation method
        if (self.estimation_method == 'map') or (self.estimation_method == 'none'):
            # Calculate dp/dy from Eq. A6 of [arXiv:1406.2279]
            dpdy = _numpy.matmul(
                self._parameter_covariance, # Sigma_Theta
                _numpy.matmul(
                    _numpy.transpose(
                        self.calculate_jacobian(self.pmean)
                    ), # (Sigma_{O,P}^{-1/2} * df/dp)^T
                    _linalg.cov_inv_SVD(
                        _gvar.evalcov(self._buf),
                        square_root = True
                    ) # Sigma_{O,P}^{-1/2}
                )
            ) # = Sigma_Theta * df/dp * Sigma_{O,P}^{-1} = dp/dy

            # Return GVars
            p = []
            for index in range(dpdy.shape[0]):
                p.append(
                    _gvar.gvar(
                        self.pmean[index],
                        _gvar.wsum_der(dpdy[index], self._buf),
                        self._buf[0].cov)
                )

            # Return new gvars
            if self._hierarchical: return self.map_keys(p, return_parameters = True)
            else: return self.map_keys(p, return_parameters = True)['I']
        elif (self.estimation_method == 'mcmc'): return self.fit.x
            
    # Calling fit.p will invoke _getp
    p = property(_getp)
        
    """ 
    Quality of fit, model selection criteria, etc.:
    
    - _Q (protected): Calculate p-value from number of degrees
      of freedom & chi^2 (augmented chi^2 or any component of it)
    """
    
    # p-value
    def _Q(self, chi2 = None, dof = None, return_pvalue = True):
        chi2 = self.chi2 if chi2 is None else chi2
        dof = self.dof if dof is None else dof
        if return_pvalue: return _gammaincc(0.5 * dof, 0.5 * chi2)
        else: self.Q = _gammaincc(0.5 * dof, 0.5 * _gvar.mean(chi2))

    # Calculate log likelihood
    def _log_likelihood(self, p):
        self.calculate_residual(p, return_residual = False)
        return -0.5 * _numpy.dot(
            self.residual[:len(self._ymean)],
            self.residual[:len(self._ymean)]
        )

    # Calculate marginal likelihood
    def _log_marginal_likelihood(self, p):
        """
        Laplace approximation of the marginal likelihood:
        -2 * logML = chi^2_aug + log ((2*pi)^{dof} * detSigma_{O,P} / detSigma_Theta)
        """
        
        # Create buffer (reused in fit.p)
        self._buf = (
            _numpy.array(self.data['y']).flat[:] if not self._prior_specified else
            _numpy.concatenate(
                (_numpy.array(self.data['y']).flat, _numpy.array(self._prior_flat).flat)
            )
        )

        # Calculate log determinants
        logdet_SigmaTheta = -_linalg.logdet(
            self.calculate_hessian(
                self.pmean, return_hessian = True
            )
        )
        logdet_SigmaOP = _linalg.logdet(_gvar.evalcov(self._buf))

        # Calculate Laplace approximation of logML
        self.logml = self.chi2 + self.dof * _numpy.log(2. * _numpy.pi)
        self.logml += logdet_SigmaOP - logdet_SigmaTheta
        self.logml *= -0.5

    # Expectation value of chi^2 [arXiv:2209.14188]
    def _chi2exp(self):
        """
        Calculation of expectation value of chi^2 from Laplace approximation.
        Calculation outlined in "On fits to correlated and auto-correlated data"
        by Rainer Sommer and Mattia Bruno [arXiv:2209.14188]
        """

        # Calculate projector
        hessian_inverse = _linalg.pinv(self.calculate_hessian(self.pmean))
        jacobian = self.calculate_jacobian(self.pmean)
        P = _numpy.matmul(
            jacobian,
            _numpy.matmul(
                hessian_inverse,
                _numpy.transpose(jacobian)
            )
        )

        # Get "weights"
        W = [
            self._icovsqrt if self._correlated_data
            else _numpy.diag(1. / _gvar.sdev(self.data['y']))
        ]
        for level in self.prior.keys():
            self._level = level
            if self._correlated_prior: W += [self.prior_fcn[level](self._pmean)['icovroot']]
            else: W += [_numpy.diag(self.prior_fcn[level](self._pmean)['icovroot'])]
        W = _numpy.linalg.block_diag(*W)

        # Calculate Cw
        Cw = _numpy.matmul(W, _numpy.matmul(_gvar.evalcov(self._buf), W))

        # Calculate <chi^2>
        self.chi2exp = _numpy.trace(
            _numpy.matmul(Cw, _numpy.identity(_numpy.shape(P)[0]) - P)
        )

        # "matrix dof"
        #self._matrix_dof = 

    # Modification of p-value that also works for uncorrelated fit [arXiv:2209.14188]
    def _Qalt(self):
       """
        Calculation p-value that has meaning for "uncorrelated" fits. 
       Outlined in "On fits to correlated and auto-correlated data"
        by Rainer Sommer and Mattia Bruno [arXiv:2209.14188]
        """ 
        
    """ 
    Do parameter estimation & collect results:

    - If "estimation_method" is "map" ("maximum a posteriori" = "MAP"):
      Optimizer object passed through call method. Fit performed according to specifications
      of fit object. Fit parameters with appropriate covariance calculated from result
      of fit via Laplace approximation. chi^2, dof & p-value calculated directly from fit.
      AIC calculated from log likelihood function & log of Bayes factor (marginal likelihood)
      calculate via result according to Laplace approximation.
    """
    
    # Call method - performs fit & wraps things up
    def __call__(self,
                 estimator,
                 p0 = None,
                 estimation_method = 'map',
                 approximate_parameter_covariance = True
                 ):
        """ Methods for estimating model parameters (MAP, MCMC, etc...) """
        # Save estimation method
        self.estimation_method = estimation_method
        
        # Do fit by calling SwissFit optimizer
        self._estimator_object = estimator
        if estimator is None: self.pmean = self._p0 if p0 is None else p0
        elif (estimation_method == 'map') or (estimation_method == 'mcmc'):
            self.fit = estimator(
                self._p0 if p0 is None else p0
            ); self.pmean = _gvar.mean(self.fit.x);
        else: # Tell user that estimation method is invalid
            print('"' + estimation_method + '"', 'is an invalid option. Exiting.'); exit();

        """ Finish up calculations """
        # Save results of fit & prepare them for output
        self._finish_results(approximate_parameter_covariance)

    def _finish_results(self, approximate_parameter_covariance):
        # Calculate covariance of fit parameters
        if (self.estimation_method == 'map') or (self.estimation_method == 'none'):
            self._parameter_covariance = _linalg.pinv(
                self.calculate_hessian(
                    self.pmean, return_hessian = True,
                    approximate_hessian = approximate_parameter_covariance
                )
            )
        elif (self.estimation_method == 'mcmc'):
            self._parameter_covariance = _gvar.evalcov(self.fit.x)
        
        # chi^2 & p-value
        self.chi2 = self.calculate_chi2(self.pmean, return_chi2 = True) # chi^2
        self.dof = len(self.data['y']) - len(self.pmean) + len(self._prior_flat) # dof counting
        self._Q(return_pvalue = False) # p-value

        # AIC & Bayes factor
        k, n = len(self.pmean), len(self.data['y'])
        self.aic = 2. * k + self.chi2
        #self.aic += 2. * (k**2. + k) / (n - k - 1.)
        self._log_marginal_likelihood(self.pmean)
        
        # Prepare out string
        self._prepare_output_string('')
        
    """ Print results of fit """
    # Get output string prepared in case user requests it
    def _prepare_output_string(self, out):
        """ Fit header """
        # Silly title
        self._out = out
        lbr = 3 * ' '
        self._out = '\nSwissFit: ' + '\U0001f9c0\n'

        # Bayesian chi2/dof
        if self.dof != 0:
            self._out += lbr + 'chi2/dof [dof] = ' + str(round(_gvar.mean(self.chi2)/self.dof, 2))
            self._out += ' [' + str(self.dof) + ']'
            self._out += lbr + 'Q = ' + str(round(self.Q, 2)) + lbr + '(Bayes) \n'

            # Frequentist chi2/dof
            freq_dof = len(self.data['y']) - len(self.pmean)
            freq_chi2 = -2. * self._log_likelihood(self.pmean) # "chi2_data"
            self.chi2_data = freq_chi2
            self.chi2_prior = self.chi2 - self.chi2_data
            if freq_dof > 0:
                self._out += lbr + 'chi2/dof [dof] = ' + str(round(freq_chi2/freq_dof, 2))
                self._out += ' [' + str(freq_dof) + ']'
                self._out += lbr + 'Q = ' + str(round(
                    self._Q(chi2 = freq_chi2, dof = freq_dof), 2
                )) + lbr + '(freq.) \n'
                self.frequentist_dof = freq_dof

        # AIC
        self._out += lbr + 'AIC [k] = ' + str(round(self.aic, 2))
        self._out += ' [' + str(len(self.pmean)) + ']'
        self._out += lbr + 'logML = ' + str(round(self.logml, 3)) + '*\n'

        # Options if Monte Carlo performed
        if (self.estimation_method == 'mcmc'):
            self._out += lbr + 'acceptance = ' + str(round(self.fit.acc_frac, 2))
            self._out += ' [' + str(self.fit.nsamples) + ']\n'

        # Get ready to show parameters
        self._out += '\n' + 'Parameters*:\n'

        """ Fit parameters """
        pcounter = 0
        for level in self.prior.keys():
            for pname in self.p0[level].keys():
                self._out += 5 * ' ' + pname + '\n'
                for pind, pval in enumerate(self.p0[level][pname]):
                    pvalg = str(
                        _gvar.gvar(
                            self.pmean[pcounter],
                            _numpy.sqrt(self._parameter_covariance[pcounter][pcounter])
                        )
                    )
                    self._out += 13 * ' '
                    if pname in self.prior[level].keys():
                        prg = str(self.prior[level][pname][pind])
                        self._out += '%-10s   %15s   [%8s]' % (str(pind + 1), pvalg, prg)
                    else:
                        prg = '0(inf)'
                        self._out += '%-10s   %15s   [n/a]' % (str(pind + 1), pvalg)
                    self._out += '\n'
                    pcounter += 1

        """ Summary of optimizer """
        if hasattr(self._estimator_object, '__str__'):
            self._out += '\n' + 'Estimator:\n'
            self._out += str(self._estimator_object)
        
        self._out += '\n' + '*Laplace approximation\n'
        
    # Print out string if user passes fit object through "print"
    def __str__(self): return self._out

    """
    Save other properties of fit:

    1.) Wrapped fit function evaluated at MAP estimate
    """

    """ Fit function evaluated at MAP estimate """
    # Result of fit function
    def _fcn(self, *args):
        if 'x' in self.data.keys(): return self._wrap_fcn(*args, p = self.p)
        p = self.p; p['I']['x'] = [*args];
        return self._wrap_fcn(self.p)

    # Set as property of class
    fcn = property(_fcn)
