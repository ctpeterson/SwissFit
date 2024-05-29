class ModelAveraging(Fitter):
    """ Average model model parameters with SwissFit

    Takes in dictionary of models & averages over their parameters
    using Bayesian model averaging

    Attributes
    ----------

    Methods
    -------
    
    """
    def __init__(self,
                 models = None, # Dictionary of SwissFit fitter objects
                 prior = None, # Model probability for each member of "models"
                 p0 = None,
                 ):
        """ ModelAveraging constructor method
        
        Parameters
        ----------
        models : dict
            Dictionary of SwissFit fitter objects
        model_prior : dict
            Dictionary of model probabilities. If not specified, 
            all models are assumed to have the same prior probabilities.

        """

        # Initialize parent class for fitter
        super().__init__("ModelAveraging")

        # Prepare models
        self._prepare_models(models)
        
        # Prepare parameters
        self._prepare_parameters(prior, p0)

    # Call method
    def __call__(self,
             estimator,
             p0 = None,
             p = None,
             ):
        tags = [estimator.tag, estimator.local_optimizer]
        if any(tag == 'scipy_least_squares' for tag in tags):
            raise SwissFitException(
                "SciPyLeastSquares not supported for ModelAveraging. Try SciPyMinimize."
            )
        else: self.call(estimator, p0 = p0, p = p, approx_cov = True)
        
    def _prepare_models(self, models):
        if (models is None) or (not isinstance(models, dict)):
            raise SwissFitException("Must specify 'models' as dictionary")
        self.models = models

    def _prepare_parameters(self, prior, p0):
        # Prepare priors
        if prior is None:
            # Normalize/specify priors by 1 / "number of models"
            norm = len(list(self.models.keys()))
            self.prior = {key: 1. / norm for key in self.models.keys()}
        else:
            # Check to make sure priors specified as dictionary
            if not isinstance(prior, dict):
                raise SwissFitException("Prior must be specified as dictionary")
            
            # Check to make sure models & priors are consistent
            if len(list(self.models.keys())) != len(list(prior.keys())):
                raise SwissFitException("Mismatch between number of models & priors")

            # Normalize & specify priors
            norm = sum(p for p, pkey in prior.items())
            self.prior = {key: prior / norm for prior, key in prior.items()}

        # Prepare model parameters
        self._pdict = {}
        for model_key, model in self.models.items():
            for pkey in model.p0:
                if pkey not in self._pdict.keys():
                    self._pdict[pkey] = [0. for p in model.p0[pkey]]
        self.pflat, self._plengths, _length = [], {}, 0
        for key, items in self._pdict.items():
            for item in items: self.pflat.append(item)
            self._plengths[key] = [_length, _length + len(items)]
            _length += len(items)
        self.p0 = self._pdict if p0 is None else p0
            
    def _filter_pdf_args(self, p, model):
        return {key: p[key] for key in self.models[model].p0.keys()}
    
    def calculate_pdf(self, p, return_pdf = True):
        """ Calculate probability distribution function for model average

        """
        self.pdf = sum(
            model.calculate_pdf(self._filter_pdf_args(p, model_key)) * self.prior[model_key]
            for model_key, model in self.models.items()
        )
        if return_pdf: return self.pdf

    def calculate_chi2(self, p, return_chi2 = True):
        """ Calculate "chi2" (-2logPDF)
        
        """
        # Map keys to dictionary
        if (not isinstance(p, dict)) and (not isinstance(p, _gvar.BufferDict)):
            p = self.map_keys(p, return_parameters = True)

        # Calculate chi2 & return if requested
        try: chi2 = -2. * _gvar.log(self.calculate_pdf(p))
        except ZeroDivisionError: chi2 = -2. * _gvar.log(self.calculate_pdf(p) + 1e-32)
        if return_chi2: return chi2

    def _gradient(self): return self.calculate_chi2(self._trackerp).deriv(self._trackerp)
        
    def calculate_gradient(self, p, return_gradient = True):
        """ Calculate gradient of chi^2

        """
        # Create "tracker"
        self._create_tracker(p)

        # Calculate & optionally return gradient
        self.gradient = self._gradient()
        if return_gradient:
            return self.gradient

    def calculate_hessian(self, p, return_hessian = True, approximate_hessian = True):
        """ Calculate Hessian matrix
        
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
        # Hessian from finite difference in Jacobian
        self.hessian = 0.5 * _jac(p, self.calculate_gradient)

        # Return Hessian if requested
        if return_hessian: return self.hessian

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
    def _chi2(self): return self.calculate_chi2(self.pmean)

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
    
    def _p(self): # arXiv:1406.2279
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            # Create buffer
            if not self._prior_specified: buf = (_numpy.array(self.data['y']).flat[:])
            else:
                buf = (_numpy.array(self.data['y']).flat, _numpy.array(self.prior_flat).flat)
                buf = (_numpy.concatenate(buf))
            
            # Calculate components of dp/dy
            pcov = self._cov() # Parameter covariance
            dfdp = _numpy.transpose(self.calculate_jacobian(self.pmean)) # df/dp
            dcov = _linalg.cov_inv_SVD(_gvar.evalcov(buf), square_root = True) # Data covariance

            # Calculate dp/dy
            dpdy = pcov @ dfdp @ dcov

            # Collect fit parameters into GVars
            p = []
            for index in range(dpdy.shape[0]):
                mean = self.pmean[index]
                deriv = _gvar.wsum_der(dpdy[index], buf)
                buffer_cov = buf[0].cov
                p.append(_gvar.gvar(mean, deriv, buffer_cov))

            # Return p as a diectionary
            return self.map_keys(p, return_parameters = True)
        elif self._estimator.method == 'MCMC':
            match self._estimator.tag:
                case 'vegas_peter_lepage': return self._estimator.p
        
    # Define calls to these functions as SwissFit properties
    chi2 = property(_chi2)
    #Q = property(_Q)
    #dof = property(_dof)
    #frequentist_dof = property(_frequentist_dof)
    #logml = property(_logml)
    #aic = property(_aic)
    cov = property(_cov)
    p = property(_p)
    
    # Printout when called as string

    def __str__(self):
        # Fit title
        out = ''
        lbr = 3 * ' '
        out = '\nSwissFit: ' + '\U0001f9c0\n'

        """
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
        out += ' [' + str(len(self.pmean)) + ']'
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            out += lbr + 'logML = ' + str(round(self._logml(), 3)) + '*\n'
        elif (self._estimator.method == 'MCMC'):
            out += lbr + 'logML = ' + str(self._logml()) + '\n'
        """

        # Get ready to show parameters
        out += '\n' + 'Parameters'
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            out += '*'
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
        if (self._estimator.method == 'MAP') or (self._estimator.method == 'none'):
            out += '\n' + '*Laplace approximation\n'

        # Return out string
        return out
