import numpy as _numpy # Number crunching
import gvar as _gvar # Gaussian error propagation
import warnings as _warnings # For warning user if something could be a problem

# Class implementing Bayesian model averaging
class BayesianModelAveraging(object):
    def __init__(
            self,
            data = {},
            models = {},
            parameter_locations = {},
    ):
        # Get inputs
        self.parameter_locations = parameter_locations
        self.models = models
        self.data = data

        # Do model-averaging
        self._average()
        
    # For grabbing mean & covariance
    def _average(self):
        # Get weights
        self.weights = {
            model: _numpy.exp(
                -0.5 * fitter.aic + len(self.data['y']) - len(fitter.data['y'])
            )
            for model, fitter in self.models.items()
        } # Unnormalized weights
        Z = sum(weight for model, weight in self.weights.items()) # Normalization factor
        for model in self.weights.keys(): self.weights[model] /= Z # Normalized weights
        weight_array = [weight for weight_key, weight in self.weights.items()]
        
        # Determine if parameters are hierarchical
        if 'I' not in self.parameter_locations.keys(): self._hierarchical = False
        else: self._hierarchical = True

        # Create information to map from array back into dictionary
        self._p = {}; self._lngths = {}; size = 0;
        if self._hierarchical:
            for level in self.parameter_locations.keys():
                self._lngths[level] = {}; self._p[level] = {};
                for key in self.parameter_locations[level].keys():
                    self._p[level][key] = self.parameter_locations[level][key]
                    self._lngths[level][key] = [size, size + len(self._p[level][key])]
                    size += len(self._p[level][key])
        else:
            for key in self.parameter_locations.keys():
                self._p[key] = self.parameter_locations[key]
                self._lngths[key] = [size, size + len(self._p[key])]
                size += len(self._p[key])

        # Check if input data are primary variables
        if all(_gvar.is_primary(self.data['y'])) and all(
                _gvar.is_primary(prior) for model, fitter in self.models.items()
                for prior in fitter.prior_flat): primary = True
        else:
            # Print warning
            _warnings.warn(
                "Some or all of data/priors are not primary GVar variables. " + \
                "Model-averaged parameters will not know about their " + \
                "correlations with either."
            )
            
            # Set primary to false
            primary = False
        
        # Save model properties
        model_property_dictionary = {}
        for model, fitter in self.models.items():
            # Get model parameters as dictionary
            parameters = fitter.p
            model_property_dictionary[model] = {}

            # Convert dictionary to flat array
            if self._hierarchical:
                model_parameter_array = [
                    _numpy.array(parameters[level][key])[index]
                    for level in parameters.keys()
                    for key, indices in self.parameter_locations[level].items()
                    for index in indices
                ]
            else:
                model_parameter_array = [
                    _numpy.array(parameters[key])[index]
                    for key, indices in self.parameter_locations.items()
                    for index in indices
                ]

            # Save mean & covariance of fit parameters
            model_property_dictionary[model] = {
                'means': _gvar.mean(model_parameter_array),
                'covariance_pp': _gvar.evalcov(model_parameter_array)
            }

            # Save covariance of model parameters with data if are primary
            if primary:
                # Create buffer of data with 
                full_dataset = (
                    _numpy.array(self.data['y']).flat[:] if not fitter.prior_specified else
                    _numpy.concatenate(
                        (_numpy.array(self.data['y']).flat, _numpy.array(fitter.prior_flat).flat)
                    )
                )

                # Save covariance for parameters with data
                model_property_dictionary[model]['covariance_py'] = _numpy.array(
                    [
                        [_gvar.cov(data, parameter) for data in full_dataset]
                        for parameter in model_parameter_array
                    ]
                )

        # Save means in more convenient arrays
        model_means = [
            [parameter for parameter in model_property_dictionary[model]['means']]
            for model in model_property_dictionary.keys()
        ] # Individual parameter means from each model
        model_means_outer = [_numpy.outer(ps, ps) for ps in model_means] # Outer product of means
        
        # Model-average parameters
        means = [
            sum(w * p for w, p in zip(weight_array, model_mean))
            for model_mean in _numpy.transpose(model_means)
        ]

        # Model-average covariance
        cov_pp = sum(
            w * model_property_dictionary[model]['covariance_pp']
            for w, model in zip(weight_array, model_property_dictionary.keys())
        )
        cov_pp += sum(w * papb for w, papb in zip(weight_array, model_means_outer))
        cov_pp -= _numpy.outer(means, means)

        # Save model average
        if primary:
            cov_py = _numpy.transpose(sum(
                w * model_property_dictionary[model]['covariance_py']
                for w, model in zip(weight_array, model_property_dictionary.keys())
            ))
            p = _gvar.gvar(
                means, cov_pp,
                full_dataset, cov_py
            )
        else: p = _gvar.gvar(means, cov_pp)
        self.p = self._map_keys(p)

    # Helper function for mapping flat array back into a dictionary
    def _map_keys(self, p):
        # Convert flattened array into dictionary
        if self._hierarchical:
            for lvl in self.p0.keys():
                for ky in self.parameter_locations[lvl].keys():
                    self._p[lvl][ky] = p[self._lngths[lvl][ky][0]:self._lngths[lvl][ky][-1]]
        else:
            for ky in self.parameter_locations.keys():
                self._p[ky] = p[self._lngths[ky][0]:self._lngths[ky][-1]]

        # Return dictionary of fit parameters
        return self._p
