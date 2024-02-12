import numpy as _numpy # General numerical operations
import gvar as _gvar # GVar Gaussian error propagation
from .neuralnetwork import NeuralNetwork as _NeuralNetwork # Parent neural network class

""" Various functions """
def _polyharmonic(rbf):
    if 'polyharmonic' in rbf:
        order = int(rbf.split('_')[-1])
        return {
            True: lambda x, p: x**order * _gvar.log(x),
            False: lambda x, p: x**order
        }[order % 2 == 0]

""" Radial basis function neural network """
class RadialBasisNeuralNetwork(_NeuralNetwork):
    def __init__(self,
                 topo,
                 normalized = False,
                 seed = 12345
                 ):
        # Initialize parent neural network class 
        super().__init__(topo = topo, seed = seed)
        
        # Get norm from dictionary of supported norms
        self._norm_dict = {
            'l1': _numpy.vectorize(lambda x: _numpy.linalg.norm(x)),
            'l2': _numpy.vectorize(lambda x: _numpy.square(_numpy.linalg.norm(x)))
        }

        # Get radial basis function (RBF) from list of supported RBFs
        self._rbf_dict = {
            'exp': lambda x, p: self.nexp(x * p),
            'inverse_quadratic': lambda x, p: self.inverse_linear(x * p),
            'multiquadratic': lambda x, p: _numpy.sqrt(1. + x * p),
            'inverse_multiquadratic': lambda x, p: self.inverse_square_root(x * p)
        }

        # Option for normalization
        self._normalized = normalized
        
        # Preparation of shapes for network parameters
        for lyr in self.topo.keys():
            # Shape of center or weight parameters
            if self.topo[lyr]['activation'] != 'linear':
                self.topo[lyr]['center.shape'] = (self.topo[lyr]['out'], self.topo[lyr]['in'])
            else: self.topo[lyr]['weight.shape'] = (self.topo[lyr]['in'], self.topo[lyr]['out'])

            # Default options for linear layer
            if self.topo[lyr]['activation'] == 'linear':
                if 'add_bias' not in self.topo[lyr].keys(): self.topo[lyr]['add_bias'] = True
                if 'normalized' not in self.topo[lyr].keys(): self.topo[lyr]['normalized'] = False

            # Default option for norm in radial basis function argument
            if 'norm' not in self.topo[lyr].keys(): self.topo[lyr]['norm'] = 'l2'

    """ Network initialization and priors - very limited base functionalities """
    
    # Initialize network parameters
    def initialize_parameters(self, initialization = 'zero', p0 = None):
        # Initialize network parameters for each layer
        parameters = {} if p0 is None else p0
        for lyr in self.topo.keys():
            # Input/output dimension
            in_dim, out_dim = self.topo[lyr]['in'], self.topo[lyr]['out']

            # Initialize network parameters
            match initialization:
                case 'zero':
                    if self.topo[lyr]['activation'] != 'linear':
                        parameters[lyr + '.center'] = [0. for c in range(in_dim * out_dim)]
                        parameters[lyr + '.bandwidth'] = [0. for b in range(out_dim)]
                    else:
                        parameters[lyr + '.weight'] = [0. for w in range(in_dim * out_dim)]
                        if 'add_bias' in self.topo[lyr].keys():
                            parameters[lyr + '.bias'] = [0. for b in range(out_dim)]
                case _: print(initialization, 'is not supported')

        # Return initialization
        return parameters
        
    # Set network parameters
    def network_priors(self,
                       prior_choice_center = {},
                       prior_choice_bandwidth = {},
                       prior_choice_weight = {},
                       prior_choice_bias = {},
                       prior = {}
                       ):
        # Set priors for network parameters
        for lyr in self.topo.keys():
            # Input/output dimension
            in_dim, out_dim = self.topo[lyr]['in'], self.topo[lyr]['out']

            # Center parameters
            if lyr in prior_choice_center.keys():
                match prior_choice_center[lyr]['prior_type']:
                    case 'ridge_regression':
                        prior[lyr + '.center'] = [
                            _gvar.gvar(
                                prior_choice_center[lyr]['mean'],
                                prior_choice_center[lyr]['standard_deviation']
                            ) for c in range(in_dim * out_dim)
                        ]
                    case _: print(prior_choice_center[lyr]['prior_type'], 'not supported')

            # Weight parameters
            if lyr in prior_choice_weight.keys():
                match prior_choice_weight[lyr]['prior_type']:
                    case 'ridge_regression':
                        prior[lyr + '.weight'] = [
                            _gvar.gvar(
                                prior_choice_weight[lyr]['mean'],
                                prior_choice_weight[lyr]['standard_deviation']
                            ) for w in range(in_dim * out_dim)
                        ]
                    case _: print(prior_choice_weight[lyr]['prior_type'], 'not supported')

            # Bandwidth parameters
            if lyr in prior_choice_bandwidth.keys():
                match prior_choice_bandwidth[lyr]['prior_type']:
                    case 'ridge_regression':
                        prior[lyr + '.bandwidth'] = [
                            _gvar.gvar(
                                prior_choice_bandwidth[lyr]['mean'],
                                prior_choice_bandwidth[lyr]['standard_deviation']
                            ) for b in range(out_dim)
                        ]
                    case _: print(prior_choice_bandwidth[lyr]['prior_type'], 'not supported')

            # Bias parameters
            if lyr in prior_choice_bias.keys():
                match prior_choice_bias[lyr]['prior_type']:
                    case 'ridge_regression':
                        prior[lyr + '.bias'] = [
                            _gvar.gvar(
                                prior_choice_bias[lyr]['mean'],
                                prior_choice_bias[lyr]['standard_deviation']
                            ) for b in range(out_dim)
                        ]
                    case _: print(prior_choice_bias[lyr]['prior_type'], 'not supported')

        # Return prior
        return prior
            
    """ Calculation of output of radial basis function network """
    
    # Convenience function for reshaping centers
    def _centers(self, lyr):
        return _numpy.reshape(self._p[lyr + '.center'], self.topo[lyr]['center.shape'])

    # Norm for specific layer
    def _norm(self, x, lyr):
        return self._norm_dict[self.topo[lyr]['norm']]([
            *map(lambda x: x if _gvar.mean(x) != 0 else x + 1e-64, x)
        ])
    
    # Radial basis function
    def _rbf(self, x, lyr):
        if self.topo[lyr]['activation'] != 'linear':
            # y = phi_{parameters}(x)
            return self._rbf_dict[self.topo[lyr]['activation']](
                self._norm(x - self._centers(lyr), lyr).flatten(),
                _gvar.abs(self._p[lyr + '.bandwidth'])
            )
        else:
            # x^T * W
            x = _numpy.dot(
                x, _numpy.reshape(self._p[lyr + '.weight'], self.topo[lyr]['weight.shape'])
            )

            # Normalization & bias options
            if self.topo[lyr]['normalized']: x /= _numpy.sum(x) # y_i = y_i / (sum_i y_i)
            if self.topo[lyr]['add_bias']: x += self._p[lyr + '.bias'] # y = x^T * W + b

            # Return result of linear layer
            if _numpy.linalg.norm(_gvar.mean(x)) < 1e-6:
                return _gvar.gvar('0(0)') if any(isinstance(xx, _gvar.GVar) for xx in x) else 0.
            else: return x

    # Output of full network
    def _lyr_out(self, x, p):
        # Check if input is scalar
        self._scalar, self._p = False, p
        
        # Forward pass through network
        for lyr in self.topo.keys(): x = self._rbf(x, lyr)

        # Return output of full network
        return x
    
