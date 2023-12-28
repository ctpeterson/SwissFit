from functools import partial as _partial # Partial evaluation of functions
import numpy as _numpy # General numerical operations
import gvar as _gvar # GVar Gaussian error propagation
from .neuralnetwork import NeuralNetwork as _NeuralNetwork # Parent neural network class

""" Feedforward neural network class """
class FeedforwardNeuralNetwork(_NeuralNetwork):
    def __init__(self,
                 topo,
                 custom_activation = None,
                 approximate_activation = False,
                 seed = 12345
                 ):
        # Initialize parent neural network class
        super().__init__(topo = topo, seed = seed)

        # Define activation functions & include (optional) custom activation function
        self.activation = {
            'linear': self.identity,
            'relu': self.RELU,
            'gelu': {
                True: self.GELU_approximation,
                False: self.GELU
            }[approximate_activation],
            'elu': self.ELU,
            'silu': self.SiLU,
            'tanh': self.tanh,
            'sigmoid': self.sigmoid
        }
        if custom_activation is not None: self.activation['cust'] = custom_activation

        # Get shape of network weights for each layer
        for lyr in self.topo.keys():
            self.topo[lyr]['weight.shape'] = (self.topo[lyr]['in'], self.topo[lyr]['out'])

        # Choose "_lyr_out" method depending on whether dropout is being used
        if any('dropout_rate' in self.topo[lyr].keys() for lyr in self.topo.keys()):
            self._lyr_out = self._lyr_out_dropout
        else: self._lyr_out = self._lyr_out_default

    """ Network initialization and priors """

    # Initialize network weights & biases
    def initialize_parameters(self, initialization = 'Kumar',
                              scale_weight_variance = {},
                              initialize_bias = True, p0 = None,
                              ):
        # Initialize network parameters for each layer
        parameters = {} if p0 is None else p0
        lyrs = list(self.topo.keys())
        for lyr in lyrs:
            # Get number of nodes coming in & number of nodes coming out
            in_dim, out_dim = self.topo[lyr]['weight.shape']

            # Rescale variance of initialization if desired
            if lyr in scale_weight_variance.keys(): weight_scale = scale_weight_variance[lyr]
            else: weight_scale = 1.

            # Initialize weights as desired
            match initialization:
                case 'zero':
                    """
                    The conventional wisdom in the machine learning literature is
                    that the initial network weights should be chosen randomly 
                    according to some appropriate distribution (usually assuming
                    that the variance of the output of any given node is of order
                    unity). This is certainly true if the network is being trained
                    (fit to data) using an algorithm that searches for the optimum
                    of the objective function by updating the network parameters
                    iteratively using the Jacobian (or any other derivative-based
                    quantity). 

                    However, if the optimization algorithm has built into it
                    occasional random perturbations of the network parameters (as
                    the 'basin hopping' algorithm does), then this randomness is,
                    in some sense, already built into the optimization process. 
                    Hence, random initialization of the coordinates may not be as
                    beneficial as it is for more conventional optimization algorithms,
                    such as stochastic gradient descent and its variants. 

                    Therefore, I personally recommend using this initialization if
                    you are using the basin hopping algorithm or, really, any 
                    optimization algorithm that has a random perturbation step
                    built into it. Of course, it is always good to check that an
                    alternative initialization does not lead to significantly 
                    different results in any case.
                    """
                    parameters[lyr + '.weight'] = [0. for w_ind in range(in_dim * out_dim)]
                case 'Kumar': # arXiv:1704.08863
                    """
                    I'm calling this 'Kumar' initialization because I am not aware of
                    a proper name for it and the convention for network initializations
                    seems to be naming it after the first or last name of the first
                    author of the originating paper (arXiv:1704.08863). The variance
                    for this initialization is

                    var[W_ij]^{-1} = (1 + act(0)^{2}) * act'(0)^2 * fan_in,

                    where 'act(0)' is the value of the activation function at x = 0,
                    act'(0) is the activation function's derivative at x = 0, and
                    'fan_in' is the output dimension of the previous layer. If the 
                    activation function is ReLU, then the variance is set to

                    var[W_ij] = 2 / fan_in (ReLU),

                    which is just the 'Kaiming' initialization.
                    """
                    # Get width of initialization
                    if self.topo[lyr]['activation'] != 'relu':
                        # Get act(0) & dact(x)/dx|_{x=0}
                        x0 = _gvar.gvar('0(0)') # Create GVar variable for autodiff
                        act = self.activation[self.topo[lyr]['activation']](x0)
                        dact = act.deriv(x0)

                        # Calculate standard deviation for initialization
                        sdev = weight_scale
                        sdev /= (1 + _gvar.mean(act)**2.) * _gvar.mean(dact)**2. * in_dim
                        sdev = _numpy.sqrt(sdev)
                    else: sdev = _numpy.sqrt(2. * weight_scale / in_dim)

                    # Finally, initialize weights
                    parameters[lyr + '.weight'] = [
                        self._rng.normal(0., sdev) for w in range(in_dim * out_dim)
                    ]
                case 'Xavier': # JMLR Workshop and Conference Proceedings, 2010.
                    """
                    Xavier initialization is really only appropriate for linear
                    or piecewise linear activations, like ReLU. Here, 'fan_in'
                    is the number of connections from the last layer going
                    into a node of the current layer (i.e., the output dimension
                    of the previous layer) and 'fan_out' is the number of connections
                    coming out of a node in the current layer (i.e., the input 
                    dimension of the next layer). 
                    """
                    # Get "fan out"
                    lyr_index = lyrs.index(lyr)
                    if lyr_index != len(lyrs) - 1:
                        fan_out = self.topo[lyrs[lyr_index + 1]]['in']
                    else: fan_out = 0.
                
                    # Set width of initialization
                    sdev = 6. * weight_scale / (in_dim + fan_out)
                    sdev = _numpy.array(sdev)
                
                    # Initialize parameters
                    parameters[lyr + '.weight'] = [
                        self._rng.normal(0., sdev) for w_ind in range(in_dim * out_dim)
                    ]
                case 'Kaiming': # arXiv:1502.01852
                    # Get width of weight initialization
                    sdev = _numpy.sqrt(2. * weight_scale / in_dim)
                
                    parameters[lyr + '.weight'] = [
                        self._rng.normal(0., sdev) for w_ind in range(in_dim * out_dim)
                    ]
                case _: print(initialization, 'not valid for weights')

            # Zero-initialization of bias (if requested)
            if initialize_bias: parameters[lyr + '.bias'] = [0. for b in range(out_dim)]

        # Return initialized parameters
        return parameters

    # Set network priors
    def network_priors(self,
                       prior_choice_weight = {},
                       prior_choice_bias = {},
                       prior = {}
                       ):
        """
        Initialization of network priors. For more complicated options, it is
        not difficult to create your own function or method based on this method.

        Priors selected by dictionaries of the form

        prior_choice_<weight or bias>[layer_key] = {
            'prior_type': <'none' or 'ridge_regression'>,
            'mean': <mean of Gaussian prior>
            'standard_deviation': <standard deviation of Gaussian prior>
        }

        for each layer key, as specified by the topology dictionary.

        + Prior choices:
        
          - 'none': No prior
          - 'ridge_regression': Equivalent to adding a "L2-regularization" term 
            to the loss function in the machine learning literature. Puts a smooth
            'cutoff' on the network parameters p_i of order 

            lambda ~ 'weight or bias standard deviation'

            by introducing a term of the form

            chi^2_{ridge reg. prior} = (1 / lambda^2) * sum_i (p_i - <p_i>)^2,

            where the mean <p_i> = 0. We allow for <p_i> != 0 to keep the 
            prior as general as possible.
        """
        # Set priors according to user's specification
        lyrs = list(self.topo.keys())
        for lyr in lyrs:
            # Input/output dimensions of current layer
            in_dim, out_dim = self.topo[lyr]['weight.shape']
            
            # Prior for weights
            if lyr in prior_choice_weight.keys():
                match prior_choice_weight[lyr]['prior_type']:
                    case 'ridge_regression': prior[lyr + '.weight'] = [
                        _gvar.gvar(
                            prior_choice_weight[lyr]['mean'],
                            prior_choice_weight[lyr]['standard_deviation']
                        )
                        for w_ind in range(in_dim * out_dim)
                    ]
                    case 'none': pass
                    case _: print(
                        prior_choice_weight[lyr]['prior_type'],
                        'is not valid for weights'
                    )

            # Prior for biases
            if lyr in prior_choice_bias.keys():
                match prior_choice_bias[lyr]['prior_type']:
                    case 'ridge_regression': prior[lyr + '.bias'] = [
                        _gvar.gvar(
                            prior_choice_bias[lyr]['mean'],
                            prior_choice_bias[lyr]['standard_deviation']
                        )
                        for w_ind in range(in_dim * out_dim)
                    ]
                    case 'none': pass
                    case _: print(
                        prior_choice_bias[lyr]['prior_type'],
                        'is not valid for biases'
                    )

        # Return prior
        return prior
            
    """ Monte Carlo dropout """
        
    # Monte Carlo dropout test
    def _dropout_test(self, lyr):
        return self._rng.uniform(0., 1.) <= self.topo[lyr]['dropout_rate']
        
    # Do Monte Carlo dropout
    def _dropout(self, lyr, x):
        if 'dropout_rate' in self.topo[lyr].keys():
            return [0. if self._dropout_test(lyr) else xx for xx in x]
        else: return x


    """ Forward pass through neural network """
    
    # Output of full network with MC dropout
    def _lyr_out_dropout(self, x, p):
        """ 
        Forward pass: 

        x = FNN(x) = (lyr_N • ... • lyr_2 • lyr_1)(x) = (•_{i=1}^{N} lyr_i)(x)
        with lyr_i = activation_i(x^T * W_i + b_i)

        - Performs Monte Carlo dropout if requested in input dictionary
        - Takes in scalar inputs, along with vector inputs (translated accordingly)
        - As written in the code
          + 1: x^T * W_i + b_i
          + 2: Monte Carlo dropout (optional)
          + 3: activation_i(x^T * W_i + b_i)
        """
        # Forward pass
        for lyr in self.topo.keys():
            x = self.activation[self.topo[lyr]['activation']]( # <---- act(x^T * W + b) ... 3
                self._dropout( # <---- Monte Carlo dropout                              ... 2
                    lyr,
                    _numpy.dot( # <---- x^T * W + b                                     ... 1
                        x, _numpy.reshape(p[lyr + '.weight'], self.topo[lyr]['weight.shape'])
                    ) + p[lyr + '.bias'] # x^T * W + b                                  ... 1
                ) # <---- Monte Carlo dropout                                           ... 2
            ) # <---- act(x^T * W + b)                                                  ... 3
        return x

    # Default output of full network
    def _lyr_out_default(self, x, p):
        """ 
        Forward pass: 

        x = FNN(x) = (lyr_N • ... • lyr_2 • lyr_1)(x) = (•_{i=1}^{N} lyr_i)(x)
        with lyr_i = activation_i(x^T * W_i + b_i)

        - Performs Monte Carlo dropout if requested in input dictionary
        - Takes in scalar inputs, along with vector inputs (translated accordingly)
        - As written in the code
          + 1: x^T * W_i + b_i
          + 2: activation_i(x^T * W_i + b_i)
        """
        for lyr in self.topo.keys():
            x = self.activation[self.topo[lyr]['activation']]( # <---- act(x^T * W + b) ... 2
                _numpy.dot( # <---- x^T * W + b                                         ... 1
                    x, _numpy.reshape(p[lyr + '.weight'], self.topo[lyr]['weight.shape'])
                ) + p[lyr + '.bias'] # x^T * W + b                                      ... 1
            ) # <---- act(x^T * W + b)                                                  ... 2
        return x
