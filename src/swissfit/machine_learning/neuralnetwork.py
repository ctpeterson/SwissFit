from functools import partial as _partial # Partial evaluation of functions
import numpy as _numpy # General numerical operations
import gvar as _gvar # GVar Gaussian error propagation
import sys as _sys # Fo exiting when there is an error

""" Constants """
_SQRT_TWO = _numpy.sqrt(2.) # sqrt(2)
_SQRT_TWO_O_PI =  _numpy.sqrt(2. / _numpy.pi) # sqrt(2/pi)
_GELU_APPROX_C = 11. / 123. / 2.
_GELU_APPROX_D = -4.480293397177321e-06 # <--- Obtained by curve fit
_SIG_lb, _SIG_ub = -100., 100.
_x2 = 0.

""" Base neural network class """
class NeuralNetwork(object):
    def __init__(self, topo = None, seed = None):
        # Save dictionary specifying neural network topology
        self.topo = self._check_and_correct_topology(topo)

        # Save random number generator
        self._rng = _numpy.random; self._rng.seed(seed);

        # Set scalar option to False by default
        self._scalar = False

    """ Output of full neural network """
    # Output of full neural network
    def out(self, x, p): return list(map(_partial(self._lyr_out, p = p), x))

    """ Checking topology dictionary """
    # Check & correct topology
    def _check_and_correct_topology(self, topo):
        # Error message for first/last layer
        input_error = 'You must specify input dimension '
        input_error += 'in topology dictionary for first layer. Exiting.'
        output_error = 'You must specify output dimension '
        output_error += 'in topology dictionary for last layer. Exiting.'

        # Go through layers & correct anything not specified
        layers = list(topo.keys()); num_layers = len(layers);
        for lyr_ind, lyr in enumerate(layers):
            # Correct lack of specification of input dimension
            if ('in' not in topo[lyr].keys()) and (lyr_ind - 1 >= 0):
                last_lyr = layers[lyr_ind - 1]
                if 'out' in topo[last_lyr].keys(): topo[lyr]['in'] = topo[last_lyr]['out']
            elif ('in' not in topo[lyr].keys()) and (lyr_ind - 1 < 0):
                print(input_error); _sys.exit();

            # Correct lack of specification of output dimension
            if ('out' not in topo[lyr].keys()) and (lyr_ind + 1 != num_layers):
                next_lyr = layers[lyr_ind + 1]
                if 'in' in topo[next_lyr].keys(): topo[lyr]['out'] = topo[next_lyr]['in']
            elif ('out' not in topo[lyr].keys()) and (lyr_ind + 1 == num_layers):
                print(output_error); _sys.exit();
                
        # Return network with appropriate inputs/outputs
        return topo
                
    """ Activation functions """
    # Identity activation
    def identity(self, x): return x
    
    # GELU approximation
    def GELU_approximation(self, x): # x * (1 + tanh[√{2/π} * x * ( 1 + 0.044715 * x^2 + ... )]) / 2
        global _x2
        _x2 = x * x
        return 0.5 * x * ( 1. + _gvar.tanh(
            _SQRT_TWO_O_PI * x * (
                1. + _GELU_APPROX_C * _x2 + _GELU_APPROX_D * _x2 * _x2
            )
        ) )

    # GELU (Gaussian error linear unit); if overflow, use approximation
    def GELU(self, x): # x * (1 + erf[x / √{2}]) / 2
        try: return 0.5 * x * (1. + _gvar.erf(x / _SQRT_TWO))
        except OverflowError: return self.GELU_approximation(x)

    # RELU (rectified linear unit)
    def RELU(self, x): # 0 if x < 0 and x if x > 0
        if hasattr(x, '__len__') or (type(x) is _numpy.ndarray):
            return _numpy.where(x > 0., x, 0.)
        else: return max(0., x)

    # ELU (exponential linear unit)
    def ELU(self, x): # exp(x) - 1 if x < 0 and x if x > 0
        if hasattr(x, '__len__') or (type(x) is _numpy.ndarray):
            return _numpy.where(x > 0., x, _gvar.exp(x) - 1.)
        return x if x > 0. else _gvar.exp(x) - 1.

    # Swish activation
    def SiLU(self, x):
        return x / (1. + _gvar.exp(-x))

    # tanh activation
    def tanh(self, x): return _gvar.tanh

    # Sigmoid activation
    def sigmoid(self, x): return 1. / (1. + _gvar.exp(-x))

    # Negative exponential activation
    def nexp(self, x): return _gvar.exp(-x)

    # Inverse linear activation
    def inverse_linear(self, x): return 1. / (1. + x)

    # Inverse root activation
    def inverse_square_root(self, x): 1. / _numpy.sqrt(1. + x)
