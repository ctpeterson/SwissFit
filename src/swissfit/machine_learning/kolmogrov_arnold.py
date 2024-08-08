from functools import partial as _partial # Partial evaluation of functions
import numpy as _numpy # General numerical operations
import gvar as _gvar # GVar Gaussian error propagation
from .neuralnetwork import NeuralNetwork as _NeuralNetwork # Parent neural network class
from ..interpolators import bspline

class _KolmogrovArnoldLayer(object):
    def __init__(self, nl, degree, layer_tag, activation, approximate_activation):
        self._layer_tag = layer_tag
        self._nl = range(nl)
        if hasattr(degree, len): self._spline = [BSpline(None, degree[n]) for n in self._nl]
        else: self._spline = [BSpline(None, degree) for n in self._nl]
        self._act = {
            'linear': self.identity,
            'relu': self.RELU,
            'gelu': {
                True: self.GELU_approximation,
                False: self.GELU
            }[approximate_activation],
            'elu': self.ELU,
            'silu': self.SiLU,
            'tanh': self.tanh,
            'sigmoid': self.sigmoid,
            None: None
        }[activation]
        
    def _spline_weight(self, p, i, j):
        return p[self._layer_tag + '.weight(' + ','.join([*map(str, [i,j])]) + ')']

    def _activation_weight(self, p, i, j):
        return p[self._layer_tag + '.activation(' + ','.join([*map(str, [i,j])]) + ')']

    def _activation(self, x, p):
        return p[0] * self._act(x)
    
    def _node_term(self, x, p, n, m):
        match self._act:
            case None: return self._spline[n](x, self._p(p, n, m)) 
            case _:
                result = self._spline[n](x, self._spline_weight(p, n, m))
                result += self._activation_weight(p, n, m)[0] * self._act(x)
                return result
                
    def _node(self, n, xknots, p):
        self._spline[n].set_xknots(xknots)
        return sum(self._node_term(xknot, p, n, m) for m, xknot in enumerate(xknots))
        
    def __call__(self, xknots, p):
        return [*map(_partial(self._node, xknots = xknots, p = p), self._nl)]
    
class KolmogrovArnoldNetwork(_NeuralNetwork):
    """ Kolmogrov-Arnold Network
    
    Python implementation of the Kolmogrov-Arnold network
    described in arXiv:2404.19756
    
    """
    def __init__(self, topo):
        super().__init__(topo = topo)
        self._construct_network()

    def _construct_network(self):
        self._layers = [
            _KolmogrovArnoldLayer(
                lyr['out'],
                lyr['degree'] if 'degree' in lyr.keys() else 3,
                lyr_tag,
                lyr['activation'] is 'activation' in lyr.keys() else None,
                lyr['approx_activation'] if 'approx_activation' in lyr.keys() else False
            ) for lyr_tag, lyr in self._topo.items()
        ]  

    # Need to write method for initialization
        
    def __call__(self, x, p):
        result = x
        for layer in self._layers: result = layer(result, p)
        return result
