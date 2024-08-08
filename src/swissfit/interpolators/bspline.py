from functools import partial as _partial # Partial evaluation of functions
import numpy as _numpy # General numerical operations
import gvar as _gvar
from .interpolator import Interpolator as _Interpolator
from scipy.interpolate import BSpline as _BSpline

class BSpline(_Interpolator):
    def __init__(self, xknots = None, degree = 3):
        self._degree = degree
        self._bspline = _BSpline
        if xknots is not None: self.set_xknots(xknots)
        else: self._xknots = []

    def set_xknots(self, xknots):
        self._xknots = [*_gvar.mean(xknots)]
        self._xknots.sort()

    @property
    def xknots(self): return self._xknots

    @property
    def interval(self): return [self._xknots[0],self._xknots[-1]]

    def _spline(self,x):
        xgv,pgv = False,False
        spline = _BSpline(self._xknots,_gvar.mean(x[1:]),self._degree)
        xv = _gvar.mean(x[0])
        if isinstance(x[0],_gvar.GVar): 
            dfdx = [spline.derivative()(xv)]
            xgv = True
        else: dfdx = [0.]
        if any(isinstance(xx,_gvar.GVar) for xx in x[1:]):
            dfdp = [] 
            for i,p in enumerate(x[1:]):
                pp = _numpy.zeros((len(x[1:]),))
                pp[i] = 1.
                sp = _BSpline(self._xknots,pp,self._degree)
                dfdp.append(sp(xv))
            pgv = True
        else: dfdp = [0. for _ in x[1:]]
        if xgv or pgv: return _gvar.gvar_function(x,spline(xv),dfdx+dfdp)
        else: return spline(xv)

    def __call__(self, x, p): 
        if hasattr(x,'__len__'): return [self._spline([xx]+list(p)) for xx in x]
        else: return self._spline([x]+list(p))[0]