import typing
from collections.abc import Callable
import numpy as _numpy
import gvar as _gvar
import gc as _gc
from functools import partial as _partial
from .. import fit as _swissfit

def _flatten(dct, pk = tuple()):
    x = []
    for k,kv in dct.items():
        nk = pk + (k,) if pk else (k,)
        is_swissfit = isinstance(kv, _swissfit.SwissFit)
        is_dct = isinstance(kv, dict) or isinstance(kv, _gvar.BufferDict)
        if (not is_swissfit) and (is_dct):
            x.extend(_flatten(kv, pk = nk).items())
        else: x.append((nk,kv))
    return _gvar.BufferDict(x)

def _expand(dct, x = {}, pk = tuple(), kk = tuple()):
    if not kk:
        for k in dct.keys():
            if k[0] not in x: x[k[0]] = {}
            _expand(dct,x=x[k[0]],pk=k[1:],kk=k)
        return x
    else:
        if len(pk) == 1: x.append(dct[kk])
        if len(pk) == 2:
            if pk[0] not in x: x[pk[0]] = []
            _expand(dct,x=x[pk[0]],pk=pk[1:],kk=kk)
        elif len(pk) > 2: 
            if pk[0] not in x: x[pk[0]] = {}
            _expand(dct,x=x[pk[0]],pk=pk[1:],kk=kk)

class BayesianModelAveraging(object): 
    def __init__(
            self,
            models: list[dict[str,any]] | list[_swissfit.SwissFit] = None,
            ydata: list[any] = None,
            parameter_directive: Callable[[dict[str,any]],dict[str,any]] = None,
            ic: str = None,
            custom_ic: Callable[[_swissfit.SwissFit],float] = None,
            ic_kwargs: dict[str,any] = {},
    ):
        self._is_list = True
        if models is not None:
            if not isinstance(models[0],_swissfit.SwissFit):
                self._models = [_flatten(m) for m in models]
                self._is_list = False
            else: self._models = [{('p',): f} for f in models]
        else: _swissfit.SwissFitException('Must specify models')

        if ydata is not None: 
            if isinstance(ydata,dict): self.data = _flatten(ydata)
            elif isinstance(ydata,int): 
                self.data = [None for _ in range(ydata)]
            else: self.data = ydata
        else: self.data = None

        if parameter_directive is None: self._pd = lambda x: x
        else: self._pd = parameter_directive

        if all(x is None for x in [ic,custom_ic]): self._ic = 'aic'
        else: self._ic = ic
        self._custom_ic = custom_ic
        self._ic_kwargs = ic_kwargs

    def _wght(self, fit, data):
        match self._ic:
            case 'logml'|'logML'|'ml'|'ML'|'bf'|'BF': 
                return _numpy.exp(-fit.logml)
            case 'aic'|'AIC':
                ic = fit.chi2
                ic += 2.*len(fit.pflat) 
                if data is not None: ic += 2.*(len(data) - len(fit.data['y']))
                return _numpy.exp(-0.5*ic)
            case _: 
                if self._custom_ic is not None: 
                    ic = self._custom_ic(fit, **self._ic_kwargs)
                    return _numpy.exp(-0.5*ic)
                else: _swissfit.SwissFitException(self._ic + ' is not supported')

    def _gdct(self): return _gvar.BufferDict()

    def _get_model_dictionary(self,n):
        return ([self._gdct() for nm,_ in enumerate(self._models)] for _ in range(n))

    def _get_buffer_dictionary(self,n):
        return (_gvar.BufferDict() for _ in range(n))
    
    def _get_wght_and_prms(self, nm):
        for kf,f in self._models[nm].items():
            if self.data is not None:
                if isinstance(self.data,dict):
                    data = self.data[kf]
                else: data = self.data
            else: data = None
            wfv = self._wght(f,data)
            pv = self._pd(f.p)
            for lprms,prms in pv.items():
                for nprm,prm in enumerate(prms):
                    k = kf+(lprms,)+(nprm,)
                    self._p[nm][k] = prm
                    for lf,_ in self._models[nm].items():
                        wv = wfv if kf == lf else 1.
                        for mprms,_ in pv.items():
                            for oprm,_ in enumerate(prms):
                                l = lf+(mprms,)+(oprm,)
                                self._w[nm][(k,l)] = wv
                                if nm == 0: self._Z[(k,l)] = wv
                                else: self._Z[(k,l)] += wv
                    if nm == 0: self._ap[k] = 0.
                    self._ap[k] += _gvar.mean(self._p[nm][k])*self._w[nm][(k,k)]
        self._pm[nm] = _gvar.mean(self._p[nm])
        self._pcov[nm] = _gvar.evalcov(self._p[nm])
        return nm

    def _get_model_average(self, nm):
        if nm == 0:
            for k in self._ap.keys(): self._ap[k] /= self._Z[(k,k)] 
        for kw in self._w[nm].keys():
            self._w[nm][kw] /= self._Z[kw]
            wv = self._w[nm][kw]
            if nm == 0: self._apcov[kw] = -self._ap[kw[0]]*self._ap[kw[-1]]
            self._apcov[kw] += self._pcov[nm][kw]*wv
            self._apcov[kw] += self._pm[nm][kw[0]]*self._pm[nm][kw[-1]]*wv
        return nm

    @property
    def p(self): # PRD103(114502), PRD109(014510)
        self._nmodels = len(self._models)
        models = list(range(self._nmodels))

        self._p, self._w, self._pm, self._pcov = self._get_model_dictionary(4)
        self._Z, self._ap, self._apcov = self._get_buffer_dictionary(3)

        [*map(self._get_wght_and_prms, models)]
        [*map(self._get_model_average, models)]

        del self._p, self._w, self._pm, self._pcov, self._Z
        _gc.collect()

        kwargs = {'x': {}, 'pk': tuple(), 'kk': tuple()}
        ps = _expand(_gvar.gvar(self._ap, self._apcov), **kwargs)
        if self._is_list: return ps['p']
        else: return ps
