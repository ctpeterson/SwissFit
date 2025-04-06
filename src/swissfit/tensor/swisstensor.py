# nim c -d:x86 -d:AVX --mm:orc --noMain --app:lib --out:backend.so --threads:on backend
import typing as _typing
from enum import Enum as _enum
import backend as _b
import numpy as _np

_active = 0

class Precision(_enum):
    Single = 32
    Double = 64

class SwissTensor(object):
    # for swissvar, you should write a method than returns data as GVar
    def __init__(self, shape: list[int], precision: int = 64):
        global _active
        self._id = _active
        self._shape = shape
        self._precision = precision
        match self._precision:
            case Precision.Single.value: self._tensor = _b.newTensor32(shape)
            case Precision.Double.value: self._tensor = _b.newTensor64(shape)
            case _: raise ValueError('Invalid precision: ' + str(precision))
        _active += 1
    
    @property 
    def get(self): return self._tensor

    @property
    def shape(self): return self._shape

    @property
    def id(self): return self._id

    @property
    def numpy(self): 
        return _np.array(_b.getLinear(self._tensor)).reshape(self._shape)

    def _copy(self): return SwissTensor(self._shape, precision = self._precision)
    def _operate(self, x): _b.set(self._tensor,x)

    def __getitem__(self, index: list[int]):
        return _b.element(self._tensor,index)
    
    def __setitem__(self, index: list[int], value):
        _b.setElement(self._tensor,index,value)

    def __add__(self, other):
        newTensor = self._copy()
        newTensor._operate(_b.add(self._tensor,other.get))
        return newTensor
    
    def __sub__(self, other):
        newTensor = self._copy()
        newTensor._operate(_b.sub(self._tensor,other.get))
        return newTensor
    
    def __mul__(self, other):
        newTensor = self._copy()
        newTensor._operate(_b.mul(self._tensor,other.get))
        return newTensor
    
    def __truediv__(self, other):
        newTensor = self._copy()
        newTensor._operate(_b.divd(self._tensor,other.get))
        return newTensor
    
    def __str__(self):
        return "[" + ",".join([*map(str,_b.getLinear(self._tensor))]) + "]"

def swisstensor(
        shape: list[int] = None, 
        tensor: list[_typing.Any] = None, 
        precision: int = 64
    ) -> SwissTensor:
    if tensor is not None: shape = tensor.shape()
    elif shape is None: ValueError('Must specify either shape or tensor')
    newTensor = SwissTensor(shape, precision = precision)
    return newTensor

if __name__ == '__main__':
    size = 1000
    def _set(t,t00,t01,t10,t11):
        t[[0,0]] = t00
        t[[0,1]] = t01
        t[[1,0]] = t10
        t[[1,1]] = t11
    def _get(t): 
        #print(str(t),t[[0,0]],t[[0,1]],t[[1,0]],t[[1,1]])
        pass
    def _npcompare(test,t1,t2,operation):
        import time as _time
        print(15 * "-." + test + 15*"-.")
        npt1 = t1.numpy
        npt2 = t2.numpy
        t0 = _time.time()
        operation(t1,t2)
        print("swissfit: ", _time.time() - t0)
        t0 = _time.time()
        operation(npt1,npt2)
        print("numpy: ", _time.time() - t0)
    tt1 = swisstensor(shape = [size,size])
    tt2 = swisstensor(shape = [size,size])
    _set(tt1,1./size/size,2./size/size,3./size/size,4./size/size)
    _set(tt2,4./size/size,3./size/size,2./size/size,1./size/size)
    _get(tt1 + tt2)
    _get(tt1 - tt2)
    _get(tt1*tt2)
    _npcompare("addition",tt1,tt2,(lambda x,y: x + y))
    _npcompare("subtraction",tt1,tt2,(lambda x,y: x - y))
    _npcompare("multiplication",tt1,tt2,(lambda x,y: x*y))
    _npcompare("division",tt1,tt2,(lambda x,y: x/y))
    
    