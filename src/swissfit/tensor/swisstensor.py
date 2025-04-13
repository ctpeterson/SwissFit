# nim c -d:x86 -d:AVX512 -d:release --passC:-Ofast --threads:on --app:lib --out:backend.so backend
import typing as _typing
from enum import Enum as _enum
import backend as _b
import numpy as _np

_active = 0
SwissTensorNim = _typing.TypeVar("SwissTensorNim")
SwissTensorPy = _typing.TypeVar("SwissTensorPy")
class _Precision(_enum):
    Single = 32
    Double = 64
(_P32,_P64) = (_Precision.Single.value,_Precision.Double.value)

class SwissTensor(object):
    # - for swissvar, you should write a method than returns data as GVar
    # - you should consider removing the Nim implementation of SwissTensor
    #   altogether in favor of just manipulating swissarrays
    # ^^^^^^^^^^^ I THINK THAT THIS IS THE NEXT BIG MOVE ^^^^^^^^^^^^^^^
    def __init__(
            self, 
            shape: list[int], 
            precision: int = 64, 
            swisstensor: SwissTensorNim = None
        ) -> None:
        global _active
        self._id = _active
        self._shape = shape
        self._precision = precision
        
        if not any(precision == p for p in [_P32,_P64]):
            raise ValueError('Invalid precision: ' + str(precision))

        if swisstensor is None:
            match self._precision:
                case _Precision.Single.value: 
                    self._tensor = _b.newSwissTensorPy32(shape)
                case _Precision.Double.value: 
                    self._tensor = _b.newSwissTensorPy64(shape)
                case _: pass
        else: self._tensor = swisstensor

        _active += 1

    @property
    def swisstensor(self) -> SwissTensorNim: return self._tensor

    @property
    def numpy(self) -> _np.array: 
        match self._precision:
            case _Precision.Single.value: storage = _b.linearizePy(self._tensor)
            case _Precision.Double.value: storage = _b.linearizePy(self._tensor)
            case _: pass
        return _np.array(storage).reshape(self._shape)

    @property
    def shape(self) -> list[int]: return self._shape

    @property
    def id(self) -> int: return self._id

    def __getitem__(self, coord: list[int]) -> _np.float32 | _np.float64: 
        match self._precision:
            case _Precision.Single.value: 
                return _np.float32(_b.getPy(self._tensor,coord))
            case _Precision.Double.value: 
                return _np.float64(_b.getPy(self._tensor,coord))
            case _: pass

    def __setitem__(self, index: list[int], value: _np.float32 | _np.float64):
        _b.setPy(self._tensor,index,value)
        
    def __add__(self, other: SwissTensorPy) -> SwissTensorPy:
        match self._precision:
            case _Precision.Single.value:
                return SwissTensor(
                    shape = self._shape,
                    precision = _Precision.Single.value,
                    swisstensor = _b.addPy(self._tensor,other.swisstensor)
                )
            case _Precision.Double.value:
                return SwissTensor(
                    shape = self._shape,
                    precision = _Precision.Double.value,
                    swisstensor = _b.addPy(self._tensor,other.swisstensor)
                )
            case _: pass
        
    
    def __sub__(self, other: SwissTensorPy) -> SwissTensorPy:
        match self._precision:
            case _Precision.Single.value:
                return SwissTensor(
                    shape = self._shape,
                    precision = _P32,
                    swisstensor = _b.subPy(self._tensor,other.swisstensor)
                )
            case _Precision.Double.value:
                return SwissTensor(
                    shape = self._shape,
                    precision = _P64,
                    swisstensor = _b.subPy(self._tensor,other.swisstensor)
                )
            case _: pass
    
    #############################################################################

    # lazy - there's a better way to do this
    def _printarray(self, printcutoff: int) -> str:
        output = "["
        nele = _b.lenPy(self._tensor)
        if nele > printcutoff:
            return "[" + ",".join([*map(str,[
                self[[0]],self[[1]],self[[2]],
                "...",
                self[[nele-2]],self[[nele-1]]
            ])]) + "]"
        else: return "[" + ",".join([*map(str,_b.linearizePy(self._tensor))]) + "]"

    # lazy - there's a better way to do this
    def _printmatrix(self, printcutoff: int) -> str:
        output = "["
        ncols = self._shape[0]
        if ncols > printcutoff:
            for idx1 in range(3):
                nrows = self._shape[-1]
                if nrows > printcutoff:
                    output += "["
                    for idx2 in range(3): 
                        if idx2 != 0: output += ", "
                        output += str(self[[idx1,idx2]])
                    output += ", ..., "
                    for idx2 in range(2): 
                        if idx2 != 0: output += ", "
                        output += str(self[[idx1,nrows-idx2-1]])
                else:
                    output += "["
                    for idx2 in range(nrows): 
                        if idx2 != 0: output += ", "
                        output += str(self[[idx1,idx2]])
                output += "],\n"
            output += "..., \n"
            for idx1 in range(2):
                nrows = self._shape[-1]
                if nrows > printcutoff:
                    output += "["
                    for idx2 in range(3): 
                        if idx2 != 0: output += ", "
                        output += str(self[[ncols-idx1-1,idx2]])
                    output += ", ..., "
                    for idx2 in range(2): 
                        if idx2 != 0: output += ", "
                        output += str(self[[ncols-idx1-1,nrows-idx2-1]])
                else:
                    output += "["
                    for idx2 in range(nrows): 
                        if idx2 != 0: output += ", "
                        output += str(self[[ncols-idx1-1,idx2]])
                output += "]"
                if idx1 != 1: output += ",\n"
        else:
            for idx1 in range(ncols):
                nrows = self._shape[-1]
                if nrows > printcutoff:
                    output += "["
                    for idx2 in range(3): 
                        if idx2 != 0: output += ", "
                        output += ", " + str(self[[idx1,idx2]])
                    output += ", ..."
                    for idx2 in range(2): 
                        if idx2 != 0: output += ", "
                        output += str(self[[idx1,nrows-idx2-1]])
                else:
                    output += "["
                    for idx2 in range(nrows): 
                        if idx2 != 0: output += ", " 
                        output += str(self[[idx1,idx2]])
                output += "]"
                if idx1 != ncols - 1: output += ",\n"
        output += "]"
        return output

    # lazy - there's a better way to do this
    def __str__(self):
        printcutoff = 10
        output = ""
        match len(self._shape):
            case 1: output += self._printarray(printcutoff)
            case 2: output += self._printmatrix(printcutoff)
            case _: pass
        return output + '\n'
        
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
    def test(size, verbosity = 0):
        print(25* '~~' + '  ' + str(size) + '  ' + 25*'~~')
        def _set(t,t00,t01,t10,t11):
            t[[0,0]] = t00
            t[[0,1]] = t01
            t[[1,0]] = t10
            t[[1,1]] = t11
        def _get(t): 
            if verbosity > 0: print(str(t),t[[0,0]],t[[0,1]],t[[1,0]],t[[1,1]])
        def _npcompare(test,t1,t2,operation):
            import time as _time
            print(15 * "-." + test + 15*"-.")
            npt1 = t1.numpy
            npt2 = t2.numpy
            t0 = _time.time()
            rsf = operation(t1,t2)
            tsf = _time.time() - t0
            print("swissfit","["+str(rsf.id)+"]:", tsf)
            t0 = _time.time()
            operation(npt1,npt2)
            tnp = _time.time() - t0
            print("numpy: ", tnp)
        tt1 = swisstensor(shape = [size,size])
        tt2 = swisstensor(shape = [size,size])
        _set(tt1,1./size/size,2./size/size,3./size/size,4./size/size)
        _set(tt2,4./size/size,3./size/size,2./size/size,1./size/size)
        _get(tt1 + tt2)
        _get(tt1 - tt2)
        _npcompare("addition",tt1,tt2,(lambda x,y: x + y))
        _npcompare("subtraction",tt1,tt2,(lambda x,y: x - y))
        print(25* '~~' + '  ' + str(size) + '  ' + 25*'~~')
    test(2, verbosity = 1)
    test(4, verbosity = 1)
    test(7)
    test(8)
    test(10)
    test(100)
    test(500)
    test(1000)
    test(1500)
    test(2000)
    test(2500) # <--- about when SwissFit is slower than NumPy w/ AVX512
    test(5000)
    
    