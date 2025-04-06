import nimpy
import ../array/[swissarray]
import tensor,attributes

template newOperationSet(O1,O2,O3: untyped) =
  proc O2*[T](x,y: SwissTensor[T]): SwissTensor[T] =
    operable(x,y)
    like(result,x)
    result.storage := O1(x.storage,y.storage)
  proc O3*[T](x: var SwissTensor[T]; y: SwissTensor[T]) =
    operable(x,y)
    x.storage := O1(x.storage,y.storage)
  proc O2*[T](x: SwissTensor[T]; y: SomeNumber): SwissTensor[T] =
    like(result,x)
    result.storage := O1(x.storage,T(y))
  proc O2*[T](x: SomeNumber; y: SwissTensor[T]): SwissTensor[T] = 
    result = O2(y,x)
  proc O3*[T](x: var SwissTensor[T]; y: T) = (x = O2(x,y))

template define() {.dirty.} =
  newOperationSet(add,`+`,`+=`)
  newOperationSet(sub,`-`,`-=`)
  newOperationSet(mul,`*`,`*=`)
  newOperationSet(divd,`/`,`/=`)

define()

proc `-`*[T](x: SwissTensor[T]): SwissTensor[T] = T(-1.0)*x