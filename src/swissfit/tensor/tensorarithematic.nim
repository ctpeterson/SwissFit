import ../array/[swissarray]
import tensor,attributes

template newOperationSet(O1,O2,O3: untyped) =
  proc O2*[V:static[int],T](x,y: SwissTensor[V,T]): SwissTensor[V,T] =
    operable(x,y)
    like(result,x)
    result.storage := O1(x.storage,y.storage)
  proc O3*[V:static[int],T](x: var SwissTensor[V,T]; y: SwissTensor[V,T]) =
    operable(x,y)
    x.storage := O1(x.storage,y.storage)
  proc O2*[V:static[int],T](x: SwissTensor[V,T]; y: SomeNumber): SwissTensor[V,T] =
    like(result,x)
    result.storage := O1(x.storage,T(y))
  proc O2*[V:static[int],T](x: SomeNumber; y: SwissTensor[V,T]): SwissTensor[V,T] = 
    result = O2(y,x)
  proc O3*[V:static[int],T](x: var SwissTensor[V,T]; y: T) = (x = O2(x,y))

template define() {.dirty.} =
  newOperationSet(add,`+`,`+=`)
  newOperationSet(sub,`-`,`-=`)
  newOperationSet(mul,`*`,`*=`)
  newOperationSet(divd,`/`,`/=`)

define()

proc `-`*[V:static[int],T](x: SwissTensor[V,T]): SwissTensor[V,T] = T(-1.0)*x

#[
proc `+`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := add(x.storage,y.storage)
proc `+=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.add(y.storage)

proc `+`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := add(x,y.storage)
proc `+`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := add(x.storage,y)
proc `+=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) = (x.storage.add(y))

proc `-`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := subtract(x.storage,y.storage)
proc `-=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.subtract(y.storage)

proc `-`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := subtract(x,y.storage)
proc `-`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := subtract(x.storage,y)
proc `-=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) = (x.storage.subtract(y))

proc `*`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := multiply(x.storage,y.storage)
proc `*=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.multiply(y.storage)

proc `*`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := multiply(x,y.storage)
proc `*`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := multiply(x.storage,y)
proc `*=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) = (x.storage.multiply(y))

proc `/`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := divide(x.storage,y.storage)
proc `/=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.divide(y.storage)

proc `/`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := divide(x.storage,y)
proc `/`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := divide(x,y.storage)
proc `/=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) = (x.storage.divide(y))
]#