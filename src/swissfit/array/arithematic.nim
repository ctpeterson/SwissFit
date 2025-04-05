import storage
import ../simd/[swisssimd]

template mixBinaryOperation(T,O1,O2: untyped; vlen: int) =
  proc O1*(x: SwissArray[T]; y: SomeNumber): SwissArray[T] = 
    like(result,x)
    for idx in countup(0,x.vcap-1,vlen):
      result.data.store(O2(load(x.data,idx),T(y).toSIMD()),idx)
    for idx in countup(x.vcap,x.len-1,1): result.data[idx] = O2(x.data[idx],T(y))
  proc O1*(x: SomeNumber; y: SwissArray[T]): SwissArray[T] = O1(y,x)

template newOperationSet(T,O1,O2: untyped) =
  let 
    vlen = case T is float32
      of true: VLENF
      of false: VLEND
  proc O1*(x,y: SwissArray[T]): SwissArray[T] =
    conformable(x,y)
    like(result,x)
    for idx in countup(0,x.vcap-1,vlen):
      result.data.store(O2(load(x.data,idx),load(y.data,idx)),idx)
    for idx in countup(x.vcap,x.len-1,1):
      result.data[idx] = O2(x.data[idx],y.data[idx])
  mixBinaryOperation(T,O1,O2,vlen)

template define(T: untyped) {.dirty.} =
  newOperationSet(T,add,`+`)
  newOperationSet(T,sub,`-`)
  newOperationSet(T,mul,`*`)
  newOperationSet(T,divd,`/`)

define(float32)
define(float64)
