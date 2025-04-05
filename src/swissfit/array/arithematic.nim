import storage
import ../simd/[swisssimd]

template mixBinaryOperation(T,O1,O2: untyped; vlen: int) =
  proc O1*(x: SwissArray[T]; y: SomeNumber): SwissArray[T] = 
    let vcap = ((x.len mod vlen) + 1)*vlen
    like(result,x)
    for idx in countup(0,x.len-vlen,vlen):
      result.data.store(O2(load(x.data,idx),T(y).toSIMD()),idx)
    for idx in countup(vcap,x.len-1,1): result.data[idx] = O2(x.data[idx],T(y))
  proc O1*(x: SomeNumber; y: SwissArray[T]): SwissArray[T] = O1(y,x)

template newOperationSet(T,O1,O2: untyped) =
  let 
    vlen = case T is float32
      of true: VLENF
      of false: VLEND
  proc O1*(x,y: SwissArray[T]): SwissArray[T] =
    let vcap = ((x.len mod vlen) + 1)*vlen
    conformable(x,y)
    like(result,x)
    for idx in countup(0,x.len-vlen,vlen):
      result.data.store(O2(load(x.data,idx),load(y.data,idx)),idx)
    for idx in countup(vcap,x.len-1,1):
      result.data[idx] = O2(x.data[idx],y.data[idx])
  mixBinaryOperation(T,O1,O2,vlen)

template define(T: untyped) {.dirty.} =
  newOperationSet(T,add,`+`)
  newOperationSet(T,sub,`-`)
  newOperationSet(T,mul,`*`)
  newOperationSet(T,divd,`/`)

define(float32)
define(float64)

if isMainModule:
  template test(N,F: untyped) =
    let size = 2*N + 1
    var (v1,v2) = (newSeq[F](size),newSeq[F](size))
    for idx in 0..<size: 
      v1[idx] = F(idx)
      v2[idx] = F(size-idx)
    var sv1,sv2: SwissArray[F]
    sv1 = new(v1)
    sv2 = new(v2)
    echo "SV1: ", $sv1
    echo "SV2: ", $sv2
    echo "VECTOR-VECTOR OPERATIONS"
    echo "ADD: ", $add(sv1,sv2)
    echo "SUB: ", $sub(sv1,sv2)
    echo "MUL: ", $mul(sv1,sv2)
    echo "DIV: ", $divd(sv1,sv2)
    echo "VECTOR-SCALAR OPERATIONS"
    echo "ADDS: ", $add(sv1,1.0), " ADDS: ", $add(sv1,1)
    echo "SUBS: ", $sub(sv1,1.0), " SUBS: ", $sub(sv1,1)
    echo "MULS: ", $mul(sv1,2.0), " MULS: ", $mul(sv1,2)
    echo "DIVS: ", $divd(sv1,2.0), " MULS: ", $divd(sv1,2)
    echo "SCALAR-VECTOR OPERATIONS"
    echo "ADDS: ", $add(1.0,sv1), " ADDS: ", $add(1,sv1)
    echo "SUBS: ", $sub(1.0,sv1), " SUBS: ", $sub(1,sv1)
    echo "MULS: ", $mul(2.0,sv1), " MULS: ", $mul(2,sv1)
    echo "DIVS: ", $divd(2.0,sv1), " MULS: ", $divd(2,sv1)
  test(15,float32)
  test(7,float64)
  test(16,float32)
  test(8,float64)


#[
proc add*[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
  #addSIMD(result.data,x.data,y.data,result.len)
  #loadSIMD(result.address,addr x.data[0])
  
  for idx in 0..<x.len: result[idx] = x[idx] + y[idx]
proc add*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  for idx in 0..<x.len: x[idx] = x[idx] + y[idx]

proc add*[T](x: SwissArray[T]; y: T): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx] + y
proc add*[T](x: T; y: SwissArray[T]): SwissArray[T] =
  like(result,y)
  for idx in 0..<y.len: result[idx] = x + y[idx]
proc add*[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = x[idx] + y

proc subtract*[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx] - y[idx]
proc subtract*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  for idx in 0..<x.len: x[idx] = x[idx] - y[idx]

proc subtract*[T](x: SwissArray[T]; y: T): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx] - y
proc subtract*[T](x: T; y: SwissArray[T]): SwissArray[T] =
  like(result,y)
  for idx in 0..<y.len: result[idx] = x - y[idx]
proc subtract*[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = x[idx] - y

proc multiply*[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx]*y[idx]
proc multiply*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  for idx in 0..<x.len: x[idx] = x[idx]*y[idx]

proc multiply*[T](x: T; y: SwissArray[T]): SwissArray[T] =
  like(result,y)
  for idx in 0..<y.len: result[idx] = x*y[idx]
proc multiply*[T](x: SwissArray[T]; y: T): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx]*y
proc multiply*[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = y*x[idx]

proc divide*[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx]/y[idx]
proc divide*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  for idx in 0..<x.len: x[idx] = x[idx]/y[idx]

proc divide*[T](x: T; y: SwissArray[T]): SwissArray[T] =
  like(result,y)
  for idx in 0..<y.len: result[idx] = x/y[idx]
proc divide*[T](x: SwissArray[T]; y: T): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx]/y
proc divide*[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = x[idx]/y
]#