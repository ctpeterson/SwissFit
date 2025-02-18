# SwissFit Tensor & SwissVar types:
#
# Notes: 
#  - SwissArray type modeled after custom sequence example in Nim documentation
#    - https://nim-lang.org/docs/destructors.html
#  - SwissTensor type modeled after Arraymancer Tensor
#    - https://github.com/mratsim/Arraymancers

import simd

type
  SwissArray*[T] = object
    len,cap: int
    data: ptr UncheckedArray[T]

  SwissTensor*[V:static[int],T] = object
    offset: int
    shape: array[V,int]
    strides: array[V,int]
    storage*: SwissArray[T]

#[
proc `=destroy`[T](x: var SwissArray[T]) =
  if x.data != nil:
    dealloc(x.data)
    x.data = nil
]#

proc `=trace`[T](x: var SwissArray[T]; env: pointer) =
  if x.data != nil:
    for idx in 0..<x.len: `=trace`(x.data[idx],env)

proc `=wasMoved`[T](x: var SwissArray[T]) = (x.data = nil)

proc `=sink`[T](x: var SwissArray[T]; y: SwissArray[T]) =
  `=destroy`(x); x.len = y.len
  x.cap = y.cap 
  x.data = y.data

proc `=copy`[T](x: var SwissArray[T]; y: SwissArray[T]) =
  if x.data == y.data: return
  x.len = y.len
  x.cap = y.cap
  if y.data != nil:
    x.data = cast[typeof(x.data)](alloc(x.cap*sizeof(T)))
    for idx in 0..<x.len: x.data[idx] = y.data[idx]

proc `=dup`[T](x: SwissArray[T]): SwissArray[T] {.nodestroy.} =
  result = SwissArray[T](len: x.len, cap: x.cap, data: nil)
  if x.data != nil:
    result.data = cast[typeof(result.data)](alloc(result.cap*sizeof(T)))
    for idx in 0..<result.len: result.data[idx] = `=dup`(x.data[idx])

proc append*[T](x: var SwissArray[T]; y: sink T) =
  if x.len >= x.cap:
    x.cap = max(x.len + 1, 2*x.cap)
    x.data = cast[typeof(x.data)](realloc(x.data, x.cap*sizeof(T)))
  x.data[x.len] = y; inc x.len;

proc new*[T](xs: seq[T]): SwissArray[T] =
  result = SwissArray[T](len: xs.len, cap: xs.len)
  result.data = cast[typeof(result.data)](alloc(result.cap*sizeof(T)))
  for idx in 0..<result.len: result.data[idx] = xs[idx]

proc new*[T](len: int; t: typedesc[T]): SwissArray[T] =
  result = SwissArray[T](len: len, cap: len)
  result.data = cast[typeof(result.data)](alloc(result.cap*sizeof(T)))

template like[T](x: var SwissArray[T]; y: SwissArray[T]) =
  x = SwissArray[T](len: y.len, cap: y.len)
  x.data = cast[typeof(x.data)](alloc(x.cap*sizeof(T)))

template `[]`*[T](x: SwissArray[T]; idx: Natural): lent T =
  assert idx < x.len
  x.data[idx]
template `[]=`*[T](x: var SwissArray[T]; idx: Natural; y: sink T) =
  assert idx < x.len 
  x.data[idx] = y
proc len*[T](x: SwissArray[T]): int {.inline.} = x.len

proc conformable[T](x,y: SwissArray[T]) =
  assert(x.len == y.len) 
  assert(x.cap == y.cap)

proc `:=`*[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = y

proc add[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx] + y[idx]
proc add[T](x: var SwissArray[T]; y: SwissArray[T]) =
  for idx in 0..<x.len: x[idx] = x[idx] + y[idx]

proc add[T](x: SwissArray[T]; y: T): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx] + y
proc add[T](x: T; y: SwissArray[T]): SwissArray[T] =
  like(result,y)
  for idx in 0..<y.len: result[idx] = x + y[idx]
proc add[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = x[idx] + y

proc subtract[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx] - y[idx]
proc subtract[T](x: var SwissArray[T]; y: SwissArray[T]) =
  for idx in 0..<x.len: x[idx] = x[idx] - y[idx]

proc subtract[T](x: SwissArray[T]; y: T): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx] - y
proc subtract[T](x: T; y: SwissArray[T]): SwissArray[T] =
  like(result,y)
  for idx in 0..<y.len: result[idx] = x - y[idx]
proc subtract[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = x[idx] - y

proc multiply[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx]*y[idx]
proc multiply[T](x: var SwissArray[T]; y: SwissArray[T]) =
  for idx in 0..<x.len: x[idx] = x[idx]*y[idx]

proc multiply[T](x: T; y: SwissArray[T]): SwissArray[T] =
  like(result,y)
  for idx in 0..<y.len: result[idx] = x*y[idx]
proc multiply[T](x: SwissArray[T]; y: T): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx]*y
proc multiply[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = y*x[idx]

proc divide[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx]/y[idx]
proc divide[T](x: var SwissArray[T]; y: SwissArray[T]) =
  for idx in 0..<x.len: x[idx] = x[idx]/y[idx]

proc divide[T](x: T; y: SwissArray[T]): SwissArray[T] =
  like(result,y)
  for idx in 0..<y.len: result[idx] = x/y[idx]
proc divide[T](x: SwissArray[T]; y: T): SwissArray[T] =
  like(result,x)
  for idx in 0..<x.len: result[idx] = x[idx]/y
proc divide[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = x[idx]/y

template tensor*[V:static[int],T](x: var SwissTensor[V,T]; shape: array): int =
  var size = 1
  x.shape = shape
  for idx in countdown(V-1,0): 
    x.strides[idx] = size
    size *= shape[idx]
  size

proc like*(x: var SwissTensor; y: SwissTensor) =
  let shape = y.shape
  discard tensor(x,shape)
  like(x.storage,y.storage)

proc index[V:static[int],T](x: SwissTensor[V,T]; i: array[V,int]): int {.inline.} =
  result = x.offset
  for idx in 0..<V: 
    result += x.strides[idx]*i[idx]

template `[]`*[V:static[int],T](x: var SwissTensor[V,T]; idx: array[V,int]): var T = 
  x.storage[x.index(idx)]

template active(x: SwissTensor) = assert(x.storage.data != nil)

template conformable(x,y: SwissTensor) = conformable(x.storage,y.storage)

template comparable(x,y: SwissTensor) =
  conformable(x,y)
  assert(x.shape == y.shape)

template operable(x,y: SwissTensor) =
  active(x)
  active(y)
  comparable(x,y)

template `:=`*(x: var SwissTensor; y: SwissTensor) =
  comparable(x,y)
  `=copy`(x.storage,y.storage)
template `:=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) = 
  x.storage := y
template `:=`*[T](x: var T; y: T) = 
  x = y

proc size*[V:static[int],T](tensor: SwissTensor[V,T]): int = tensor.storage.len

proc newTensor*[V:static[int],T](shape: array[V,int]; t: typedesc[T]): SwissTensor[V,T] =
  result.storage := new(tensor(result,shape),t)

proc newTensor*[V:static[int]](
    shape: array[V,int]; 
    x: int32 | int64 | float32 | float64
  ): SwissTensor[V,type(x)] =
  result.storage := new(tensor(result,shape),type(x))
  for idx in 0..<result.storage.len: result.storage[idx] = x

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
proc `+=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) =
  x.storage.add(y)

proc `-`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := subtract(x.storage,y.storage)
proc `-=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.subtrace(y.storage)

proc `-`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := subtract(x,y.storage)
proc `-`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := subtract(x.storage,y)
proc `-=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) =
  x.storage.subtract(y)

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
proc `*=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) =
  x.storage.multiply(y)

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
proc `/=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) =
  x.storage.divide(y)

if isMainModule:
  var 
    ts1 = newTensor([2,2],float)
    ts2 = newTensor([2,2],float)
    ts3 = newTensor([2,2],float)
  ts2 := ts1
  ts2[[0,1]] := 1.0
  ts1[[0,0]] := ts2[[0,1]]
  echo ts1[[0,0]]
  echo ts1[[0,0]]
  ts2[[0,0]] := 1.0
  ts2[[1,1]] := 2.0
  ts3[[0,0]] := 3.0
  ts3[[1,1]] := 4.0
  ts1 := ts2 + ts3
  echo ts1[[0,0]]," ",ts1[[0,1]]," ",ts1[[1,0]]," ",ts1[[1,1]]
  ts1 := ts2 - ts3
  echo ts1[[0,0]]," ",ts1[[0,1]]," ",ts1[[1,0]]," ",ts1[[1,1]]
  ts1 := ts2*ts3
  echo ts1[[0,0]]," ",ts1[[0,1]]," ",ts1[[1,0]]," ",ts1[[1,1]]
  ts3[[0,1]] := -1.0
  ts3[[1,0]] := -1.0
  ts1 := ts2/ts3
  echo ts1[[0,0]]," ",ts1[[0,1]]," ",ts1[[1,0]]," ",ts1[[1,1]]
  ts1 := float(1.0)
  ts1 += ts2