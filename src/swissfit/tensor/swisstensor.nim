# SwissTensor type
#
# Notes: 
#  - SwissTensor type modeled after Arraymancer Tensor
#    - https://github.com/mratsim/Arraymancers

type
  SwissTensor*[V:static[int],T] = object
    offset: int
    shape: array[V,int]
    strides: array[V,int]
    storage*: SwissArray[T]

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