# SwissTensor type
#
# Notes: 
#  - SwissTensor type modeled after Arraymancer Tensor
#    - https://github.com/mratsim/Arraymancers

import arraytype

type
  SwissTensor*[V:static[int],T] = object
    offset: int
    strides: array[V,int]
    shape*: array[V,int]
    storage*: SwissArray[T]

# Allocate resources to SwissTensor
proc size*[V:static[int],T](tensor: SwissTensor[V,T]): int = tensor.storage.len
proc new*[V:static[int],T](x: var SwissTensor[V,T]; shape: array): int {.inline.} =
  result = 1
  x.shape = shape
  for idx in countdown(V-1,0): 
    x.strides[idx] = result
    result *= shape[idx]
proc like*(x: var SwissTensor; y: SwissTensor) =
  let shape = y.shape
  discard new(x,shape)
  like(x.storage,y.storage)

# Accessors and assignments
proc index[V:static[int],T](x: SwissTensor[V,T]; i: array[V,int]): int {.inline.} =
  result = x.offset
  for idx in 0..<V: result += x.strides[idx]*i[idx]
template `:=`*(x: var SwissTensor; y: SwissTensor) =
  assert(x.shape == y.shape)
  conformable(x.storage,y.storage)
  `=copy`(x.storage,y.storage)
template `:=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) = (x.storage := y)
template `<-`*[V:static[int],T](x: SwissTensor[V,T]; y: T) = (x := y)
template `[]`*[V:static[int],T](x: var SwissTensor[V,T]; idx: array[V,int]): var T = 
  x.storage[x.index(idx)]

# Constructors
proc newTensor*[V:static[int],T](shape: array[V,int]; t: typedesc[T]): SwissTensor[V,T] =
  result.storage := new(new(result,shape),t)
proc newTensor*[V:static[int]](
    s: array[V,int]; 
    x: float32 | float64
  ): SwissTensor[V,type(x)] =
  result.storage := new(new(result,s),type(x))
  for idx in 0..<result.storage.len: result.storage[idx] = x
proc newTensor*[V:static[int],T](
    s: array[V,int]; 
    x: SwissArray[T]
  ): SwissTensor[V,type(x)] =
  discard new(result,s)
  result.storage := x