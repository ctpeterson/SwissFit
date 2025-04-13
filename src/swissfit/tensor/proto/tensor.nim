# SwissTensor type
#
# Notes: 
#  - SwissTensor type modeled after Arraymancer Tensor
#    - https://github.com/mratsim/Arraymancers

import ../array/swissarray
import ../sequence/swissseq

type
  SwissTensor*[T] = object
    offset: int
    strides: SwissSeq[int]
    shape*: SwissSeq[int]
    storage*: SwissArray[T]

# Allocate resources to SwissTensor
proc size*[T](tensor: SwissTensor[T]): int = tensor.storage.len
proc newSwissTensor*[T](x: var SwissTensor[T]; shape: array): int {.inline.} =
  result = 1
  x.shape = newSwissSeq(shape)
  x.strides.like(x.shape)
  for idx in countdown(x.shape.len-1,0): 
    x.strides[idx] = result
    result *= shape[idx]
proc newSwissTensor*[T](x: var SwissTensor[T]; shape: SwissSeq[int]): int {.inline.} =
  result = 1
  x.shape := shape
  x.strides.like(x.shape)
  for idx in countdown(x.shape.len-1,0): 
    x.strides[idx] = result
    result *= shape[idx]
proc like*(x: var SwissTensor; y: SwissTensor) =
  let shape = y.shape
  discard newSwissTensor(x,shape)
  like(x.storage,y.storage)

# Accessors and assignments
proc index*[T](x: SwissTensor[T]; i: SwissSeq[int]): int {.inline.} =
  assert(i.len == x.shape.len)
  result = x.offset
  for idx in 0..<i.len: result += x.strides[idx]*i[idx]
proc index[V:static[int],T](x: SwissTensor[T]; i: array[V,int]): int {.inline.} =
  x.storage[x.index(newSwissSeq(i))]
proc index[T](x: SwissTensor[T]; i: seq[int]): int {.inline.} =
  assert(i.len == x.shape.len)
  result = x.offset
  for idx in 0..<i.len: result += x.strides[idx]*i[idx]
template `:=`*(x: var SwissTensor; y: SwissTensor) =
  assert(x.shape == y.shape)
  conformable(x.storage,y.storage)
  `=copy`(x.storage,y.storage)
template `:=`*[T](x: var SwissTensor[T]; y: T) = (x.storage := y)
template `<-`*[T](x: SwissTensor[T]; y: T) = (x := y)
template `[]`*[V:static[int],T](x: var SwissTensor[T]; idx: array[V,int]): var T = 
  x.storage[x.index(idx)]
#template `[]`*[T](x: var SwissTensor[T]; idx: SwissSeq[int]): var T = 
#  x.storage[x.index(idx)]
#template `[]`*[V:static[int],T](x: SwissTensor[T]; idx: array[V,int]): T = 
#  x.storage[x.index(idx)]
#template `[]`*[T](x: var SwissTensor[T]; idx: seq[int]): var T = 
#  x.storage[x.index(idx)]
#template `[]`*[T](x: SwissTensor[T]; idx: seq[int]): T = x.storage[x.index(idx)]

# Constructors
proc newTensor*[V:static[int],T](
    shape: array[V,int]; 
    t: typedesc[T]
  ): SwissTensor[T] = (result.storage := new[T](newSwissTensor(result,shape)))
proc newTensor*[T](shape: SwissSeq[int]; t: typedesc[T]): SwissTensor[T] = 
  result.storage := new[T](newSwissTensor(result,shape))
proc newTensor*[T](shape: SwissSeq[int]): SwissTensor[T] = 
  result.storage := new[T](newSwissTensor(result,shape))
proc newTensor*[V:static[int]](
    s: array[V,int]; 
    x: float32 | float64
  ): SwissTensor[type(x)] =
  result.storage := new[type(x)](newSwissTensor(result,s))
  for idx in 0..<result.storage.len: result.storage[idx] = x
proc newTensor*[V:static[int],T](
    shape: array[V,int]; 
    x: SwissArray[T]
  ): SwissTensor[type(x)] =
  discard newSwissTensor(result,shape)
  result.storage := x
proc newTensor*[T](shape: array): SwissTensor[T] =
  discard newSwissTensor(result,shape)