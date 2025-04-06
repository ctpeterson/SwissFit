# nim c -d:x86 -d:AVX --mm:orc --noMain --app:lib --out:backend.so --threads:on backend
import nimpy
import swisstensor

# Python-facing tensor type
type 
  AnyTensor[T] = concept x
    x.tensor is SwissTensor[T]
  SwissTensor32* = ref object of PyNimObjectExperimental
    tensor*: SwissTensor[float32]
  SwissTensor64* = ref object of PyNimObjectExperimental
    tensor*: SwissTensor[float64]

proc newTensor32*(shape: seq[int]): SwissTensor32 {.exportpy.} =
  result = SwissTensor32(tensor: newTensor[float32](newSwissSeq(shape)))
proc newTensor64*(shape: seq[int]): SwissTensor64 {.exportpy.} =
  result = SwissTensor64(tensor: newTensor[float64](newSwissSeq(shape)))

proc element*(x: SwissTensor32, index: seq[int]): float32 {.exportpy.} = 
  x.tensor.storage[x.tensor.index(newSwissSeq(index))]
proc element*(x: SwissTensor64, index: seq[int]): float64 {.exportpy.} = 
  x.tensor.storage[x.tensor.index(newSwissSeq(index))]

proc getLinear*(x: SwissTensor32): seq[float32] {.exportpy.} =
  let len = x.tensor.storage.len
  result = newSeq[float32](len)
  for idx in 0..<len: result[idx] = x.tensor.storage[idx]
proc getLinear*(x: SwissTensor64): seq[float64] {.exportpy.} =
  let len = x.tensor.storage.len
  result = newSeq[float64](len)
  for idx in 0..<len: result[idx] = x.tensor.storage[idx]

proc setElement*(x: SwissTensor32, index: seq[int], value: float64) {.exportpy.} = 
  x.tensor.storage[x.tensor.index(newSwissSeq(index))] = value
proc setElement*(x: SwissTensor64, index: seq[int], value: float64) {.exportpy.} = 
  x.tensor.storage[x.tensor.index(newSwissSeq(index))] = value

proc set*(x: SwissTensor32; y: SwissTensor32) {.exportpy.} = (x.tensor := y.tensor)
proc set*(x: SwissTensor64; y: SwissTensor64) {.exportpy.} = (x.tensor := y.tensor)

proc add*(x,y: SwissTensor32): SwissTensor32 {.exportpy.} = 
  var r {.noinit.} = x.tensor + y.tensor
  result = SwissTensor32(tensor: newTensor[float32](x.tensor.shape))
  result.tensor := r
proc add*(x,y: SwissTensor64): SwissTensor64 {.exportpy.} =
  var r {.noinit.} = x.tensor + y.tensor
  result = SwissTensor64(tensor: newTensor[float64](x.tensor.shape))
  result.tensor := r

proc sub*(x,y: SwissTensor32): SwissTensor32 {.exportpy.} = 
  var r {.noinit.} = x.tensor - y.tensor
  result = SwissTensor32(tensor: newTensor[float32](x.tensor.shape))
  result.tensor := r
proc sub*(x,y: SwissTensor64): SwissTensor64 {.exportpy.} =
  var r {.noinit.} = x.tensor - y.tensor
  result = SwissTensor64(tensor: newTensor[float64](x.tensor.shape))
  result.tensor := r

proc mul*(x,y: SwissTensor32): SwissTensor32 {.exportpy.} = 
  var r {.noinit.} = x.tensor*y.tensor
  result = SwissTensor32(tensor: newTensor[float32](x.tensor.shape))
  result.tensor := r
proc mul*(x,y: SwissTensor64): SwissTensor64 {.exportpy.} =
  var r {.noinit.} = x.tensor*y.tensor
  result = SwissTensor64(tensor: newTensor[float64](x.tensor.shape))
  result.tensor := r

proc divd*(x,y: SwissTensor32): SwissTensor32 {.exportpy.} = 
  var r {.noinit.} = x.tensor/y.tensor
  result = SwissTensor32(tensor: newTensor[float32](x.tensor.shape))
  result.tensor := r
proc divd*(x,y: SwissTensor64): SwissTensor64 {.exportpy.} =
  var r {.noinit.} = x.tensor/y.tensor
  result = SwissTensor64(tensor: newTensor[float64](x.tensor.shape))
  result.tensor := r