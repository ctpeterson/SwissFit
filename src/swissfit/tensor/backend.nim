# nim c -d:x86 -d:AVX512 -d:release --passC:-Ofast --threads:on --app:lib --out:backend.so backend
import nimpy
import ../simd/[swisssimd]
import ../sequence/[swissseq]
import ../array/[swissarray]

template newPrimitiveOperation(O,B,F: untyped) =
  proc `O F`(y: `"SwissTensor" F`; xa,xb: `"SwissTensor" F`) =
    assert(xa.shape == xb.shape)
    for idx in countup(0,xa.vcap-1,xa.vlen):
      y.storage.store(O(load(xa.storage,idx),load(xb.storage,idx)),idx)
    for idx in countup(xa.vcap,xa.len-1,1):
      y.storage[idx] = B(xa.storage[idx],xb.storage[idx])

template define(F: untyped) {.dirty.} =
  type
    `"SwissTensor" F`* = ref object of PyNimObjectExperimental
      offset,len*,vlen*,cap*,vcap*: int
      strides,shape: SwissSeq[int]
      storage*: ptr UncheckedArray[`"float" F`]
  
  proc `"newSwissTensor" F`(shape: SwissSeq[int]): `"SwissTensor" F` =
    let (alen,slen,vlen) = (shape.product,shape.len,vlen[`"float" F`]())
    var strides = 1
    result = `"SwissTensor" F`(
      offset: 0,
      len: alen,
      cap: alen,
      vlen: vlen,
      vcap: (alen div vlen)*vlen,
      shape: shape,
      strides: newSwissSeq[int](slen)
    )
    for idx in countdown(slen-1,0):
      result.strides[idx] = strides
      strides *= result.shape[idx]
    result.storage = cast[ptr UncheckedArray[`"float" F`]](
      aligned_alloc[`"float" F`](alen*sizeof(`"float" F`))
    )
  
  proc `"linearize" F`(x: `"SwissTensor" F`): seq[`"float" F`] =
    result = newSeq[`"float" F`](x.len)
    for idx in 0..<x.len: result[idx] = x.storage[idx]

  proc `"index" F`(x: `"SwissTensor" F`; coord: SwissSeq[int]): int =
    assert(x.shape.len == coord.len)
    result = x.offset
    for idx in 0..<coord.len: result += x.strides[idx]*coord[idx]

  newPrimitiveOperation(add,`+`,F)
  newPrimitiveOperation(sub,`-`,F)

  proc `"add" F`(xa,xb: `"SwissTensor" F`): `"SwissTensor" F` =
    assert(xa.shape == xb.shape)
    result = `"newSwissTensor" F`(xa.shape)
    `"add" F`(result,xa,xb)

  proc `"sub" F`(xa,xb: `"SwissTensor" F`): `"SwissTensor" F` =
    assert(xa.shape == xb.shape)
    result = `"newSwissTensor" F`(xa.shape)
    `"sub" F`(result,xa,xb)

define(32)
define(64)

proc linearizePy*(x: SwissTensor32): seq[float32] {.exportpy.} = linearize32(x)
proc linearizePy*(x: SwissTensor64): seq[float64] {.exportpy.} = linearize64(x)

proc getPy*(x: SwissTensor32; coord: seq[int]): float32 {.exportpy.} = 
  x.storage[x.index32(newSwissSeq(coord))]
proc getPy*(x: SwissTensor64; coord: seq[int]): float64 {.exportpy.} = 
  x.storage[x.index64(newSwissSeq(coord))]

proc setPy*(x: SwissTensor32; coord: seq[int]; y: float32) {.exportpy.} = 
  `=copy`(x.storage[x.index32(newSwissSeq(coord))],y)
proc setPy*(x: SwissTensor64; coord: seq[int]; y: float64) {.exportpy.} = 
  `=copy`(x.storage[x.index64(newSwissSeq(coord))],y)

proc newSwissTensorPy32(shape: seq[int]): SwissTensor32 {.exportpy.} =
  newSwissTensor32(newSwissSeq(shape))
proc newSwissTensorPy64(shape: seq[int]): SwissTensor64 {.exportpy.} =
  newSwissTensor64(newSwissSeq(shape))

proc addPy(x,y: SwissTensor32): SwissTensor32 {.exportpy.} = add32(x,y)
proc addPy(x,y: SwissTensor64): SwissTensor64 {.exportpy.} = add64(x,y)

proc subPy(x,y: SwissTensor32): SwissTensor32 {.exportpy.} = sub32(x,y)
proc subPy(x,y: SwissTensor64): SwissTensor64 {.exportpy.} = sub64(x,y)

proc lenPy(x: SwissTensor32): int {.exportpy.} = x.len
proc lenPy(x: SwissTensor64): int {.exportpy.} = x.len