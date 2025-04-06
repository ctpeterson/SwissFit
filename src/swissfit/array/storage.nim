## SwissArray type
##
## This is a SIMD-vectorized linear array supporting single- and double-
## precision floating-point arithematic
##
## Notes: 
##  - SwissArray type modeled after custom sequence example in Nim documentation
##    - https://nim-lang.org/docs/destructors.html
##  - More information about manual Nim memory management can be found in 
##    Andreas Rumpf's "Mastering Nim"
##  - Requires c++15 or newer (aligned_alloc)
##  - SwissSeq modelled after dynamic stack array in Arraymaner 
##    (https://github.com/mratsim/Arraymancers)

import ../simd/[swisssimd]

{.pragma: stdlib, header: "<stdlib.h>".}

type
  Index = Natural or BackwardsIndex
  HeapAlloc = object
  SwissArray*[T] = object
    len,vlen,cap,vcap: int
    data*: ptr UncheckedArray[T]

# Allocate memory to SIMD-aligned vector
proc aligned_alloc(align,size: int): pointer {.tags: [HeapAlloc], importc, stdlib.}
proc aligned_alloc[T](size: int): pointer =
  assert((T is float32) or (T is float64))
  if T is float32: result = aligned_alloc(VLENF,size)
  if T is float64: result = aligned_alloc(VLEND,size)

# Deallocate memory assigned to SIMD-aligned vector
proc aligned_free(p: pointer) {.tags:[HeapAlloc], importc: "free", stdlib.}

# Reallocate memory assigned to SIMD-aligned vector
proc aligned_realloc[T](x: var SwissArray[T]; newSize: int): pointer =
  let 
    oldSize = x.len*sizeof(T)
    (oldAddr,newAddr) = (addr x.data,aligned_alloc[T](newSize))
  copyMem(newAddr,oldAddr,min(oldSize,newSize))
  aligned_free(addr x.data[0])
  result = newAddr

# Free SwissArray resources
proc `=destroy`[T](x: SwissArray[T]) = 
  if x.data != nil: 
    for idx in 0..<x.len: `=destroy`(x.data[idx])
    aligned_free(addr x.data[0])

# Tells `=destroy` that SwissArray resources were moved
proc `=wasMoved`*[T](x: var SwissArray[T]) = (x.data = nil)

# Move SwissArray resources to target & tell `=destroy` not to free target
proc `=sink`*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  `=destroy`(x) 
  `=wasMoved`(x)
  (x.len,x.cap,x.vcap,x.data) = (y.len,y.cap,y.vcap,y.data)

# Create duplicate of SwissArray in memory
proc `=dup`*[T](y: SwissArray[T]): SwissArray[T] = 
  result = SwissArray[T](len: y.len, cap: y.len, vcap: y.vcap, data: nil)
  if y.data != nil:
    result.data = cast[ptr UncheckedArray[T]](aligned_alloc[T](y.cap*sizeof(T)))
    for idx in 0..<y.len: result.data[idx] = y.data[idx]

# Cover scenarious operation can't be transformed into `=sink`
proc `=copy`*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  if x.data == y.data: return
  `=destroy`(x) 
  `=wasMoved`(x)
  x = `=dup`(y)

# Support for Nim's cycle collator (--m:orc)
proc `=trace`*[T](x: var SwissArray[T]; env: pointer) =
  if x.data != nil: (for idx in 0..<x.len: `=trace`(x.data[idx],env))

# Add element to SwissArray
proc append*[T](x: var SwissArray[T]; y: sink T) =
  if x.len >= x.cap:
    x.cap = max(x.len + 1, 2*x.cap)
    x.data = cast[ptr UncheckedArray[T]](aligned_realloc(x.data, x.cap*sizeof(T)))
  x.data[x.len] = y
  inc x.len
  x.vcap = (x.len div vlen[T]())*vlen[T]()

# Allocate resources to SwissArray
proc allocate[T](x: var SwissArray[T]; len: int) =
  assert((T is float32) or (T is float64))
  x = SwissArray[T](len: len, cap: len)
  x.data = cast[ptr UncheckedArray[T]](aligned_alloc[T](len*sizeof(T)))
  x.vcap = (x.len div vlen[T]())*vlen[T]()

# SwissArray constructors
proc new*[T](xs: seq[T]): SwissArray[T] =
  allocate[T](result,xs.len)
  for idx in 0..<result.len: result.data[idx] = xs[idx]
proc new*[T](len: int; x: T): SwissArray[T] =
  allocate[T](result,len)
  for idx in 0..<len: result.data[idx] = x
proc new*[T](len: int; x: SwissArray[T]): SwissArray[T] =
  assert(x.data != nil)
  allocate[T](result,len)
  for idx in 0..<len: result.data[idx] = x[idx]
proc new*[T](len: int): SwissArray[T] = allocate[T](result,len)

proc isUnit*[T](x: SwissArray[T]): bool {.inline.} =
  result = true
  for idx in 0..<x.len: (if x[idx] != T(1.0): result = false)
proc isNull*[T](x: SwissArray[T]): bool {.inline.} =
  result = true
  for idx in 0..<x.len: (if x[idx] != T(0.0): result = false)

template like*[T](x: var SwissArray[T]; y: SwissArray[T]) = allocate[T](x,y.len)

proc conformable*[T](x,y: SwissArray[T]) =
  assert(x.len == y.len) 
  assert(x.cap == y.cap)

template `<-`*[T](x: SwissArray[T]; y: T) = (x := y)
template `<-`*[S,T](x: S; y: T) = (x = S(y))

proc len*[T](x: SwissArray[T]): int {.inline.} = x.len
proc vlen*[T](x: SwissArray[T]): int {.inline.} = x.vlen
proc vcap*[T](x: SwissArray[T]): int {.inline.} = x.vcap
proc index[T](x: SwissArray[T]; idx: Index): int =
  result = case idx is BackwardsIndex
    of true: x.len - int(idx)
    of false: int(idx)
template `[]`*[T](x: SwissArray[T]; idx: Index): lent T =
  assert idx < x.len
  x.data[x.index(idx)]
template `[]`*[T](x: var SwissArray[T]; idx: Index): lent T =
  assert idx < x.len
  x.data[x.index(idx)]
template `[]=`*[T](x: var SwissArray[T]; idx: Index; y: sink T) =
  assert idx < x.len 
  x.data[x.index(idx)] = y

proc `:=`*[T](x: var SwissArray[T]; y: T) = (for idx in 0..<x.len: x[idx] = y)
template `:=`*[T](x: var SwissArray[T]; y: SwissArray[T]) = `=copy`(x,y)

proc `$`*[T](x: SwissArray[T]): string =
  result = "[" & $x[0]
  for idx in 1..<x.len: result &= ", " & $x[idx]
  result &= "]"

if isMainModule:
  var t1 = new(@[1.0,2.0,3.0,4.0])
  echo "created array: ", $t1