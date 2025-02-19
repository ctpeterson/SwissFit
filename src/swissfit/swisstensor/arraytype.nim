# SwissArray type
#
# Notes: 
#  - SwissArray type modeled after custom sequence example in Nim documentation
#    - https://nim-lang.org/docs/destructors.html
#  - More information about manual Nim memory management can be found in 
#    Andreas Rumpf's "Mastering Nim"

import ../simd/[simd]

type
  SwissArray*[T] = object
    len,cap: int
    data*: ptr UncheckedArray[T]

# Free SwissArray resources
proc `=destroy`[T](x: SwissArray[T]) = 
  if x.data != nil: 
    for idx in 0..<x.len: `=destroy`(x.data[idx])
    dealloc(x.data)

# Tells `=destroy` that SwissArray resources were moved
proc `=wasMoved`*[T](x: var SwissArray[T]) = (x.data = nil)

# Move SwissArray resources to target & tell `=destroy` not to free target
proc `=sink`*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  `=destroy`(x) 
  `=wasMoved`(x)
  (x.len,x.cap,x.data) = (y.len,y.cap,y.data)

# Create duplicate of SwissArray in memory
proc `=dup`*[T](y: SwissArray[T]): SwissArray[T] = 
  result = SwissArray[T](len: y.len, cap: y.len, data: nil)
  if y.data != nil:
    result.data = cast[typeof(y.data)](alloc(y.cap*sizeof(T)))
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
proc add*[T](x: var SwissArray[T]; y: sink T) =
  if x.len >= x.cap:
    x.cap = max(x.len + 1, 2*x.cap)
    x.data = cast[typeof(x.data)](realloc(x.data, x.cap*sizeof(T)))
  x.data[x.len] = y; inc x.len;

# Allocate resources to SwissArray
template allocate[T](x: var SwissArray[T]; len: int; t: typedesc[T]) =
  x = SwissArray[T](len: len, cap: len)
  x.data = cast[typeof(x.data)](alloc(len*sizeof(T)))

# SwissArray constructors
proc new*[T](xs: seq[T]): SwissArray[T] =
  result.allocate(x.len,type(t))
  for idx in 0..<result.len: result.data[idx] = xs[idx]
proc new*[T](len: int; t: typedesc[T]): SwissArray[T] = result.allocate(len,t)
proc new*[T](len: int; x: T): SwissArray[T] =
  result.allocate(len,type(x))
  for idx in 0..<len: result.data[idx] = x
proc new*[T](len: int; x: SwissArray[T]): SwissArray[T] =
  assert(x.data != nil)
  result.allocate(len,type(x[0]))
  for idx in 0..<len: result.data[idx] = x[idx]

proc isUnit*[T](x: SwissArray[T]): bool {.inline.} =
  result = true
  for idx in 0..<x.len: (if x[idx] != T(1.0): result = false)
proc isNull*[T](x: SwissArray[T]): bool {.inline.} =
  result = true
  for idx in 0..<x.len: (if x[idx] != T(0.0): result = false)

template like*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  x = SwissArray[T](len: y.len, cap: y.len)
  x.data = cast[typeof(x.data)](alloc(x.cap*sizeof(T)))

proc conformable*[T](x,y: SwissArray[T]) =
  assert(x.len == y.len) 
  assert(x.cap == y.cap)

template `<-`*[T](x: SwissArray[T]; y: T) = (x := y)
template `<-`*[S,T](x: S; y: T) = (x = S(y))

template `[]`*[T](x: SwissArray[T]; idx: Natural): lent T =
  assert idx < x.len
  x.data[idx]
template `[]=`*[T](x: var SwissArray[T]; idx: Natural; y: sink T) =
  assert idx < x.len 
  x.data[idx] = y
proc len*[T](x: SwissArray[T]): int {.inline.} = x.len

proc `:=`*[T](x: var SwissArray[T]; y: T) =
  for idx in 0..<x.len: x[idx] = y
template `:=`*[T](x: var T; y: T) = (x = y)