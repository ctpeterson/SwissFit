# SwissArray type
#
# Notes: 
#  - SwissArray type modeled after custom sequence example in Nim documentation
#    - https://nim-lang.org/docs/destructors.html

import ../simd/[simd]

type
  SwissArray*[T] = object
    len,cap: int
    data*: ptr UncheckedArray[T]

proc `=destroy`[T](x: SwissArray[T]) = 
  if x.data != nil: 
    for idx in 0..<x.len: `=destroy`(x.data[idx])
    dealloc(x.data)

proc `=trace`*[T](x: var SwissArray[T]; env: pointer) =
  if x.data != nil: (for idx in 0..<x.len: `=trace`(x.data[idx],env))

proc `=wasMoved`*[T](x: var SwissArray[T]) = (x.data = nil)

proc `=sink`*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  `=destroy`(x); x.len = y.len
  x.cap = y.cap 
  x.data = y.data

proc `=copy`*[T](x: var SwissArray[T]; y: SwissArray[T]) =
  if x.data == y.data: return
  x.len = y.len
  x.cap = y.cap
  if y.data != nil:
    x.data = cast[typeof(x.data)](alloc(x.cap*sizeof(T)))
    for idx in 0..<x.len: x.data[idx] = y.data[idx]

proc `=dup`*[T](x: SwissArray[T]): SwissArray[T] {.nodestroy.} =
  result = SwissArray[T](len: x.len, cap: x.cap, data: nil)
  if x.data != nil:
    result.data = cast[typeof(result.data)](alloc(result.cap*sizeof(T)))
    for idx in 0..<result.len: result.data[idx] = `=dup`(x.data[idx])

proc append*[T](x: var SwissArray[T]; y: sink T) =
  if x.len >= x.cap:
    x.cap = max(x.len + 1, 2*x.cap)
    x.data = cast[typeof(x.data)](realloc(x.data, x.cap*sizeof(T)))
  x.data[x.len] = y; inc x.len;

template allocate[T](x: var SwissArray[T]; len: int; t: typedesc[T]) =
  x = SwissArray[T](len: len, cap: len)
  x.data = cast[typeof(x.data)](alloc(len*sizeof(T)))

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

proc add*[T](x,y: SwissArray[T]): SwissArray[T] =
  like(result,x)
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