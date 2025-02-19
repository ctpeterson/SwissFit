import arraytype
import ../simd/[simd]

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