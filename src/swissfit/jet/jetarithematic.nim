import jettypes,jetattributes
import ../tensor/[swisstensor]

proc `+`*[T](a,b: Swiss1Jet[T]): Swiss1Jet[T] =
  result = (a.x + b.x).prolong(pushforward = pushforward(a,b), derived = true)
  case result.pushforward:
    of true: result.dx = a.dx + b.dx
    of false:
      result.back = proc(x: Swiss1Jet[T]) =
        if not a.pushforward: a.dx += x.dx
        if not b.pushforward: b.dx += x.dx
      result.register(a)
      result.register(b)

proc `+`*[T](a: T; b: Swiss1Jet[T]): Swiss1Jet[T] =
  result = (a + b.x).prolong(pushforward = b.pushforward, derived = true)
  case result.pushforward:
    of true: result.dx = b.dx
    of false:
      result.back = proc(x: Swiss1Jet[T]) = (if not b.pushforward: b.dx += x.dx)
      result.register(b)
proc `+`*[T](a: Swiss1Jet[T]; b: T): Swiss1Jet[T] =
  result = (a.x + b).prolong(pushforward = a.pushforward, derived = true)
  case result.pushforward:
    of true: result.dx = a.dx
    of false:
      result.back = proc(x: Swiss1Jet[T]) = (if not a.pushforward: a.dx += x.dx)
      result.register(a)

proc `-`*[T](a,b: Swiss1Jet[T]): Swiss1Jet[T] =
  result = (a.x - b.x).prolong(pushforward = pushforward(a,b), derived = true)
  case result.pushforward:
    of true: result.dx = a.dx - b.dx
    of false:
      result.back = proc(x: Swiss1Jet[T]) =
        if not a.pushforward: a.dx += x.dx
        if not b.pushforward: b.dx -= x.dx
      result.register(a)
      result.register(b)

proc `-`*[T](a: T; b: Swiss1Jet[T]): Swiss1Jet[T] =
  result = (a - b.x).prolong(pushforward = b.pushforward, derived = true)
  case result.pushforward:
    of true: result.dx = -b.dx
    of false:
      result.back = proc(x: Swiss1Jet[T]) = (if not b.pushforward: b.dx -= x.dx)
      result.register(b)
proc `-`*[T](a: Swiss1Jet[T]; b: T): Swiss1Jet[T] =
  result = (a.x - b).prolong(pushforward = a.pushforward, derived = true)
  case result.pushforward:
    of true: result.dx = a.dx
    of false:
      result.back = proc(x: Swiss1Jet[T]) = (if not a.pushforward: a.dx += x.dx)
      result.register(a)

proc `*`*[T](a,b: Swiss1Jet[T]): Swiss1Jet[T] =
  result = (a.x * b.x).prolong(pushforward = pushforward(a,b), derived = true)
  case result.pushforward:
    of true: result.dx = a.dx*b.x + a.x*b.dx
    of false:
      result.back = proc(x: Swiss1Jet[T]) =
        if not a.pushforward: a.dx += b.x*x.dx
        if not b.pushforward: b.dx += a.x*x.dx
      result.register(a)
      result.register(b)

proc `*`*[T](a: T; b: Swiss1Jet[T]): Swiss1Jet[T] =
  result = (a * b.x).prolong(pushforward = b.pushforward, derived = true)
  case result.pushforward:
    of true: result.dx = a*b.dx
    of false:
      result.back = proc(x: Swiss1Jet[T]) = (if not b.pushforward: b.dx += a*x.dx)
      result.register(b)
proc `*`*[T](a: Swiss1Jet[T]; b: T): Swiss1Jet[T] =
  result = (a.x * b).prolong(pushforward = a.pushforward, derived = true)
  case result.pushforward:
    of true: result.dx = a.dx*b
    of false:
      result.back = proc(x: Swiss1Jet[T]) = (if not a.pushforward: a.dx += b*x.dx)
      result.register(a)

proc `/`*[T](a,b: Swiss1Jet[T]): Swiss1Jet[T] =
  result = (a.x / b.x).prolong(pushforward = pushforward(a,b), derived = true)
  case result.pushforward:
    of true: result.dx = a.dx/b.x - a.x*b.dx/(b.x*b.x)
    of false:
      result.back = proc(x: Swiss1Jet[T]) =
        if not a.pushforward: a.dx += x.dx/b.x
        if not b.pushforward: b.dx -= a.x*x.dx/(b.x*b.x)
      result.register(a)
      result.register(b)

proc `/`*[T](a: T; b: Swiss1Jet[T]): Swiss1Jet[T] =
  result = (a / b.x).prolong(pushforward = b.pushforward, derived = true)
  case result.pushforward:
    of true: result.dx = (-a)*b.dx/(b.x*b.x)
    of false:
      result.back = proc(x: Swiss1Jet[T]) =
        if not b.pushforward: b.dx -= a*x.dx/(b.x*b.x)
      result.register(b)
proc `/`*[T](a: Swiss1Jet[T]; b: T): Swiss1Jet[T] =
  result = (a.x / b).prolong(pushforward = a.pushforward, derived = true)
  case result.pushforward:
    of true: result.dx = a.dx/b
    of false:
      result.back = proc(x: Swiss1Jet[T]) = (if not a.pushforward: a.dx += x.dx/b)
      result.register(a)