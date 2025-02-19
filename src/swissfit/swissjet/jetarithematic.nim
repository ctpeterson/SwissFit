import jettypes,jetattributes
import ../swisstensor/[swisstensor]

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

if isMainModule:
  var x = 1.0
  var 
    jx = x.prolong()
    jxc = [2].newTensor(x)
    jxv = [2].vectorize(jx)
    jxv2 = jxv + jxv
    jxv3 = jxv - jxv
    jxv4 = jxv*jxv
    jxv5 = jxv/jxv
    jxv6 = jxv*jxv + jxv
  discard jxv + jxc
  discard jxc + jxv
  discard jxv - jxc
  discard jxc - jxv
  discard jxv * jxc
  discard jxc * jxv
  discard jxv / jxc
  discard jxc / jxv
  var 
    jy = x.prolong(pushforward = false)
    jyc = [2].newTensor(x)
    jyv = [2].vectorize(jy)
    jyv2 = jyv + jyv
    jyv3 = jyv - jyv
    jyv4 = jyv*jyv
    jyv5 = jyv/jyv
    jvy6 = jyv*jyv + jyv
  echo "pullback:"
  jyv2.pullback()
  echo jyv.dx[[0]]
  jyv3.pullback()
  echo jyv.dx[[0]]
  jyv4.pullback()
  echo jyv.dx[[0]]
  jyv5.pullback()
  echo jyv.dx[[0]]
  jvy6.pullback()
  echo jyv.dx[[0]]
  echo "pushforward:"
  discard jyv + jyc
  discard jyc + jyv
  discard jyv - jyc
  discard jyc - jyv
  discard jyv * jyc
  discard jyc * jyv
  discard jyv / jyc
  discard jyc / jyv
  echo jxv2.dx[[0]]
  echo jxv3.dx[[0]]
  echo jxv4.dx[[0]]
  echo jxv5.dx[[0]]
  echo jxv6.dx[[0]]