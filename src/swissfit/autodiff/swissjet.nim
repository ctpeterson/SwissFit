# SwissJet type
# Represents 1-jet (dual number); able to do both
# forward-mode and reverse-mode automatic differentiation.
# Prologation "functor" (not quite a functor) takes types 
# "T" to their first-jet prolongation; elementary arithematic of 
# first-jet prolongation taken care of by operator overloading

import ../tensor/[swisstensor,swissarray]

type 
  SwissJet*[T] = ref object
    x,dx: T
    case forward: bool
      of true: discard
      of false:
        backprop: proc(x: SwissJet[T])
        stack: seq[SwissJet[T]]
    derived: bool

proc prolong*[T](x: T; forward: bool = true; derived: bool = false): SwissJet[T] =
  result = SwissJet[T](x: x, forward: forward, derived: derived)
  if not derived:
    case forward
      of true: result.dx = T(1.0)
      of false: 
        result.backprop = proc(x: SwissJet[T]) = discard
        result.dx = T(0.0)
  else: (if not forward: result.dx = T(1.0))
  if not forward: result.stack = newSeq[SwissJet[T]]()

proc newTensor*[V:static[int],T](
    shape: array[V,int]; 
    x: SwissJet[T]
  ): SwissTensor[V,SwissJet[T]] =
  result.storage := new(tensor(result,shape),SwissJet[T])
  for idx in 0..<result.storage.len: 
    result.storage[idx] = (x.x).prolong(forward = x.forward, derived = x.derived)

template forward(a,b): bool = ((a.forward) or (b.forward))

template register[T](x: var SwissJet[T]; y: SwissJet[T]) = 
  (if (y.dx != T(0.0)): y.dx = T(0.0)); x.stack.add(y);

proc `+`*[T](a,b: SwissJet[T]): SwissJet[T] =
  result = (a.x + b.x).prolong(forward = forward(a,b), derived = true)
  case result.forward:
    of true: result.dx = a.dx + b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += x.dx
        if not b.forward: b.dx += x.dx
      result.register(a)
      result.register(b)

proc `+`*[T](a: T; b: SwissJet[T]): SwissJet[T] =
  result = (a + b.x).prolong(forward = b.forward, derived = true)
  case result.forward:
    of true: result.dx = b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) = (if not b.forward: b.dx += x.dx)
      result.register(b)
proc `+`*[T](a: SwissJet[T]; b: T): SwissJet[T] =
  result = (a.x + b).prolong(forward = a.forward, derived = true)
  case result.forward:
    of true: result.dx = a.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) = (if not a.forward: a.dx += x.dx)
      result.register(a)

proc `-`*[T](a,b: SwissJet[T]): SwissJet[T] =
  result = (a.x - b.x).prolong(forward = forward(a,b), derived = true)
  case result.forward:
    of true: result.dx = a.dx - b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += x.dx
        if not b.forward: b.dx -= x.dx
      result.register(a)
      result.register(b)

proc `-`*[T](a: T; b: SwissJet[T]): SwissJet[T] =
  result = (a - b.x).prolong(forward = b.forward, derived = true)
  case result.forward:
    of true: result.dx = -b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) = (if not b.forward: b.dx -= x.dx)
      result.register(b)
proc `-`*[T](a: SwissJet[T]; b: T): SwissJet[T] =
  result = (a.x - b).prolong(forward = a.forward, derived = true)
  case result.forward:
    of true: result.dx = a.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) = (if not a.forward: a.dx += x.dx)
      result.register(a)

proc `*`*[T](a,b: SwissJet[T]): SwissJet[T] =
  result = (a.x * b.x).prolong(forward = forward(a,b), derived = true)
  case result.forward:
    of true: result.dx = a.dx*b.x + a.x*b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += b.x*x.dx
        if not b.forward: b.dx += a.x*x.dx
      result.register(a)
      result.register(b)

proc `*`*[T](a: T; b: SwissJet[T]): SwissJet[T] =
  result = (a * b.x).prolong(forward = b.forward, derived = true)
  case result.forward:
    of true: result.dx = a*b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) = (if not b.forward: b.dx += a*x.dx)
      result.register(b)
proc `*`*[T](a: SwissJet[T]; b: T): SwissJet[T] =
  result = (a.x * b).prolong(forward = a.forward, derived = true)
  case result.forward:
    of true: result.dx = a.dx*b
    of false:
      result.backprop = proc(x: SwissJet[T]) = (if not a.forward: a.dx += b*x.dx)
      result.register(a)

proc `/`*[T](a,b: SwissJet[T]): SwissJet[T] =
  result = (a.x / b.x).prolong(forward = forward(a,b), derived = true)
  case result.forward:
    of true: result.dx = a.dx/b.x - a.x*b.dx/(b.x*b.x)
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += x.dx/b.x
        if not b.forward: b.dx -= a.x*x.dx/(b.x*b.x)
      result.register(a)
      result.register(b)

proc `/`*[T](a: T; b: SwissJet[T]): SwissJet[T] =
  result = (a / b.x).prolong(forward = b.forward, derived = true)
  case result.forward:
    of true: result.dx = (-a)*b.dx/(b.x*b.x)
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not b.forward: b.dx -= a*x.dx/(b.x*b.x)
      result.register(b)
proc `/`*[T](a: SwissJet[T]; b: T): SwissJet[T] =
  result = (a.x / b).prolong(forward = a.forward, derived = true)
  case result.forward:
    of true: result.dx = a.dx/b
    of false:
      result.backprop = proc(x: SwissJet[T]) = (if not a.forward: a.dx += x.dx/b)
      result.register(a)

template sweep[T](graph: var seq[SwissJet[T]]; work: untyped) =
  var rank {.inject.} = 0
  while graph.len > 0:
    var node {.inject.} = graph.pop()
    work 
    inc rank
    for next in node.stack: graph.add(next)

proc sanitize[T](head: SwissJet[T]): seq[SwissJet[T]] =
  var graph = @[head]
  graph.sweep: 
    if (node.dx != T(1.0)) and (rank == 0): node.dx = T(1.0)
    if (node.dx != T(0.0)) and (rank > 0): node.dx = T(0.0)
  result = @[head]

proc backprop*[T](jx: SwissJet[T]) =
  assert(not jx.forward)
  var graph = jx.sanitize()
  graph.sweep: node.backprop(node)

proc backprop*[V:static[int],T](jx: SwissTensor[V,T]) =
  for idx in 0..<jx.size: backprop(jx.storage[idx])

if isMainModule:
  var x = 1.0
  var 
    jx = x.prolong()
    jxv = [2].newTensor(jx)
    jxv2 = jxv + jxv
    jxv3 = jxv - jxv
    jxv4 = jxv*jxv
    jxv5 = jxv/jxv
    jxv6 = jxv*jxv + jxv
  discard jxv + jx
  discard jx + jxv
  discard jxv - jx
  discard jx - jxv
  discard jxv * jx
  discard jx * jxv
  discard jxv / jx
  discard jx / jxv
  var 
    jy = x.prolong(forward = false)
    jyv = [2].newTensor(jy)
    jyv2 = jyv + jyv
    jyv3 = jyv - jyv
    jyv4 = jyv*jyv
    jyv5 = jyv/jyv
    jvy6 = jyv*jyv + jyv
  echo "reverse:"
  jyv2.backprop()
  echo jyv[[0]].dx
  jyv3.backprop()
  echo jyv[[0]].dx
  jyv4.backprop()
  echo jyv[[0]].dx
  jyv5.backprop()
  echo jyv[[0]].dx
  jvy6.backprop()
  echo jyv[[0]].dx
  echo "forward:"
  discard jyv + jy
  discard jy + jyv
  discard jyv - jy
  discard jy - jyv
  discard jyv * jy
  discard jy * jyv
  discard jyv / jy
  discard jy / jyv
  echo jxv2[[0]].dx
  echo jxv3[[0]].dx
  echo jxv4[[0]].dx
  echo jxv5[[0]].dx
  echo jxv6[[0]].dx