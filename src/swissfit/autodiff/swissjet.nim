# SwissJet type
# Represents 1-jet (dual number); able to do both
# forward-mode and reverse-mode automatic differentiation

import sequtils
import swisstensor

export swisstensor

type 
  SwissJet[T] = ref object
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
  result.storage = new(tensor(result,shape),SwissJet[T])
  for idx in 0..<result.storage.len: 
    result.storage[idx] = (x.x).prolong(forward = x.forward, derived = x.derived)

template forward(a,b): bool = ((a.forward) or (b.forward))

proc `+`*[T](a,b: SwissJet[T]): SwissJet[T] =
  result = (a.x + b.x).prolong(forward = forward(a,b), derived = true)
  case result.forward:
    of true: result.dx = a.dx + b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += x.dx
        if not b.forward: b.dx += x.dx
      result.stack.add(a)
      result.stack.add(b)

proc `-`*[T](a,b: SwissJet[T]): SwissJet[T] =
  result = (a.x - b.x).prolong(forward = forward(a,b), derived = true)
  case result.forward:
    of true: result.dx = a.dx - b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += x.dx
        if not b.forward: b.dx -= x.dx
      result.stack.add(a)
      result.stack.add(b)

proc `*`*[T](a,b: SwissJet[T]): SwissJet[T] =
  result = (a.x * b.x).prolong(forward = forward(a,b), derived = true)
  case result.forward:
    of true: result.dx = a.dx*b.x + a.x*b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += b.x*x.dx
        if not b.forward: b.dx += a.x*x.dx
      result.stack.add(a)
      result.stack.add(b)
proc `*`*[T](a: T; b: SwissJet[T]): SwissJet[T] =
  result = (a * b.x).prolong(forward = b.forward, derived = true)
  case result.forward:
    of true: result.dx = a*b.dx
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not b.forward: b.dx += a.x*x.dx
      result.stack.add(b)
proc `*`*[T](a: SwissJet[T]; b: T): SwissJet[T] =
  result = (a.x * b).prolong(forward = a.forward, derived = true)
  case result.forward:
    of true: result.dx = a.dx*b
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += b*x.dx
      result.stack.add(a)

proc `/`*[T](a,b: SwissJet[T]): SwissJet[T] =
  result = (a.x / b.x).prolong(forward = forward(a,b), derived = true)
  case result.forward:
    of true: result.dx = a.dx/b.x - a.x*b.dx/(b.x*b.x)
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += x.dx/b.x
        if not b.forward: b.dx -= a.x*x.dx/(b.x*b.x)
      result.stack.add(a)
      result.stack.add(b)
proc `/`*[T](a: T; b: SwissJet[T]): SwissJet[T] =
  result = (a / b.x).prolong(forward = b.forward, derived = true)
  case result.forward:
    of true: result.dx = (-a)*b.dx/(b.x*b.x)
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not b.forward: b.dx -= a*x.dx/(b.x*b.x)
      result.stack.add(b)
proc `/`*[T](a: SwissJet[T]; b: T): SwissJet[T] =
  result = (a.x / b).prolong(forward = a.forward, derived = true)
  case result.forward:
    of true: result.dx = a.dx/b
    of false:
      result.backprop = proc(x: SwissJet[T]) =
        if not a.forward: a.dx += x.dx/b
      result.stack.add(a)

proc backprop*[T](jx: SwissJet[T]) =
  assert(not jx.forward)
  var graph = @[jx]
  while graph.len > 0:
    var node = graph.pop()
    node.backprop(node)
    for next in node.stack: graph.add(next)

proc backprop*[V:static[int],T](jx: SwissTensor[V,T]) =
  for idx in 0..<jx.size: backprop(jx.storage[idx])

if isMainModule:
  var x = 1.0
  var 
    jx = x.prolong()
    jxv = [2].newTensor(jx)
    jxv2 = jxv + jxv
  var 
    jy = x.prolong(forward = false)
    jyv = [2].newTensor(jy)
    jyv2 = jyv + jyv
  jyv2.backprop()
  echo jxv2[[0]].x, " ", jxv2[[0]].dx
  echo jyv[[0]].x, " ", jyv[[0]].dx

  #  jx2 = jx*jx
  #  jy = x.prolong(requiresGrad = true)
  #  jy2 = jy*jy
  #echo jx2.x[[0]], " ", jx2.dx[[0]]
  #jy2.backprop()
  #echo jy2.grad[[0]], " ", jy.grad[[0]]