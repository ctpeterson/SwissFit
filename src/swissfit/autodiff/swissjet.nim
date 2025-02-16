# SwissJet type
# Represents 1-jet (dual number); able to do both
# forward-mode and reverse-mode automatic differentiation

import sequtils
import swisstensor

export swisstensor

type 
  SwissJet[V:static[int],T] = ref object
    x: SwissTensor[V,T]
    case requiresGrad: bool
      of true:
        grad: SwissTensor[V,T]
        y: seq[SwissJet[V,T]]
        propagate: proc(x: var SwissJet[V,T])
      of false:
        dx: SwissTensor[V,T]

proc prolong*[V:static[int],T](
    x: SwissTensor[V,T]; 
    requiresGrad: bool = false
  ): SwissJet[V,T] = 
  result = SwissJet[V,T](x: x, requiresGrad: requiresGrad)
  if result.requiresGrad: 
    like(result.grad,x)
    result.grad := T(0.0)
    result.propagate = proc(x: var SwissJet[V,T]) = discard
    result.y = newSeq[SwissJet[V,T]]()
  else: 
    like(result.dx,x)
    result.dx := T(1.0)

template requiresGrad(a,b): bool = ((a.requiresGrad) or (b.requiresGrad))

proc newGrad[V:static[int],T](x,a,b: var SwissJet[V,T]) =
  if a.requiresGrad: like(x.grad,a.grad)
  else: like(x.grad,b.grad)
  x.grad := T(1.0)

proc `+`*[V:static[int],T](a,b: var SwissJet[V,T]): SwissJet[V,T] =
  result = SwissJet[V,T](requiresGrad: requiresGrad(a,b))
  like(result.x,a.x)
  result.x := a.x + b.x
  case result.requiresGrad: 
    of true:
      result.newGrad(a,b)
      result.propagate = proc(x: var SwissJet[V,T]) =
        if x.y[0].requiresGrad: 
          x.y[0].grad := x.grad + x.y[0].grad
        if x.y[1].requiresGrad: 
          x.y[1].grad := x.grad + x.y[0].grad
      result.y = @[a,b]
    of false: 
      like(result.dx,a.dx)
      result.dx := a.dx + b.dx

proc `-`*[V:static[int],T](a,b: var SwissJet[V,T]): SwissJet[V,T] =
  result = SwissJet[V,T](requiresGrad: requiresGrad(a,b))
  like(result.x,a.x)
  result.x := a.x + b.x
  case result.requiresGrad: 
    of true:
      result.newGrad(a,b)
      result.propagate = proc(x: var SwissJet[V,T]) =
        if x.y[0].requiresGrad: 
          x.y[0].grad := x.grad + x.y[0].grad
        if x.y[1].requiresGrad: 
          x.y[1].grad := x.grad - x.y[0].grad
      result.y = @[a,b]
    of false: 
      like(result.dx,a.dx)
      result.dx := a.dx - b.dx

proc `*`*[V:static[int],T](a,b: var SwissJet[V,T]): SwissJet[V,T] =
  result = SwissJet[V,T](requiresGrad: requiresGrad(a,b))
  like(result.x,a.x)
  result.x := a.x *. b.x
  case result.requiresGrad: 
    of true:
      result.newGrad(a,b)
      result.propagate = proc(x: var SwissJet[V,T]) =
        if x.y[0].requiresGrad: 
          x.y[0].grad := x.y[0].x + (x.y[1].x *. x.grad)
        if x.y[1].requiresGrad: 
          x.y[1].grad := x.y[1].x + (x.y[0].x *. x.grad)
      result.y = @[a,b]
    of false: 
      like(result.dx,a.dx)
      result.dx := (a.dx *. b.x) + (a.x *. b.dx)

proc `/`*[V:static[int],T](a,b: var SwissJet[V,T]): SwissJet[V,T] =
  result = SwissJet[V,T](requiresGrad: requiresGrad(a,b))
  like(result.x,a.x)
  result.x := a.x /. b.x
  case result.requiresGrad: 
    of true:
      result.newGrad(a,b)
      result.propagate = proc(x: var SwissJet[V,T]) =
        if x.y[0].requiresGrad: 
          x.y[0].grad := x.y[0].x + (x.grad /. x.y[1].x)
        if x.y[1].requiresGrad: 
          x.y[1].grad := x.y[1].x - ((x.grad *. x.y[0].x) /. (x.y[1].x *. x.y[1].x))
      result.y = @[a,b]
    of false: 
      like(result.dx,a.dx)
      result.dx := (a.dx /. b.x) - ((a.x * b.dx) /. (b.x *. b.x))

proc backprop*[V:static[int],T](y: var SwissJet[V,T]) = 
  assert(y.requiresGrad)
  var graph: seq[SwissJet[V,T]] = @[y]
  while graph.len > 0:
    var node = graph.pop()
    node.propagate(node)
    for next in node.y: graph.add(next)

if isMainModule:
  var x = new([1],float)
  x := 1.0
  var
    jx = x.prolong()
    jx2 = jx*jx
    jy = x.prolong(requiresGrad = true)
    jy2 = jy*jy
  echo jx2.x[[0]], " ", jx2.dx[[0]]
  jy2.backprop()
  echo jy2.grad[[0]], " ", jy.grad[[0]]