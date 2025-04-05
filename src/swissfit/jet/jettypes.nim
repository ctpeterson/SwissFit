# Swiss1Jet type
# Represents 1-jet (dual number); able to do both
# pushforward-mode and reverse-mode automatic differentiation.
# Prologation "functor" (not quite a functor) takes types 
# "T" to their first-jet prolongation; elementary arithematic of 
# first-jet prolongation taken care of by operator overloading

import ../tensor/[swisstensor,attributes]

type 
  Swiss1Jet*[T] = ref object
    # Represents a 1-jet
    x*,dx*: T
    case pushforward*: bool
      of true: discard
      of false:
        back*: proc(x: Swiss1Jet[T])
        stack*: seq[Swiss1Jet[T]]
    derived*: bool

proc prolong*[T](
    x: T; 
    pushforward: bool = true; 
    derived: bool = false
  ): Swiss1Jet[T] =
  result = Swiss1Jet[T](x: x, pushforward: pushforward, derived: derived)
  case pushforward:
    of true: (if not derived: result.dx <- 1.0)
    of false:
      if not derived: result.dx <- 0.0
      result.back = proc(x: Swiss1Jet[T]) = discard
      result.stack = newSeq[Swiss1Jet[T]]()

proc prolong*[V:static[int],T](
    x: SwissTensor[V,T]; 
    pushforward: bool = true; 
    derived: bool = false
  ): Swiss1Jet[SwissTensor[V,T]] =
  result = Swiss1Jet[SwissTensor[V,T]](
    x: x.shape.newTensor(typeof(T)),
    dx: x.shape.newTensor(typeof(T)),
    pushforward: pushforward, 
    derived: derived
  )
  result.x := x
  case pushforward:
    of true: (if not derived: result.dx <- 1.0)
    of false:
      if not derived: result.dx <- 0.0
      result.back = proc(x: Swiss1Jet[SwissTensor[V,T]]) = discard
      result.stack = newSeq[Swiss1Jet[SwissTensor[V,T]]]()

proc vectorize*[V:static[int],T](
    s: array[V,int]; 
    x: Swiss1Jet[T]
  ): Swiss1Jet[SwissTensor[V,T]] =
  assert(not x.derived)
  result = Swiss1Jet[SwissTensor[V,T]](pushforward: x.pushforward, derived: false)
  (result.x,result.dx) = (s.newTensor(x.x), s.newTensor(x.dx))
  if not result.pushforward: 
    result.back = proc(x: Swiss1Jet[SwissTensor[V,T]]) = discard
    result.stack = newSeq[Swiss1Jet[SwissTensor[V,T]]]()

template sweep[T](graph: var seq[Swiss1Jet[T]]; work: untyped) =
  var rank {.inject.} = 0
  while graph.len > 0:
    var node {.inject.} = graph.pop()
    work 
    inc rank
    for next in node.stack: graph.add(next)

proc sanitize[T](head: Swiss1Jet[T]): seq[Swiss1Jet[T]] =
  var graph = @[head]
  graph.sweep:
    if (not node.dx.isUnit) and (rank == 0): node.dx <- 1.0
    if (not node.dx.isNull) and (rank > 0): node.dx <- 0.0
  result = @[head]

proc pullback*[T](jx: Swiss1Jet[T]) =
  assert(not jx.pushforward)
  var graph = jx.sanitize()
  graph.sweep: node.back(node)