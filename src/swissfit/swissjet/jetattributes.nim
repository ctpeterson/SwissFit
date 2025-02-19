import jettypes
import ../swisstensor/[swisstensor]

proc isUnit*[T](x: T): bool {.inline.} = 
  result = true
  if x is SwissArray: result = x.isUnit
  else: result = (if x is SwissTensor: x.storage.isUnit else: x == x)

proc isNull*[T](x: T): bool {.inline.} =
  result = true
  if x is SwissArray: result = x.isNull
  else: result = (if x is SwissTensor: x.storage.isNull else: x == x)

template pushforward*(a,b): bool = ((a.pushforward) or (b.pushforward))

template register*[T](x: var Swiss1Jet[T]; y: Swiss1Jet[T]) = 
  (if y.dx.isNull: y.dx <- 0.0); x.stack.add(y);

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