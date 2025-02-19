import jettypes
import ../swisstensor/[swisstensor,tensorattributes]

template pushforward*(a,b): bool = ((a.pushforward) or (b.pushforward))

template register*[T](x: var Swiss1Jet[T]; y: Swiss1Jet[T]) = 
  if y.dx.isNull: y.dx <- 0.0
  x.stack.add(y)