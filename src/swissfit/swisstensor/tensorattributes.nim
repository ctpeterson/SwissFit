import arraytype,tensortype

template active*(x: SwissTensor) = assert(x.storage.data != nil)

template conformable*(x,y: SwissTensor) = conformable(x.storage,y.storage)

template comparable*(x,y: SwissTensor) =
  conformable(x,y)
  assert(x.shape == y.shape)

template operable*(x,y: SwissTensor) =
  active(x)
  active(y)
  comparable(x,y)

proc isUnit*[T](x: T): bool {.inline.} = 
  result = true
  if x is SwissArray: result = x.isUnit
  else: result = (if x is SwissTensor: x.storage.isUnit else: x == x)

proc isNull*[T](x: T): bool {.inline.} =
  result = true
  if x is SwissArray: result = x.isNull
  else: result = (if x is SwissTensor: x.storage.isNull else: x == x)