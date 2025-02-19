import arraytype,tensortype,tensorattributes

proc `+`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := add(x.storage,y.storage)
proc `+=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.add(y.storage)

proc `+`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := add(x,y.storage)
proc `+`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := add(x.storage,y)
proc `+=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) =
  x.storage.add(y)

proc `-`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := subtract(x.storage,y.storage)
proc `-=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.subtract(y.storage)

proc `-`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := subtract(x,y.storage)
proc `-`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := subtract(x.storage,y)
proc `-=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) =
  x.storage.subtract(y)

proc `*`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := multiply(x.storage,y.storage)
proc `*=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.multiply(y.storage)

proc `*`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := multiply(x,y.storage)
proc `*`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := multiply(x.storage,y)
proc `*=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) =
  x.storage.multiply(y)

proc `/`*(x,y: SwissTensor): SwissTensor =
  operable(x,y)
  like(result,x)
  result.storage := divide(x.storage,y.storage)
proc `/=`*(x: var SwissTensor; y: SwissTensor) =
  operable(x,y)
  x.storage.divide(y.storage)

proc `/`*[V:static[int],T](x: SwissTensor[V,T]; y: T): SwissTensor[V,T] =
  like(result,x)
  result.storage := divide(x.storage,y)
proc `/`*[V:static[int],T](x: T; y: SwissTensor[V,T]): SwissTensor[V,T] =
  like(result,y)
  result.storage := divide(x,y.storage)
proc `/=`*[V:static[int],T](x: var SwissTensor[V,T]; y: T) =
  x.storage.divide(y)

proc `-`*[V:static[int],T](x: SwissTensor[V,T]): SwissTensor[V,T] = T(-1.0)*x

if isMainModule:
  var 
    ts1 = newTensor([2,2],float)
    ts2 = newTensor([2,2],float)
    ts3 = newTensor([2,2],float)
  ts2 := ts1
  ts2[[0,1]] := 1.0
  ts1[[0,0]] := ts2[[0,1]]
  echo ts1[[0,0]]
  echo ts1[[0,0]]
  ts2[[0,0]] := 1.0
  ts2[[1,1]] := 2.0
  ts3[[0,0]] := 3.0
  ts3[[1,1]] := 4.0
  ts1 := ts2 + ts3
  echo ts1[[0,0]]," ",ts1[[0,1]]," ",ts1[[1,0]]," ",ts1[[1,1]]
  ts1 := ts2 - ts3
  echo ts1[[0,0]]," ",ts1[[0,1]]," ",ts1[[1,0]]," ",ts1[[1,1]]
  ts1 := ts2*ts3
  echo ts1[[0,0]]," ",ts1[[0,1]]," ",ts1[[1,0]]," ",ts1[[1,1]]
  ts3[[0,1]] := -1.0
  ts3[[1,0]] := -1.0
  ts1 := ts2/ts3
  echo ts1[[0,0]]," ",ts1[[0,1]]," ",ts1[[1,0]]," ",ts1[[1,1]]
  ts1 := float(1.0)
  ts1 += ts2