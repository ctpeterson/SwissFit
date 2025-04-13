## SwissSeq (dynamic stack array) type
## 
## Mocks up sequence data type in Nim
## 
## Notes:
##   - Modelled after dynamic stack array in Arraymaner 
##    (https://github.com/mratsim/Arraymancers)

when defined(x86):
  const 
    MAXSEQRANK* {.intdefine.} = 6
    MAXRANK* {.intdefine.} = 7

type
  Index = Natural or BackwardsIndex
  SwissSeq*[T] = object
    len: int
    data*: array[MAXRANK,T]

proc `:=`*[T](x: var SwissSeq[T]; y: seq[T]) =
  assert(y.len <= MAXSEQRANK)
  for idx in 0..<y.len: x.data[idx] = y[idx] 
proc `:=`*[V:static[int],T](x: var SwissSeq[T]; y: array[V,T]) =
  assert(V <= MAXSEQRANK)
  for idx in 0..<V: x.data[idx] = y[idx] 
proc `:=`*[T](x: var SwissSeq[T]; y: SwissSeq[T]) {.inline.} = (x = y)

proc newSwissSeq*[T](len: int): SwissSeq[T] =
  assert(len <= MAXRANK)
  result = SwissSeq[T](len: len)
proc newSwissSeq*[T](y: seq[T]): SwissSeq[T] =
  result = newSwissSeq[T](y.len)
  result := y
proc newSwissSeq*[V:static[int],T](y: array[V,T]): SwissSeq[T] =
  result = newSwissSeq[T](V)
  result := y
proc like*[T](x: var SwissSeq[T]; y: SwissSeq[T]) = (x = newSwissSeq[T](y.len))

proc len*[T](x: SwissSeq[T]): int {.inline.} = x.len
proc index[T](x: SwissSeq[T]; idx: Index): int =
  result = case idx is BackwardsIndex
    of true: x.len - int(idx)
    of false: int(idx)
template `[]`*[T](x: SwissSeq[T]; idx: Index): lent T =
  assert idx < x.len
  x.data[x.index(idx)]
template `[]`*[T](x: var SwissSeq[T]; idx: Index): lent T =
  assert idx < x.len
  x.data[x.index(idx)]
template `[]=`*[T](x: var SwissSeq[T]; idx: Index; y: sink T) =
  assert idx < x.len 
  x.data[x.index(idx)] = y

proc items*[T](x: SwissSeq[T]): T =
  for idx in 0..<x.len: yield x.data[idx]
proc append*[T](x: var SwissSeq[T]; y: T) {.inline.} =
  x[x.len] = y
  inc x.len
proc product*[T](x: SwissSeq[T]): T =
  result = 1
  for idx in 0..<x.len: result *= x[idx]
proc `@`*[T](x: SwissSeq[T]): seq[T] =
  result = newSeq[T](x.len)
  for idx in 0..<x.len: result[idx] = x[idx]
proc `$`*[T](x: SwissSeq[T]): string =
  result = "[" & $x[0]
  for idx in 1..<x.len: result &= ", " & $x[idx]
  result &= "]"