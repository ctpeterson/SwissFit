## SIMD approach based on https://github.com/jcosborn/qex/blob/devel/src/simd/

import macros

when defined(SSE):
  import x86
  export x86
  const (VLENF*,VLEND*) = (4,2)
when defined(AVX):
  import x86
  export x86
  const (VLENF*,VLEND*) = (8,4)
when defined(AVX512):
  import x86
  export x86
  const (VLENF*,VLEND*) = (16,8)

const codegenDecl = "inline __attribute__((always_inline)) $# $#$#"
{.passL: "-lm".}
{.pragma: alwaysInline, inline, codegenDecl: codegenDecl.}

template mixUnaryOperationVar(T,O1,O2: untyped) = 
  template O1*(y: T; x: SomeNumber): T = O2(y,x.to(T))

template mixBinaryOperation(T,F,O1,O2: untyped) =
  template O1*(x: SomeNumber; y: T): T = O2(y,F(x).toSIMD())
  template O1*(x: T; y: SomeNumber): T = O2(x,F(y).toSIMD())

template newOperationSet(T,F,P,S,O1,O2,S1,S2: untyped) =
  template O1*(x,y: T): T = `P "_"O1"_" S`(x,y)
  mixBinaryOperation(T,F,O1,O1)
  template O1*(y: T; x1,x2: T) = y = O1(x1,x2)
  template S1*(x,y: T): T = O1(x,y)
  mixBinaryOperation(T,F,S1,O1)
  proc O2*(y: var T; x: T) {.alwaysInline.} = O1(y,y,x)
  mixUnaryOperationVar(T,O2,O2)
  template S2*(y: T; x: T) = O2(y,x)

template define(T,F,N,P,S: untyped) {.dirty.} =
  proc assign*(y: var T; x: SomeNumber) = (y = `P "_set1_" S`(F(x)))
  proc assign*(y: var T; x: array[N,SomeNumber]) {.alwaysInline.} =
    when x[0] is F: y = `P "_loadu_" S`(cast[ptr F](unsafeAddr(x)))
    else:
      var t {.noInit.}: array[N,F]
      for idx in 0..<N: t[idx] = F(x[idx])
      assign(y,x)

  proc toSIMD*(x: array[N,F]): T = `P "_loadu_" S`(unsafeAddr x[0])
  proc toSIMD*(x: ptr UncheckedArray[F]): T = `P "_loadu_" S`(x)
  proc toSIMD*(x: F): T = `P "_set1_" S`(x)
  proc load*(x: ptr UncheckedArray[F]; idx: int): T = `P "_loadu_" S`(addr x[idx])
  proc toArray*(x: T): array[N,F] {.alwaysInline, noInit.} = 
    `P "_storeu_" S`(addr result[0], x)
  proc store*(x: ptr UncheckedArray[F]; y: T; idx: int) =
    `P "_storeu_" S`(addr x[idx], y)
  proc `[]`*(x: T; i: SomeInteger): F {.alwaysInline, noInit.} = toArray(x)[i]

  newOperationSet(T,F,P,S,add,iadd,`+`,`+=`)
  newOperationSet(T,F,P,S,sub,isub,`-`,`-=`)
  newOperationSet(T,F,P,S,mul,imul,`*`,`*=`)
  newOperationSet(T,F,P,S,divd,idiv,`/`,`/=`)

  proc `$`*(x: T): string =
    result = "[" & $x[0]
    for i in 1..<N: result &= ", " & $x[i]
    result &= "]"

proc vlen*[T](): int = 
  result = case T is float32
    of true: VLENF
    of false: VLEND

when defined(SSE):
  define(m128s,float32,VLENF,mm128,ps)
  define(m128d,float64,VLEND,mm128,pd)
when defined(AVX):
  define(m256s,float32,VLENF,mm256,ps)
  define(m256d,float64,VLEND,mm256,pd)
when defined(AVX512):
  define(m512s,float32,VLENF,mm512,ps)
  define(m512d,float64,VLEND,mm512,pd)
