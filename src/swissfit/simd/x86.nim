when defined(SSE):
  {.passC: "-msse2".}
  {.passL: "-msse2".}
when defined(AVX):
  {.passC: "-mavx2".}
  {.passL: "-mavx2".}
when defined(AVX512):
  {.passC: "-mavx512f".}
  {.passL: "-mavx512f".}

when defined(SISD):
  type
    m32s* {.incompleteStruct.} = ref object
      v*: cfloat
    m64d* {.incompleteStruct.} = ref object
      v*: cdouble
when defined(SSE):
  {.pragma: imms, header: "xmmintrin.h", incompleteStruct.} 
  type
    m128s* {.importc: "__m128", imms.} = object
    m128d* {.importc: "__m128d", imms.} = object
when defined(AVX):
  {.pragma: imms, header: "immintrin.h", incompleteStruct.} 
  type
    m256s* {.importc: "__m256", imms.} = object
    m256d* {.importc: "__m256d", imms.} = object
when defined(AVX512):
  {.pragma: imms, header: "immintrin.h", incompleteStruct.} 
  type
    m512s* {.importc: "__m512", imms.} = object
    m512d* {.importc: "__m512d", imms.} = object
  
template intrinsics(T,S,P,F: untyped; t,s: string) =
  {.pragma: imm, header: "xmmintrin.h".}
  proc `"m" T "_loadu_" S`*(x: pointer): `T F` 
    {.importc: "_" & t & "_loadu_" & s, imm.}
  proc `"m" T "_storeu_" S`*(x: pointer; y: `T F`) 
    {.importc: "_" & t & "_storeu_" & s, imm.}
  proc `"m" T "_set1_" S`*(a: `P`): `T F` 
    {.importc: "_" & t & "_set1_" & s, imm.}
  proc `"m" T "_add_" S`*(x,y: `T F`): `T F` 
    {.importc: "_" & t & "_add_" & s, imm.}
  proc `"m" T "_sub_" S`*(x,y: `T F`): `T F` 
    {.importc: "_" & t & "_sub_" & s, imm.}
  proc `"m" T "_mul_" S`*(x,y: `T F`): `T F` 
    {.importc: "_" & t & "_mul_" & s, imm.}
  proc `"m" T "_divd_" S`*(x,y: `T F`): `T F` 
    {.importc: "_" & t & "_div_" & s, imm.}

when defined(SSE):
  intrinsics(m128,ps,cfloat,s,"mm","ps")
  intrinsics(m128,pd,cdouble,d,"mm","pd")
when defined(AVX):
  intrinsics(m256,ps,cfloat,s,"mm256","ps")
  intrinsics(m256,pd,cdouble,d,"mm256","pd")
when defined(AVX512):
  intrinsics(m512,ps,cfloat,s,"mm512","ps")
  intrinsics(m512,pd,cdouble,d,"mm512","pd")
when defined(SISD): 
  # Some annoyances w/ nim in creating a template for this - to do
  proc mm32_loadu_ps*(x: pointer): m32s = m32s(v: cast[ptr cfloat](x)[])
  proc mm64_loadu_pd*(x: pointer): m64d = m64d(v: cast[ptr cdouble](x)[])

  proc mm32_storeu_ps*(x: pointer; y: m32s) = (y.v = cast[ptr cfloat](x)[])
  proc mm64_storeu_pd*(x: pointer; y: m64d) = (y.v = cast[ptr cdouble](x)[])

  proc mm32_set1_ps*(x: float32): m32s = m32s(v: x)
  proc mm64_set1_pd*(x: float64): m64d = m64d(v: x)

  proc mm32_add_ps*(x,y: m32s): m32s = m32s(v: x.v + y.v)
  proc mm64_add_pd*(x,y: m64d): m64d = m64d(v: x.v + y.v)

  proc mm32_sub_ps*(x,y: m32s): m32s = m32s(v: x.v - y.v)
  proc mm64_sub_pd*(x,y: m64d): m64d = m64d(v: x.v - y.v)

  proc mm32_mul_ps*(x,y: m32s): m32s = m32s(v: x.v*y.v)
  proc mm64_mul_pd*(x,y: m64d): m64d = m64d(v: x.v*y.v)

  proc mm32_divd_ps*(x,y: m32s): m32s = m32s(v: x.v/y.v)
  proc mm64_divd_pd*(x,y: m64d): m64d = m64d(v: x.v/y.v)

  #[
  proc `[]`*(x: m32s; idx: Natural): lent cfloat =
    assert idx == 0
    result = x.v
  proc `[]`*(x: m64d; idx: Natural): lent cdouble =
    assert idx == 0
    result = x.v
  ]#

#[
template default(T,S,P,F: untyped) =
  proc `"m" T "_loadu_" S`*(x: pointer): `T F` = `T F`(v: cast[ptr `P`](x)[])
  proc `"m" T "_storeu_" S`*(x: pointer; y: `T F`) = (y.v = cast[ptr `P`](x)[])
  proc `"m" T "_set1_" S`*(a: `P`): `T F` = `T F`(v: a)
  proc `"m" T "_add_" S`*(x,y: `T F`): `T F` = `T F`(v: x.v + y.v)
  proc `"m" T "_sub_" S`*(x,y: `T F`): `T F` = `T F`(v: x.v - y.v)
  proc `"m" T "_mul_" S`*(x,y: `T F`): `T F` = `T F`(v: x.v*y.v)
  proc `"m" T "_divd_" S`*(x,y: `T F`): `T F` = `T F`(v: x.v/y.v)
]#