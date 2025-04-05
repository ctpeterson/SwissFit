when defined(SSE):
  {.passC: "-mavx".}
  {.passL: "-mavx".}
when defined(AVX):
  {.passC: "-mavx".}
  {.passL: "-mavx".}
when defined(AVX512):
  {.passC: "-mavx512f".}
  {.passL: "-mavx512f".}

{.pragma: imm, header: "immintrin.h".}
{.pragma: imms, header: "immintrin.h", incompleteStruct.} 

type
  m128s* {.importc: "__m128", imms.} = object
  m256s* {.importc: "__m256", imms.} = object
  m512s* {.importc: "__m512", imms.} = object

  m128d* {.importc: "__m128d", imms.} = object
  m256d* {.importc: "__m256d", imms.} = object
  m512d* {.importc: "__m512d", imms.} = object

template intrinsics(T,S,P,F: untyped; t,s: string) =
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