import ../../src/swissfit/simd/[swisssimd]

template test(N,T,F: untyped) =
  var 
    x,y,z: T
    v: array[N,F]
  for idx in 0..<N: v[idx] = F(idx)
  assign(x,1)
  echo x
  assign(x,1.0)
  echo x
  assign(y,2)
  assign(z,0)
  discard x + y
  y += x
  y += z
  echo y
  assign(z,v)
  echo z
  z += x
  echo z
  z = x - y
  echo z
  z -= y
  echo z
  y = z*x
  echo y
  y *= z
  echo y
  y = z/x
  echo y
  y /= z
  echo y
when defined(SSE):
  echo "----------- float32 -----------"
  test(4,`m128s`,float32)
  echo "----------- float64 -----------"
  test(2,`m128d`,float64)
when defined(AVX): 
  echo "----------- float32 -----------"
  test(8,`m256s`,float32)
  echo "----------- float64 -----------"
  test(4,`m256d`,float64)
when defined(AVX512):
  echo "----------- float32 -----------"
  test(16,`m512s`,float32)
  echo "----------- float64 -----------"
  test(8,`m512d`,float64)