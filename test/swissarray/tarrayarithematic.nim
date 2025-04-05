import ../../src/swissfit/array/[swissarray]

template test(N,F: untyped) =
  let size = N #2*N + 1
  var (v1,v2) = (newSeq[F](size),newSeq[F](size))
  for idx in 0..<size: (v1[idx],v2[idx]) = (F(idx),F(size-idx))
  var sv1,sv2: SwissArray[F]
  sv1 = new(v1)
  sv2 = new(v2)
  echo "SV1: ", $sv1
  echo "SV2: ", $sv2
  echo "VECTOR-VECTOR OPERATIONS"
  echo "ADD: ", $add(sv1,sv2)
  echo "SUB: ", $sub(sv1,sv2)
  echo "MUL: ", $mul(sv1,sv2)
  echo "DIV: ", $divd(sv1,sv2)
  echo "VECTOR-SCALAR OPERATIONS"
  echo "ADDS: ", $add(sv1,1.0), " ADDS: ", $add(sv1,1)
  echo "SUBS: ", $sub(sv1,1.0), " SUBS: ", $sub(sv1,1)
  echo "MULS: ", $mul(sv1,2.0), " MULS: ", $mul(sv1,2)
  echo "DIVS: ", $divd(sv1,2.0), " MULS: ", $divd(sv1,2)
  echo "SCALAR-VECTOR OPERATIONS"
  echo "ADDS: ", $add(1.0,sv1), " ADDS: ", $add(1,sv1)
  echo "SUBS: ", $sub(1.0,sv1), " SUBS: ", $sub(1,sv1)
  echo "MULS: ", $mul(2.0,sv1), " MULS: ", $mul(2,sv1)
  echo "DIVS: ", $divd(2.0,sv1), " MULS: ", $divd(2,sv1)
test(5,float32)
test(5,float64)
test(15,float32)
test(7,float64)
test(16,float32)
test(8,float64)