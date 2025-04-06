# nim c -d:x86 -d:AVX512 -d:release --app:lib --noMain --gc:orc swissvar.nim
import nimpy
import ../tensor/[swisstensor]
import ../jet/[swissjet]
export swissjet
export swisstensor

