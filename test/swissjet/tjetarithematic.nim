import ../../src/swissfit/swissjet/swissjet
import ../../src/swissfit/swisstensor/swisstensor

var x = 1.0
var 
  jx = x.prolong()
  jxc = [2].newTensor(x)
  jxv = [2].vectorize(jx)
  jxv2 = jxv + jxv
  jxv3 = jxv - jxv
  jxv4 = jxv*jxv
  jxv5 = jxv/jxv
  jxv6 = jxv*jxv + jxv
discard jxv + jxc
discard jxc + jxv
discard jxv - jxc
discard jxc - jxv
discard jxv * jxc
discard jxc * jxv
discard jxv / jxc
discard jxc / jxv
var 
  jy = x.prolong(pushforward = false)
  jyc = [2].newTensor(x)
  jyv = [2].vectorize(jy)
  jyv2 = jyv + jyv
  jyv3 = jyv - jyv
  jyv4 = jyv*jyv
  jyv5 = jyv/jyv
  jvy6 = jyv*jyv + jyv
echo "pullback:"
jyv2.pullback()
echo jyv.dx[[0]]
jyv3.pullback()
echo jyv.dx[[0]]
jyv4.pullback()
echo jyv.dx[[0]]
jyv5.pullback()
echo jyv.dx[[0]]
jvy6.pullback()
echo jyv.dx[[0]]
echo "pushforward:"
discard jyv + jyc
discard jyc + jyv
discard jyv - jyc
discard jyc - jyv
discard jyv * jyc
discard jyc * jyv
discard jyv / jyc
discard jyc / jyv
echo jxv2.dx[[0]]
echo jxv3.dx[[0]]
echo jxv4.dx[[0]]
echo jxv5.dx[[0]]
echo jxv6.dx[[0]]