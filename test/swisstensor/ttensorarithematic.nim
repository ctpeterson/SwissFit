import ../../src/swissfit/swisstensor/swisstensor

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