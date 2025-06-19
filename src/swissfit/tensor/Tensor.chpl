/*
  MIT License
  Copyright (c) 2023 Curtis Taylor Peterson
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
 
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/
/* END LEGAL */

module VectorizedArrayArithematic
{

operator=(ref x: [?s] ?T, y: T){foreach i in x.domain do {x[i] = y;}}

operator+(x: [?s] ?T, y: [s] T)
{var r: [s] T; foreach i in r.domain do {r[i] = x[i] + y[i];} return r;}

operator-(x: [?s] ?T, y: [s] T)
{var r: [s] T; foreach i in r.domain do {r[i] = x[i] - y[i];} return r;}

}

module Tensor
{

public use VectorizedArrayArithematic;
import Time;

record Tensor
{
  /**
    Base tensor data type. Standard standard overloaded/vectorized arithematic
    operations. 

    Fields:
      - `T`: Field over which Tensor takes its values
      - `S`: Rank of tensor
      - `s`: Shape of tensor
      - `t`: Tensor storage as Chapel array
   */
  type T;
  param S: int;
  var s: domain(S,int);
  var t: [s] T;

  // --- base constructors --- //

  proc init(param S: int, type T){this.T = T; this.S = S;}

  proc init(type T, s: int ...?S)
  {
    var r: S*range;
    init(S,T); 
    for (ss,rr) in zip(s,r) do {rr = 0..#ss;}
    this.s = {(...r)};
  }
  
  proc init(t: [?s] ?T){var s = t.domain; init(s.rank,T); this.s = s; this.t = t;}

  // --- assignment constructors --- //

  proc init=(x: Tensor(?T,?S)){init(S,T); (this.s,this.t) = (x.s,x.t);}

  proc init=(x: [?s] ?T){init(x.rank,T); (this.s,this.t) = (s,x);}

  // --- assignment --- //

  operator=(ref x: Tensor(?T,?S), y: Tensor(T,S)){(x.s,x.t) = (y.s,y.t);}

  operator=(ref x: Tensor(?T,?S), y: [?s] T){assert(s.rank == S); (x.s,x.t) = (s,y);}

  // --- casting --- //

  operator:(in x: [?s] ?T, type y: Tensor(T,?S)): Tensor(T,S) {return new Tensor(x);}

  // --- arithematic --- //

  operator+(x: Tensor(?T,?S), y: Tensor(T,S)){return new Tensor(x.t + y.t);}

  operator-(x: Tensor(?T,?S), y: Tensor(T,S)){return new Tensor(x.t - y.t);}
  
}

proc main()
{
  var 
    ta = new Tensor(real(64),1000,1000),
    tb = new Tensor(real(64),1000,1000),
  tc = ta + tb,
  td = tc - tb;
  ta = tb.t;
}

}