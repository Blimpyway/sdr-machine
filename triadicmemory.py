"""

triadicmemory.py

Copyright (c) 2022 Cezar Totth

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the “Software”), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

==============================================================================================


A basic numpy/python implementation of Peter Overmann's Triadic Memory idea.

see https://peterovermann.com/TriadicMemory.pdf

inspired by the reference C implemetation: 
https://github.com/PeterOvermann/TriadicMemory/blob/main/triadicmemory.c

"""

import numpy as np

class TriadicMemory:
    def __init__(self, N = 1000, P = 10): 
        self.N = N
        self.P = P 
        self._mem = np.zeros((N,N,N), dtype = np.uint8)


    def store(self, x,y,z): 
        self._mem[x,y,z] += 1


    def query(self, x,y,z = None):
        # One of x, y, z must be None. That will be queried for
        if x is None:
            return self.queryX(y, z)
        elif y is None: 
            return self.queryY(x, z)
        elif z is None:
            return self.queryZ(x, y)
        else:
            pass
            # Here we can either store x,y,x triple, return nothing or throw error
    
    def queryX(self, y,z): 
        # Find X knowing Y and Z
        r = np.argsort(self._mem[:,y,z].sum(axis=1))[-self.P:]
        r.sort()
        return r


    def queryY(self, x,z):
        # Find Y knowing X and Z
        r = np.argsort(self._mem[x,:,z].sum(axis=0))[-self.P:]
        r.sort()
        return r


    def queryZ(self, x,y):
        # Find Z knowing Y and X
        r = np.argsort(self._mem[x,y,:].sum(axis=0))[-self.P:]
        r.sort()
        return r



if __name__ == "__main__":
    # example usage
    
    tm = TriadicMemory()

    X = [10,11,12,13,14,15,16,17,18,19]
    Y = [100,101,102,103,104,105,106,107,108,109]
    Z = [200,201,202,203,204,205,206,207,208,209]


    tm.store(X,Y,Z) 

    print(tm.query(None, Y, Z)) 

    print(tm.query(X, None, Z))

    print(tm.query(X, Y))
