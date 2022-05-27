"""

triadicmemory.py

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
