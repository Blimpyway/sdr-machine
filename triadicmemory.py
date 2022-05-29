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
        self.P = P 
        self._mem = np.zeros((N,N,N), dtype = np.uint8)

    def store(self, x,y,z): 
        for i in x:
            for j in y:
                for k in z:
                    self._mem[i,j,k] += 1

    def query(self, x,y,z = None):
        # Only  one of x, y, z can be None. That will be queried for
        # If neither is None it will do a store() of the triplet and return None
        if z is None: # most common case first
            sums = self._mem[x,:,:][:,y,:].sum(axis = (0,1))
        elif y is None: 
            sums = self._mem[x,:,:][:,:,z].sum(axis=(0,2))
        elif x is None:
            sums = self._mem[:,y,:][:,:,z].sum(axis=(1,2))
        else: 
            # neither is None - we don't know what to query for. 
            # but we can store it..
            self.store(x,y,z)
            return None
        return self._clear_response(sums)

    def _clear_response(self, sums):
        # this does what "binarize()" 
        ssums = sums.copy()
        ssums.sort()
        threshval = ssums[-self.P] 
        if threshval == 0:
            return np.where(sums)[0] 
        else:
            return np.where(sums >= threshval)[0]

if __name__ == "__main__":
    # example usage
    
    tm = TriadicMemory(N=100)

    W = [10,11,12,13,14,15,16,17,18,19]
    X = [20,21,22,23,24,25,26,27,28,29]    
    Y = [30,31,32,33,34,35,36,37,38,39]
    Z = [40,41,42,43,44,45,46,47,48,49]
    OMEGA = [50,51,52,53,54,55,56,57,58,59]

    tm.store(W,X,Y)
    tm.store(X,Y,Z)
    # tm.store(X,Y,OMEGA)

    print("W is ", W)
    print("X is ", X)
    print("Y is ", Y) 
    print("Z is ", Z)
    print("Expect W: ", tm.query(None, X, Y))
    print("Expect X: ", tm.query(W, None, Y)) 
    print("Expect Z: ", tm.query(X, Y))
    print("Partial X,Y query:", tm.query(X,Y[:-2]))
