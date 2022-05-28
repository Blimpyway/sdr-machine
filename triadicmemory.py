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
        self._mem[x,y,z] += 1


    def query(self, x,y,z = None):
        # One of x, y, z must be None. That will be queried for
       
        if z is None: # most common case first
            sums = self._mem[x,y,:].sum(axis=0)
        elif y is None: 
            sums = self._mem[x,:,z].sum(axis=0)
        elif x is None:
            sums = self._mem[:,y,z].sum(axis=1)
        else: 
            # neither is None - we don't know what to query for. 
            # but we can store it..
            self.store(x,y,z)
            return None
        return self._clear_response(sums)
    

    def _clear_response(self, sums):
        ssums = sums.copy()
        ssums.sort()
        #print (sums)
        threshval = ssums[-self.P] 
        if threshval == 0:
            return np.argwhere(sums) 
        else:
            return np.where(sums >= threshval)



if __name__ == "__main__":
    # example usage
    
    tm = TriadicMemory()

    W = [10,11,12,13,14,15,16,17,18,19]
    X = [100,101,102,103,104,105,106,107,108,109]
    
    Y = [200,201,202,203,204,205,206,207,208,209]
    Z = [300,301,302,303,304,305,306,307,308,309]

    tm.store(W,X,Y)
    tm.store(X,Y,Z)

    print("W is ", W)
    print("X is ", X)
    print("Y is ", Y) 
    print("Z is ", Z)
    print("Expect W: ", tm.query(None, X, Y))
    print("Expect X: ", tm.query(W, None, Y)) 
    print("Expect Z: ", tm.query(X, Y))
