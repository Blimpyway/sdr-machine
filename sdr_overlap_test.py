
summary = """
This test compares overlap counting speed in numba vs htm core

The numba version uses a "ordered sparse" representation, which means 
ordered positions of 1 bits

Use it with: 
    $ ipython -i sdr_overlap_test.py

    %timeit overlap(a,b)

    %timeit sdr_a.getOverlap(sdr_b)

It also tests different sdr distance metrics which will be tested with pynndescent.

"""

import numba
from htm.bindings.sdr import SDR
import numpy as np

import pynndescent

# This is a "naive" python implementation which "compiles" well in numba
def py_overlap(n1,n2):
    out = 0
    i1, i2 = 0, 0
    while i1 < n1.size and i2 < n2.size:
        v1, v2 = n1[i1], n2[i2]
        if v1 == v2:
            out += 1
            i2 += 1
        elif v1 > v2:
            i2 += 1
            continue
        i1 += 1
    return out

# This is first step of compilation, next one is to call it once so numba can generate a typed prototype
overlap = numba.njit(py_overlap, fastmath=True)

# Here are various distance functions. 
# Unlike overlap, distance gets smaller as the two vectors are closer

# overlap subtrated from the smallest size of the two vectors.
# The problem with this is sdr + extra bits on is as close as itself (zero distance)
@numba.njit(fastmath=True)
def d_min(n1, n2): 
    return min(n1.size, n2.size) - overlap(n1,n2)

# To "fix" the above here we subtract the overlap from the average size of the two vectors. 
@numba.njit(fastmath=True)
def d_avg(n1,n2):
    return (n1.size + n2.size) / 2 - overlap(n1,n2)

# And here is a ratio version which has a higher increase in distance with the smaller overlap 
@numba.njit(fastmath=True)
def d_ratio(n1,n2):
    # The 0.1 below avoids division by zero (on no overlap) 
    # The -.998 makes distance close to zero on "perfect" overlap
    return (n1.size + n2.size) / (2 * overlap(n1,n2) + 0.1)  - .998

sparse_random = lambda size, nbits: np.random.permutation(size).astype(np.uint16)[:nbits]

# Generate two random SDRs with 100/2000 sparsity
a = sparse_random(2000,100)
b = sparse_random(2000,100)
a.sort(), b.sort()

sdr_a = SDR(2000)
sdr_a.sparse = a
sdr_b = SDR(2000)
sdr_b.sparse = b 

# Testing they compute the same result
print(
        sdr_a.getOverlap(sdr_b) == overlap(b,a)
    )


