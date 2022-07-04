"""
Value Correlator Map for SDRs 

The Value Correlator Map here is a bitpair corellation map. 
It's main operations are "adding" of a SDR and "querying" 

The principle is very simple: 
    The SDR is self expanded in  bitpairs. 
    At each corresponding bitpair address in the map a scalar is incremented with
    the specified "add" value 

    Queries are performed against same or different SDRs and are two kinds:
    query and score 
    - query() retrieves values for each bitpair expanded from the query SDR
    - score() used for convenience computes average query result for the whole SDR

The pretentious name comes from the fact that it *highlights correlations*
between all pairs of bits within a *SDR space*

In score() mode a value map can be seen like a key->scalar dictionary where keys are SDRs and values are simple scalars

====================================================================================

@Copyright 2022 Cezar Totth 

Use it as you like
"""

import numpy as np
import numba

@numba.njit
def addr2(sdr):
    # Projects sdr into a plane
    for x in range(1,sdr.size):
        xv = sdr[x]*(sdr[x] - 1) // 2
        for y in range(x):
            yield (x,y), xv + sdr[y]

@numba.njit
def _value_add2(sdr, value_map, value): 
    """
    increments value_map by value at sdr's 2d address points
    """
    msize = value_map.shape[0]
    num_points = 0
    for xy, addr in addr2(sdr):
        value_map[addr % msize] += value
        num_points += 1
    return num_points

@numba.njit
def _value_query2(sdr, value_map): 
    msize = value_map.shape[0]
    for xy, addr in addr2(sdr):
        yield xy, value_map[addr % msize]

@numba.njit
def _value_score2(sdr, value_map): 
    num_points = 0
    vsum = 0
    for xy, value in _value_query2(sdr, value_map):
        num_points += 1
        vsum += value
    return vsum / num_points

class ValueCorrMap:
    def __init__(self, sdr_size = None, mem_size = None):
        """
        at least one of sdr_size or mem_size should be specified

        sdr_size input: memory size is computed as "canonical size" 
                aka the number of available bitpairs in the sdr_size space
                which is the sub-diagonal area of the square of  sdr_size width

        mem_size input: 
            if sdr_size is also specified,
                then mem_size is considered as a not-to-exceed size in bytes
            if sdr_size is not specified, 
                the memory map is created such its size in bytes matches this value

        """
        if sdr_size is None:
            assert mem_size is not None
            mem_size = mem_size // 4
        elif mem_size is None:
            assert sdr_size is not None
            mem_size = sdr_size * (sdr_size - 1)//2
        else:
            sz1 = mem_size // 4
            sz2 = sdr_size * (sdr_size - 1) // 2
            mem_size = min(sz1, sz2)

        self.vmap = np.zeros(mem_size, dtype = np.int32)
        self.totals = 0

    def score(self, sdr):
        return _value_score2(sdr, self.vmap)

    def add(self, sdr, value = 1):
        """
        Increments value map with specified value on all sdr's bit pairs.
        returns the total value added and sum of all values into the map.
        """
        plus = _value_add2(sdr, self.vmap, value)
        self.totals += plus
        return plus, self.totals

    def query(self, sdr):
        """
        returns an iterator over individual values for each bit pair in sdr. 
        The results can be used to highlight bit pairs with unusual values.
        each step yields a tuple consisting of bit pairs and corresponding values
        """
        return _value_query2(sdr, self.vmap)

    def mem_size(self):
        return self.vmap.size * 4

    def mean(self): 
        """
        computes mean value in value map. 
        It can be used as comparison metric for queried bit pairs or SDRs 
        """
        return self.totals / self.vmap.size

@numba.njit
def addr3(sdr):
    # Projects sdr into a cube. Not used yet, a 3bit map was the original idea
    for x in range(2,sdr.size): 
        sx = sdr[x]
        xv = sx * (sx-1) * (sx-2) // 6
        for y in range(1,x):
            sy = sdr[y]
            yv = sy * (sy-1) // 2 
            for z in range(y):
                yield xv + yv + sdr[z], (x,y,z)

@numba.jit
def test_addr3(sdrs): 
    naddrs = 0
    for sdr in sdrs:
        for a, t in addr3(sdr):
            naddrs+=1
    return naddrs

@numba.jit
def test_addr2(sdrs):
    naddrs = 0
    for sdr in sdrs:
        for a,t in addr2(sdr):
            naddrs += 1
    return naddrs

@numba.jit
def test_map_query2(sdrs, vmap):
    results = []
    for sdr in sdrs:
        qresult = _value_score2(sdr, vmap)

        results.append(qresult)
    return results

if __name__ == "__main__":
    from sdr_util import random_sdrs
    from time import time

    SDR_SIZE = 200
    SDR_LEN  =  28  # 200/28 was quite successful in cartpole example

    MAP_SIZE = SDR_SIZE * (SDR_SIZE - 1) // 2
    vmap = ValueCorrMap(sdr_size = SDR_SIZE)
    value_map = vmap.vmap
    print(f"value map shape: { value_map.shape }")

    num_sdrs = 1000000
    sdrs = np.array(random_sdrs(num_sdrs, SDR_SIZE, SDR_LEN))

    t = time()
    for i in range(num_sdrs):
        vmap.add(sdrs[i],np.random.randint(1,10))
    t = int((time() - t) * 1000)
    print(f"{num_sdrs} of size/len {SDR_SIZE}/{SDR_LEN} added in {t} ms")
    qresults = test_map_query2(sdrs[:100], vmap.vmap)
