
"""
This is an attempt to make a cleaner, numbified sdr_mem2d.py version
"""

import numba
import numpy as np 


@numba.jit(nopython = True)
def _addr(x, mem):
    num_slots = mem.shape[0]
    addr = []
    for i in range(1,len(x)):
        for j in range(i):
            addr.append((x[i]*(x[i]-1)//2 + x[j]) % num_slots)
    return addr

@numba.jit
def save(mem, x, yid): 
    slotsize = mem.shape[1]
    for a in _addr(x, mem): 
        mem[a,(yid * a) % slotsize] = yid

def _query(mem, x, thresh = 5):
    """
    For whatever reason there-s a problem numbifying this. Probably numba couldn't
    outperform np.unique()
    since performance matches closely _id_counter() below I suspect numpy.unique() uses
    a similar algorithm as _id_counter()
    """
    flat = mem[_addr(x,mem)].flatten()
    unique, counts = np.unique(flat, return_counts = True)
    return unique, counts

@numba.jit(nopython = True)
def _id_counter(mem, x, thresh = 5):
    found = {}
    for a in _addr(x, mem):
        for yid in mem[a]:
            if not yid: # skip zeros
                continue
            if yid in found:
                found[yid] += 1
            else:
                found[yid] = 0
    ret = []
    
    for yid, cnt in found.items():
        if cnt > thresh:
            ret.append((cnt,yid))
    return sorted(ret, reverse=True)


@numba.jit(fastmath=True, nogil=True, cache=True)
def _bit_query(mem, x, thresh): 
    """
    unlike _id_counter this uses a bitmap to remove 
    """
    bmsize = 2**14
    bitmap = np.zeros((bmsize), dtype = np.uint64)
    whlist = []
    bval = numba.uint64(0)
    def bittest(val, bit):
            return (val // 2 ** bit) % 2
    cnt = 0
    for a in _addr(x, mem):
        cnt += 1
        # print(cnt)
        for yid in mem[a]:
            if not yid:
                continue
            bit  = yid % 64
            bpos = yid % bmsize
            if bittest(bitmap[bpos], bit) :
                whlist.append(yid)
            else:
                bitmap[bpos] = bitmap[bpos] + 2 ** bit
    found = {}
    for yid in whlist:
        if yid in found: 
            found[yid] += 1
        else:
            found[yid] = 0

    ret = []
    for yid, cnt in found.items():
        if cnt > thresh:
            ret.append((cnt, yid))
    return sorted(ret, reverse = True)



class SDR_MEM:
    def __init__(self, mem_size, slot_size = 31):
        num_slots = mem_size // (slot_size * 4) # Mem size would be specified in bytes. It won't be implicit, users have to allocate it.
                                                # 4 is the size in bytes of an id - np.uint32
        self.mem = np.zeros((num_slots, slot_size), dtype = np.uint32)

    def store(self, sdr, sid): 
        save(self.mem, sdr, sid)

    def query(self, sdr, thresh = 5):
        """
        query sdr in mem with answers more frequent than thresh bitpair hits
        """

        # return _id_counter(self.mem, sdr, thresh)
        # return _query(self.mem, sdr, thresh)
        return _bit_query(self.mem, sdr, thresh)

    def num_slots(self):
        # since number of slots are computed dynamically in __init__() from mem_size and slot_size 
        # here-s a co
        return self.mem.shape[0]

    def min_sdr_size(self):
        """ 
        recommended minimum size sdr in order to adequately use the num_slots wide space
        """
        return int((self.num_slots() * 2) ** .5 + 1)

def random_sdrs(num_sdrs, sdr_size, on_bits): 
    tor = np.zeros((num_sdrs, on_bits), dtype = np.uint32)
    a = np.arange(sdr_size, dtype=np.uint32)
    for l in tor: 
        np.random.shuffle(a) 
        l += sorted(a[:on_bits])
    return tor

if __name__ == "__main__":
    mem_size = 1_000_000_000 # How big a memory to allocate in bytes
    sdr_size =   10000
    bit_size =      20      # sdr solidity
    num_sdrs = 1000000     # How many sdrs to store
    qry_sdrs = 1000000     # How many sdrs to query
    slot_size =   23

    mem = SDR_MEM(mem_size, slot_size = slot_size)
    print(f"Testing {num_sdrs} entries AM with slot_size={slot_size} and {bit_size}/{sdr_size} bit sdrs")

    print(f"Minimum sdr size: {mem.min_sdr_size()}")
    from time import time
    t = time()
    sdrs = random_sdrs(num_sdrs, sdr_size, bit_size)
    t = time() - t
    print(f"{len(sdrs)} random sdrs generated in {int(t*1000)} ms")
    print("mem size:" , mem.mem.size * 4 // 2 ** 20)
    t = time()
    for num, sdr in enumerate(sdrs): 
        # print(num, sdr)
        mem.store(sdr, num+1)               # Here it is how to store into memory
    t = time() - t

    print(f"{len(sdrs)} inserted in {int(t*1000)} ms")

    print("Testing query speed")
    results =[]
    t = time()
    for sdr in sdrs[:qry_sdrs]:
        resp = mem.query(sdr)              # Here it is how one queries
        results.append(resp)
    t = time() - t
    print(f"{len(sdrs)} queries in {int(t*1000)} ms")

