"""
Two point-based associative memory for SDRs

Unlike "canonical" associative memories it does not attempt to remember a "stored" SDR from a "query" SDR
Instead each SDR must have attached an uint32 ID which is spread across a (quite) large collision space. 

It is user's problem to generate and associate IDs with their SDRs, which from computers POVs is trivial even for 
large ammount of data. 
So as long as the IDs returned from queries are correct there is no problem retrieving initial, unscrambled SDRs
if that is waned.

But retrieving IDs instead of SDR may have some surprising advantages, e.g.
- if a large problem space is split into multiple "processors" the IDs can store addresses of these processors. 
- can memorise sequences if at SDR(state t) is stored the ID of SDR(state t+1)  
- ... more to investigate

The sdr map uses point pairs as multiple indices in a (quite) large table 

the table's addresses  occupy the area under the diagonal of a SIZE x SIZE matrix (SIZE is the SDR length)
Each address in this mapped space contains a SLOT which in this toy implementation can hold 64 IDs 

A point pair can be any two ON points (==1) in a SDR and each point pair maps to a unique address in the table
From a SDR with e.g. 32 bits ON, 32*31/2 = 496 point pairs can be stored in the map


"""
import numpy as np

def pairs2addr(plist):
    l0,l1 = plist.T
    return l0 + l1*(l1-1)//2
    

class SDRMap():
    def __init__(self, sdr_size = 2048, slot_size = 64):
        self.SDR_SIZE  = sdr_size
        self.NUM_SLOTS = pairs2addr(np.array([[sdr_size-2,sdr_size-1]]))[0]+1
        self.SLOT_SIZE = slot_size
        self.MAP = np.empty((self.NUM_SLOTS, slot_size), dtype = np.uint32)

        self.bit_pairs = {} 
        for size in range(5,120):
            pairs = []
            for b1 in range(0,size-1):
                for b2 in range(b1+1,size):
                    pairs.append((b1,b2))
            pairs = np.array(pairs,dtype = np.uint32)
            self.bit_pairs[size] = pairs

    def sdr2address(self,sdr,sdr_id=None): 
        sdr.sort()
        size = len(sdr) 
        pairs = self.bit_pairs[size]
        slots = pairs2addr(sdr[pairs])
        np.random.seed(sdr_id)
        slotpos = np.random.randint(0,self.SLOT_SIZE, size = len(slots))
        return slots, slotpos


    def store(self, id_list, sdr_list):
        """ 
        Stores in  sdr_mem a list of sdrs. 
        sdr_list - a list of tuple containing (id sdr) each
        """
        for i,sdr in zip(id_list, sdr_list):
            addr = self.sdr2address(sdr,i)
            self.MAP[addr] = i

    def query_extended(self,sdrs, min_counts = 4):
        """
        retrieves from sdr_mem a list of potential ids for a given sdr 
        Inputs:
            sdrs    - a list of query SDRs

            min_counts - returns ids for which were found more than min_counts pair_nodes

        returns a list of found ids and a list of corresponding counts for each sdr
            most encountered ids most likely match a previously stored id
        """
        value_list = []
        count_list = []
        for sdr in sdrs:
            addr, slotpos = self.sdr2address(sdr)
            vals = self.MAP[addr]
            values, counts = np.unique(vals, return_counts=True)
            which = counts > min_counts
            value_list.append(values[which])
            count_list.append(counts[which])
        return value_list, count_list

    def raw_query(self,sdr):
        addr, _ = self.sdr2address(sdr)
        return self.MAP[addr]

    def query(self, sdrs, first=4):
        """
        like raw_query but returns only the most significant results 
        and also removes the usually useless zero-ids from responses
        
        parameters:
        sdrs = the sdr list to query
        first = defaults to 4  best matching results.
        """
        id_list, count_list = self.query_extended(sdrs)
        id_out, count_out = [], []
        for i, sdr in enumerate(sdrs):
            ids = id_list[i]
            counts = count_list[i] 
            if ids.size < 2:
                id_out.append(ids)
                count_out.append(counts)
                continue
            if ids[0] == 0: 
                ids = ids[1:]
                counts = counts[1:]
            upto = min(first, len(ids))
            ordered = np.flip(np.argsort(counts))[0:upto]
            id_out.append(ids[ordered])
            count_out.append(counts[ordered])
        return id_out, count_out



if __name__ == "__main__":
    from time import time,sleep

    sdr_map = SDRMap(slot_size=112)

    def genRandomSDRs(numsdrs = 10000, sdr_bits = 32): 
        sdrs = np.zeros((numsdrs,sdr_bits), dtype=np.uint32)
        for i in range(numsdrs): 
            sdrs[i] = np.random.permutation(2048)[:sdr_bits]
        return sdrs

    def storeSpeedTest(sdrs, ids): 
        print(f"Begin write speed test .. ",end="")
        t = time()
        sdr_map.store(ids,sdrs)            ################## STORE ID UNDER SDR ADDRESS
        t = time() - t
        print(f"done writing {len(ids)} sdrs in {int(t*1000)}ms")


    def querySpeedTest(sdrs, query_bits=16): 
        sdrs = sdrs[:,:query_bits] ##### Using only half (16) bits for query
        print(f"Query {len(sdrs)} q_bits = {query_bits} ...", end='',flush=True)
        t = time()
        ids, counts = sdr_map.query(sdrs)  ################### FIND IDs associated with SDRs ##################
        t = time()-t
        print(f" Done reading in {int(t*1000)}ms")
        return ids, counts

        ##### RAW query comment return above if you want to see its output  ##################
        t=time()
        raw_out = []
        for sdr in sdrs[:,:8]:
            raw_out.append(sdr_map.raw_query(sdr))
        t=time()-t
        print(f"time for {len(raw_out)} raw_query: {int(t*1000)}ms, out line shape={raw_out[0].shape}")
        return ids, counts


    NUM_SDRS = 100_000
    print(f"generating {NUM_SDRS} sdrs...", end ='',flush=True)
    sdrs = genRandomSDRs(NUM_SDRS)
    print("done!")
    ids = np.arange(0, NUM_SDRS)
    istart, iend = 0, 10000
    while istart < NUM_SDRS:
        storeSpeedTest(sdrs[istart:iend], ids[istart:iend])
        istart, iend = iend, iend + (iend - istart)
        iend = min(iend, NUM_SDRS)

    l_ids, l_counts = [], []
    istart, iend = 0, 10000 
    NUM_SDRS = NUM_SDRS // 10
    while istart < NUM_SDRS  :
        ids, counts = querySpeedTest(sdrs[istart:iend], query_bits=16)
        l_ids.append(ids)
        l_counts.append(counts) 
        istart, iend = iend, iend + (iend - istart)
        iend = min(iend, NUM_SDRS)

    # print("qsdr:" , qsdr)
    for i in (500,501,502,100,101,102):
        print("result for:", i, " ids:" , l_ids[0][i], " counts:", l_counts[0][i])

