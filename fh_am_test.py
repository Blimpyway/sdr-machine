
"""
Testing fly_hash encoder and associative memory to test MNIST retrieval accuracy

We get >%91 accuracy using first 20k train digits
This is just a poor replacement for knn nearest neighbor. 

Accuracy losses may be caused by both FHEncoder and associative memory overlaps.

"""
from sdr_mem2d import SDRMap
from fly_hash_encoder import FHEncoder
from load_mnist_data import x_train, x_test, y_train, y_test, normalize

x_train = normalize(x_train) # Our normalize aranges all images flattened and all image sum()-s are 1.0 
x_test  = normalize(x_test)

import numpy as np
from time import time



# Which x_train digits will be stored in memory
istart,ilen = 0,60000 
#istart,ilen = 0,20000 
iend   = istart + ilen
SDR_size = 2048
SLOT_size = 112

# Use this many bits to store train data in memory
store_sdr_len = 32  # The length of stored SDRs

# Use these many bits to query sdr memory
query_sdr_len = 32  # The length of query SDRs

# how many input pixels are "watched" by every output pixel
encoder_size = 200


# Initialise the memory 
print(f"Initialize SDRMap sdr_size={SDR_size}, slot_size={SLOT_size}")
print(f"Will use sdr lengths for storage: {store_sdr_len}; for query: {query_sdr_len}")
smap = SDRMap(sdr_size = SDR_size,slot_size = SLOT_size)
# Create a FlyHash encoder. The hasher converts a alist of MNIST image to a SDR list of size 2048
print(f"Generate Flyhash encoder, pixels_per_encoder={encoder_size}...")
hasher = FHEncoder(sdr_size = SDR_size, random_seed = 3, pixels_per_encoder = encoder_size)

print(f"hasher initialised, we use it to convert {ilen} x_train images to SDRs")
t = time()
sdrs = hasher.compute_sdrs(x_train[istart:iend],sdr_len = store_sdr_len)
t = time() - t
print(f"{ilen} sdrs computed in {int(t*1000)}ms")
print(f"Training sdrs.shape:{sdrs.shape}, dtype:{sdrs.dtype}")


# IDs will encode both y_train values and train index (position in x_train)
sdr_ids = y_train[istart:iend] + np.arange(istart,iend) * 100 + 10000000
# If above looks weird, here-s an  example of what it does: if y_train[15632] == 7 then the ID becomes: 
# 11563207
# _NNNNN_D   - where '_' are ignored 'D' is the digit value and NNNNN is the row num in the x_train array

print(f"Begin storing {ilen} SDRs in associative memory (a.k.a sdr map)")
t = time()
smap.store(sdr_ids,sdrs)
t = time()-t
print(f"sdrs stored in {int(1000*t)}ms")

# "training" done 

# Testing 

xtest = x_test
ytest = y_test
print("Begin querrying memory map with x_test")
t=time()
sdrs = hasher.compute_sdrs(xtest, sdr_len=query_sdr_len)
idlists, idcounts = smap.query(sdrs,first=8)
t = time()-t
print(f"Associative query {len(idlists)} done in {int(t*1000)}ms")

print(f"\nNext we retrieve predicted digit numbers from the responses")
t = time()
idresults = []

for idlist, idcount in zip(idlists, idcounts):
    results = np.zeros(10)
    for i in range(len(idlist)):
        pred = idlist[i] % 100 # restore digit value from id.
        results[pred] += idcount[i]
    idresults.append(np.argmax(results))
idresults = np.array(idresults)
# print(f"idresults.sum() {idresults.sum()}, dtype={idresults.dtype}")

comps = idresults == ytest
t=time()-t

print(f" result len = {len(comps)}, positive = {comps.sum()/100}%, computed in {int(t*1000)}ms")
