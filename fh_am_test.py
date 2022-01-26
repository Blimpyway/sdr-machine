
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


smap = SDRMap(slot_size = 112)

# Create a FlyHash encoder. The hasher converts a alist of MNIST image to a SDR list of size 2048
# x_train here is given to adjust "biases" to equalise SDR bits chances to produce 1
hasher = FHEncoder(x_train[10000:20000])

istart,ilen = 0,20000 # Which x_train digits will stored in memory
iend   = istart + ilen

print(f"hasher initialised, we use it to convert {ilen} x_train images to SDRs")
t = time()
sdrs = hasher.compute_sdrs(x_train[istart:iend],sdr_len = 25)
t = time() - t
print(f"{ilen} sdrs computed in {int(t*1000)}ms")
print(f"Training sdrs.shape:{sdrs.shape}, dtype:{sdrs.dtype}")


# This encodes both y_train values and train index (position in x_train
sids = y_train[istart:iend] + np.arange(istart,iend) * 100 + 10000000
# If above looks weird, here-s an  example of what it does: if y_train[15632] == 7 then the ID becomes: 
# 11563207
# _NNNNN_D   - where '_' are ignored 'D' is the digit value and NNNNN is the row num in the x_train array

print(f"Begin storing {ilen} SDRs in associative memory (a.k.a sdr map)")
t = time()
smap.store(sids,sdrs)
t = time()-t
print(f"sdrs stored in {int(1000*t)}ms")


# uncomment these blocks if you want to encode&store all 60000 x_train digits
istart,ilen = 20000,20000
iend = istart+ilen
sdrs = hasher.compute_sdrs(x_train[istart:iend],sdr_len = 40)
sids = y_train[istart:iend] + np.arange(istart,iend) * 100 + 10000000
smap.store(sids,sdrs)

istart,ilen = 40000,20000
iend = istart+ilen
sdrs = hasher.compute_sdrs(x_train[istart:iend],sdr_len = 40)
sids = y_train[istart:iend] + np.arange(istart,iend) * 100 + 10000000
smap.store(sids,sdrs)

# "training" done 

# Testing 

#xtest = x_train[:-10000]
#ytest = y_train[:-10000]
xtest = x_test
ytest = y_test
print("Begin querrying memory map with x_test")
t=time()
sdrs = hasher.compute_sdrs(xtest, sdr_len=25)
idlists, idcounts = smap.query(sdrs,first=6)
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
