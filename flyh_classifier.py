

#
# This is an attempt to use htm classifier with fly hash encoder. 
# and check what scores we get on MNIST.
# The code is inspired by htm.core mnist example by replacing Spatial Pooler
# with FHEncoder

from fly_hash_encoder import FHEncoder
from load_mnist_data import x_train, y_train, x_test, y_test, normalize

from htm.bindings.algorithms import Classifier
from htm.bindings.sdr import SDR

import numpy as np
from time import time

# SDR_SIZE = 79*79   # That's the output SDR size in mnist example code.
SDR_SIZE = 6240
SDR_LEN  = SDR_SIZE // 13  #  sparsity as measured from htm.core mnist.py example
X_train = normalize(x_train)
X_test  = normalize(x_test)

fhe = FHEncoder(sdr_size = SDR_SIZE, spread = 900)

fh_train = np.zeros((X_train.shape[0], SDR_LEN), dtype = np.uint32)


print("Now computing fly hashes for x_train and x_test..", end = "")
tms = time()
blocks = range(0, len(X_train)+1, 10000) # 10k steps blocks to avoid memory exhaust in dot product
for b_start, b_end in zip(blocks[:-1], blocks[1:]):
    print(b_end, end = " ", flush=True)
    fh_train[b_start:b_end] = fhe.compute_sdrs(X_train[b_start:b_end], sdr_len = SDR_LEN)
fh_test  = fhe.compute_sdrs(X_test, sdr_len = SDR_LEN)

tms = int((time() - tms)*1000)
print(f"Done\nEncoding done in {tms}ms")


classifier = Classifier() 
sdr = SDR(SDR_SIZE)

# let's train the classifier directly on fly hash encoded mnist digits
print ("Begin training the classifier on fly hash encoded data")
tms = time()
for i, fh_sparse in enumerate(fh_train): 
    sdr.sparse = fh_sparse
    classifier.learn(sdr, y_train[i])

tms = int((time() - tms)*1000)
print (f"SDR Classifier training done in {tms} ms")

tms = time()
y_tested = y_test.copy() # Don't worry we'll overwrite these
for i, fh_sparse in enumerate(fh_test):
    sdr.sparse = fh_sparse
    y_tested[i] = np.argmax(classifier.infer(sdr))
tms = int((time() - tms)*1000)
print (f"SDR Classifier inference done in {tms} ms")


print(f"Correct results: {(y_tested == y_test).sum()/100}%")
