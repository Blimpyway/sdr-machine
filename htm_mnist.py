# ------------------------------------------------------------------------------
# This is a slight variation of Numenta's  mnist.py spatial pooler + sdr classifier example 
# The main differences are:
# 1. Replace the slow fetch_openml used in load_ds() with our local pickled loader load_mnist()
# 
# 2. Train in two separate stages: first the Spatial Pooler then the SDRClassifier
#   Reasons:
#   - Have a fair comparison baseline with flyh_classifier.py, where the
#     SDRClassifier is trained on already "trained" fly hash encoder outputs
#   - In a single stage the first digits of the SDRClassifier are trained on an "unoptimized"
#     Spatial Pooler. This stage sepparation improves accuracy by 0.1-0.2% 
#
# 3. Calculated average sparsity of Spatial Pooler in order to use the same sparsity in
#    flyh_classifier.py, again for fair comparison
#
# 4. imported time for measuring timings.
# 
# ------------------------------------------------------------------------------
""" An MNIST classifier using Spatial Pooler."""

import random
import numpy as np
import sys
from time import time

# Use our faster mnist loader
# from sklearn.datasets import fetch_openml

from htm.bindings.algorithms import SpatialPooler, Classifier
from htm.bindings.sdr import SDR, Metrics


def load_mnist():
    from load_mnist_data import x_train, y_train, x_test, y_test 
    return y_train, x_train, y_test, x_test

def encode(data, out):
    """
    encode the (image) data
    @param data - raw data
    @param out  - return SDR with encoded data
    """
    out.dense = data >= np.mean(data) # convert greyscale image to binary B/W.
    #TODO improve. have a look in htm.vision etc. For MNIST this is ok, for fashionMNIST in already loses too much information


# These parameters can be improved using parameter optimization,
# see py/htm/optimization/ae.py
# For more explanation of relations between the parameters, see 
# src/examples/mnist/MNIST_CPP.cpp 
default_parameters = {
    'potentialRadius': 7,
    'boostStrength': 7.0,
    'columnDimensions': (79, 79),
    'dutyCyclePeriod': 1402,
    'localAreaDensity': 0.1,
    'minPctOverlapDutyCycle': 0.2,
    'potentialPct': 0.1,
    'stimulusThreshold': 6,
    'synPermActiveInc': 0.14,
    'synPermConnected': 0.5,
    'synPermInactiveDec': 0.02
}


def main(parameters=default_parameters, argv=None, verbose=True):

    # Load data.
    train_labels, train_images, test_labels, test_images = load_mnist() # ?  HTM: ~95.6%
    # train_labels, train_images, test_labels, test_images = load_ds('mnist_784', 10000, shape=[28,28]) # HTM: ~95.6%
    #train_labels, train_images, test_labels, test_images = load_ds('Fashion-MNIST', 10000, shape=[28,28]) # HTM baseline: ~83%

    training_data = list(zip(train_images, train_labels))
    test_data     = list(zip(test_images, test_labels))
    random.shuffle(training_data)

    # Setup the AI.
    enc = SDR(train_images[0].shape)
    sp = SpatialPooler(
        inputDimensions            = enc.dimensions,
        columnDimensions           = parameters['columnDimensions'],
        potentialRadius            = parameters['potentialRadius'],
        potentialPct               = parameters['potentialPct'],
        globalInhibition           = True,
        localAreaDensity           = parameters['localAreaDensity'],
        stimulusThreshold          = int(round(parameters['stimulusThreshold'])),
        synPermInactiveDec         = parameters['synPermInactiveDec'],
        synPermActiveInc           = parameters['synPermActiveInc'],
        synPermConnected           = parameters['synPermConnected'],
        minPctOverlapDutyCycle     = parameters['minPctOverlapDutyCycle'],
        dutyCyclePeriod            = int(round(parameters['dutyCyclePeriod'])),
        boostStrength              = parameters['boostStrength'],
        seed                       = 0, # this is important, 0="random" seed which changes on each invocation
        spVerbosity                = 99,
        wrapAround                 = False)
    columns = SDR( sp.getColumnDimensions() )
    columns_stats = Metrics( columns, 99999999 )
    sdrc = Classifier()

    print("Begin training..", end="")
    tms = time()
    # Training Loop was split in 2: first train the spatial pooler, then train the classifier over the outputs
    # of already trained spatial pooler. 
    for i in range(len(train_images)):
        img, lbl = training_data[i]
        encode(img, enc)
        sp.compute( enc, True, columns )
    tms = int((time() - tms)*1000)
    print(f"spatial pooler trained in {tms}ms, now we train the classifier")
    bits_on, bits_size = 0, 0 
    tms = time()
    for i in range(len(train_images)): 
        img, lbl = training_data[i]
        encode(img, enc)
        sp.compute( enc, False, columns ) # Now the spatial pooler only outputs learned encodings
        bits_on += columns.sparse.size; bits_size += columns.dense.size
        sdrc.learn( columns, lbl ) #TODO SDRClassifier could accept string as a label, currently must be int
    tms = int((time() - tms)*1000)
    print(f"classifier trained in {tms}ms, average sparsity = {bits_on / bits_size}")

    print(str(sp))
    print(str(columns_stats))

    # Testing Loop
    score = 0
    for img, lbl in test_data:
        encode(img, enc)
        sp.compute( enc, False, columns )
        if lbl == np.argmax( sdrc.infer( columns ) ):
            score += 1
    score = score / len(test_data)

    print('Score:', 100 * score, '%')
    return score

# baseline: without SP (only Classifier = logistic regression): 90.1%
# kNN: ~97%
# human: ~98%
# state of the art: https://paperswithcode.com/sota/image-classification-on-mnist , ~99.9%
if __name__ == '__main__':
    sys.exit( main() < 0.95 )
