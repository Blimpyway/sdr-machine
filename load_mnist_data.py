import numpy as np
import os

dname = os.path.dirname(__file__) 
if len(dname) == 0:
    dname = '.'

_d = np.load(dname + "/mnist_data.npz") 
x_test,  y_test  = _d['x_test'], _d['y_test']
x_train, y_train = _d['x_train'],_d['y_train']

def normalize(X):
    """
    transforms X so sum() of each digit is 1. 
    X must be adimensional, e.g. for MNIST each digit of shape (28,28) should be reshaped to (784,)
    """
    X = X.reshape(X.shape[0],-1)
    sums = X.sum(axis=1)
    return (X.T / sums).T.astype(np.float32)

if __name__ == "__main__":
    # np.savez("mnist_data", x_test = x_test, x_train = x_train, y_test = y_test, y_train = y_train)
    print(f"mnist data loaded in {[k[0] for k in _d.items()]}")
    print({normalize(x_test).sum()})
