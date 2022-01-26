import numpy as np
class FHEncoder():

    def __init__(self, x = None, fname = None, random_state = None, sdr_size=2048, pixels_per_encoder=16): 
        if fname is None: 
            self.new_encoders(x.shape[1],sdr_size,pixels_per_encoder, random_state)
            self.init_factors(x)
        else:
            self.load(fname)

    def compute_sdrs(self, x, sdr_len = 32):
        scores = x[:,self.encoders].sum(axis=2) / self.factors
        sdrs = np.flip(np.argsort(scores),axis = -1)
        return sdrs[:,:sdr_len].astype(np.uint32)

    def new_encoders(self,input_size, sdr_size, pixels_per_encoder, random_state):
        encoders = [] 
        np.random.seed(random_state)
        for _ in range(sdr_size):
            encoders.append(np.random.permutation(input_size)[:pixels_per_encoder])
        self.encoders = np.array(encoders,dtype=np.uint32)
   
    def init_factors(self,x):
        factors = []
        tops = x.shape[0]//50
        for encoder in self.encoders:
            scores = x[:,encoder].sum(axis=1)
            order = np.argsort(scores)
            top2pcnt = order[-tops:]
            factors.append(scores[top2pcnt].mean())
        self.factors = np.array(factors).astype(np.float32)

    
    def load(self,fname):
        if fname.split('.')[-1] != "npz": 
            fname += ".npz"
        data = np.load(fname)
        self.encodings = data["encodings"]
        self.factors = data["factors"]

    def save(self, fname):
        np.savez(fname, encodings = self.encodings, factors = self.factors) 

if __name__ == "__main__":
    from load_mnist_data import x_train,normalize
    from time import time

    X = normalize(x_train)
    SDR_SIZE = 1024

    t = time()
    fhe = FHEncoder(X[:10000],sdr_size=SDR_SIZE)
    t = time() - t
    #fhe.load_from_dump("fpackteam1.txt")
    


    print(f"{fhe.encoders.shape[0]} encoders generated in {int(t*1000)}ms")
    print(f"encoders shape: {fhe.encoders.shape}")

    num = 10000

    t = time()
    sdrs = fhe.compute_sdrs(X[:num], sdr_len = 24)
    t = time() - t

    print(f"{num} sdrs of shape {sdrs.shape} computed in {int(t*1000)}ms")


    
