import numpy as np
class FHEncoder():

    def __init__(self, file_name = None, random_seed = 1, sdr_size=2048, pixels_per_encoder=32): 
        if file_name is None: 
            self.sdr_size = sdr_size
            self.pixels_per_encoder = pixels_per_encoder
            self.random_seed = random_seed
            self.dot_encoders = None
        else:
            self.load(file_name)

    def generate_dot_encoders(self, input_size):
        np.random.seed(self.random_seed)
        dot_enc_shape = (input_size, self.sdr_size) 
        dot_encoders = np.zeros(dot_enc_shape, dtype = np.float32)
        dot_encoders[:,:self.pixels_per_encoder] = 1
        for line in dot_encoders:
            np.random.shuffle(line)
        # print(f"Generated self.dot_encoders of shape {dot_encoders.shape}")
        self.dot_encoders = dot_encoders
       
    def compute_sdrs(self, x, sdr_len = 32): 
        # Use .dot variant since gets faster as # of pixels_per_encoder increases
        if self.dot_encoders is None:
            self.generate_dot_encoders(x.shape[1])
            self.init_factors(x)
        scores = x.dot(self.dot_encoders) 
        sdrs = np.flip(np.argsort(scores / self.factors),axis = -1)
        return sdrs[:,:sdr_len].astype(np.uint32)
        
    def init_factors(self,x): 
        # This attempt to "skip" factors leads to a huge drop in accuracy from 94% to 90%
        tops = x.shape[0] // 50
        dotscores = x.dot(self.dot_encoders).T
        dotscores.sort(axis=1)
        dotscores = dotscores[:,-tops:].sum(axis=1)
        self.factors = dotscores / dotscores.mean()

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
    SDR_SIZE = 2048

    fhe = FHEncoder(sdr_size=SDR_SIZE)
    

    num = 10000

    t = time()
    sdrs = fhe.compute_sdrs(X[:num])
    t = time() - t
    print(f"{num} sdrs of shape {sdrs.shape} computed in {int(t*1000)}ms")

    # print(fhe.factors)
