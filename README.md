# SDR Machine experimental ideas

* explore  possibilities to build an SDR associative memory 
* fly hash encoder - transforms any vectorized data into a SDR. Losely similar with HTM Spatial Pooler, only the "learning" part is much more rudimentary


## Files:

### Fly hash and "2D" associative memory

* fly_hash_encoder.py - an approximative implementation of Fly Hash encoder. 
	It is used to convert (reatively) low sparsity mnist digits to 2048 bit sparse distributed representation
	Since memory performance degrades with the square of 1 bits, the SDRs tend to be low <1-2% sparsity
* load_mnist_data.py - quick&dirty loader of numpy savez mnist digits
* mnist_data.npz  - the actual mnist files (x_test, y_test, x_train, y_train) 
* sdr_mem2d.py - The actual 2d associative memory code see how it works below 
* fh_am_test.py - The main program using fly hash encoded mnist digits with the associative memory 

### Testing  fly hash with HTM SDR Classifier.

* flyh_classifier.py - a rewrite of HTM mnist.py classifier example, using fly hash encoder instead of Spatial Pooler. 
* htm_mnist.py - a modified copy of HTM mnist.py example, exact changes are detailed in the file comments.

## Requirements:
Since mnist loader is included, fh_am_test.py depends only on python3 and numpy

For classifier tests htm.core is needed  


## Running: 

The main demo program is fh_am_test.py. 
 
  $ python3 fh_am_test.py

It reads in MNIST digits, then queries same memory for all 10000 test digits, use results to "vote" on result's digit value. 
It only counts sdr bit pair matches in associative memory to output results without any 
distance measurements between the actual MNIST encoding of queries and responses.  

Every other .py program here runs its own if __name__ == "__main__" stuff for testing.
e.g. sdr_mem2d.py runs a performance test on storing 100k random SDRs

## How is a SDR encoded and indexed

SDR Size:   the number of all (ON or OFF) bits in any SDR 
SDR Length: the number of ON (==1) bits

The associative memory and fly hash encoder operate with sparse representation of SDRs, which is an ordered list
of ON bit positions.
We prefer to use SDRs with equal lengths (same number of ON bits), to exploit numpy array processing 

Index is built by expanding ON bits in pairs. 
e.g. if a SDR has the bits [15, 64, 429, 900] its pair expansion is: 
[(15,64), (15,429), (15,900), (64,429), (64,900), (429,900)]

If the SDR Size is N, the number of possible bit pairs is N\*(N-1)/2 

Each pair above projects into a memory SLOT. Each SLOT stores a number of ID positions. This is the SLOT_Size
Default slot size is 64. 

An ID is just a 32bit int, used by programmers to connect it with meaningful data. e.g. in the examples the ID encodes
both the 0..9 digit and the 0..59999 position of the digit in x_train

Memory Size is total number of SLOTs == how many possible bit pairs of SDRs of size N 
e.g. SDR Size is 1024 then memory size is 1024\*1023/2 =~ 512k number of SLOTs

Its actual RAM usage in bytes is found by multiplying the 
(Memory size) x (SLOT size) x (int32 size)

e.g. for N = 1024 and SLOT_size = 64 we get :

1024 \* 1023/2 \* 64 \* 4 =~ 128 MBytes 

If we double the SDR size we get an 4x increase in memory needed to store the whole index.

Projecting an SDR and corresponding ID into the memory: 

* build the list of SDR slots by expanding SDR's ON bits in pairs. 
* At each SLOT in the above list the corresponding ID is in one of available positions
* position within a slot is chosen to be reproductible - same ID will be written at same position in the same slot. 
  
Example:
When writing  32 bit long  SDRs, they will have their IDs spreaded in  32 * 31 / 2 = 496 SLOTS. 

Which means there is a certain likelihood that after significant amount of overwrites, the original ID can be restored from remaining slots when queried. 


## How it is used

Sparse encoding: All SDRs here are represented as lists of corresponding 1 bit positions in the 0..2047 bit SDR space. 
e.g a SDR of 30  one bits is a numpy array of 30 int32 values.

The SDR associative memory here is actually a mapper between int32 "ids" and SDRs  
To store ids and sdr use an instance of SDRMap: 

SDRMap.store(sdr_id_list, sdr_list)

sdr_id_list has to by a 1d numpy int32 array, 
sdr_list has to be a matching 2d array, one sparse encoded sdr per row for every id in the sdr_id_list


When queried the memory does not return actual SDRs but (what it considers to be) best matching 32bit sdr_id-s
used during store(). Is the user's responsibility to encode/decode "meaning" onto these ids.

SDRMap.query(sdr_list, max_ids = 4)

For each sdr in sdr_list returns a list of most frequent sdr_id and number of index hits. The list length does not exceed max_ids 


SDRMap.queryExtended(sdr_list, min_hits = 4)  
For each sdr in sdr_lists returns a list of sdr_id and number of index hits. Drops out ids with less than min_hits hits

## TLDR

The above explanations are quite ... raw, sorry. I'll hopefully get time to clarify things. 
