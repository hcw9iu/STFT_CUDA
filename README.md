# STFT CUDA Acceleration

[![hackmd-github-sync-badge](https://hackmd.io/vnXeF_LCS9CxYbE6mV8_mw/badge)](https://hackmd.io/vnXeF_LCS9CxYbE6mV8_mw)

This project implements Short-Time Fourier Transform (STFT) using CUDA for effecient computation on GPUs. 
It includes one CUDA module and a test python script.

## Features
* Implement STFT in CUDA.
* Compare Python and CUDA version time consumption.

## Building from Source
### Prerequisites
- CUDA Toolkit (>= 11.0)
- Python (>= 3.8)
- CMake (>= 3.10)
- C++ Compiler (g++ >= 11.0)
### Build Steps
1. Clone the repo
2. Prepare environment using conda
    ```bash
    conda install -c conda-forge cmake pybind11
    conda install -c nvidia cuda-toolkit
    ```
3. Check compiler version *(g++ after 11 not supported)*

    ```bash
    which g++ #(returns default, may be newer than 11)
    whereis g++-11 #(returns g++ 11)
    ```

4. Install Python Dependencies

    ```bash
    pip install -r requirements.txt
    ```
5. Build CUDA Extension *(with specified compiler)*

    ```bash
    cmake -B build \ #create and build in folder "build"
        -DCMAKE_C_COMPILER=gcc-11 \
        -DCMAKE_CXX_COMPILER=g++-11 .
    cmake --build build #make in "build" folder
    ```

## Usage
Python Setups
```python
import numpy as np
import matplotlib.pyplot as plt
from build import stft_cuda

l = 1024
window_size = 256
hop = 128
N = 256     #fft size
Fs = 1000   #sampling rate

s = np.random.rand(l).astype(np.float32)
t = np.arange(l)/Fs
c = np.sin(2*np.pi*200*t)
x = c*s #input signal

...

out_shape = (N,seg)
out = np.empty(out_shape, dtype=np.complex64)
stft_cuda.stft(x,out,window_size,hop,N,Fs)

out = out[:1+N//2,...] #only keep the positive frequencies
```
## Benchmarking
Use `test_cuda.py` script to benchmark CUDA and Python implementation.
```bash
python test_cuda.py
```
### Performance
The following graph shows the performance comparison between CUDA and Python implementations:

![CUDA vs Python Performance Benchmark](https://wdnmd-nft.infura-ipfs.io/ipfs/QmPESJ5B3u8DDXEAhJfjptSgN4EDrwBfevwqiBabmyGBZi)

- CUDA implementation shows significant speedup for larger inputs
- Performance scales well with input size

### STFT Result
Comparison of STFT results between CUDA and Python implementations:

![STFT Result Comparison](https://wdnmd-nft.infura-ipfs.io/ipfs/QmSrSaUvpgPUEJkopTqVHTycXs1izC4SiLRLhixyH5vECD)

Both implementations produce identical results, validating the correctness of the CUDA implementation.

## Additional Resource
For more information visit 
