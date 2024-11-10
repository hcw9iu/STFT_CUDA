# STFT CUDA Acceleration
This project implements Short-Time Fourier Transform (STFT) using CUDA for effecient computation on GPUs. 
It includes one CUDA module and a test python script.

## Features
* Implement STFT in CUDA.
* Compare Python and CUDA version time consumption.

## Requirements
* CUDA Toolkit
* CMake
* Python 3.x
* pybind11
* `numpy` , `matplotlib` , `perfplot`
## Installation
1. Clone the repo
2. Check compiler version *(gcc after 11 not supported)*
    ```bash
    which gcc #(returns default, may be newer than 11)
    which gcc-11 #(returns gcc 11)
    ```
4. Build *(with specifying compiler)*
    ```bash
    mkdir build
    cd build
    cmake --fresh \\
    -DCMAKE_C_COMPILER=/usr/bin/gcc-11 \\
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 ..
    make
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
N = 256
Fs = 1000
...
```
## Benchmarking
Use `test_cuda.py` script to benchmark CUDA and Python implementation.
```bash
python test_cuda.py
```
**Performance**
![](https://wdnmd-nft.infura-ipfs.io/ipfs/QmPESJ5B3u8DDXEAhJfjptSgN4EDrwBfevwqiBabmyGBZi)
**STFT Result**
![](https://wdnmd-nft.infura-ipfs.io/ipfs/QmSrSaUvpgPUEJkopTqVHTycXs1izC4SiLRLhixyH5vECD)
## Additional Resource
For more information visit 
