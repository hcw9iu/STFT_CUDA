# STFT CUDA Acceleration

[![hackmd-github-sync-badge](https://hackmd.io/vnXeF_LCS9CxYbE6mV8_mw/badge)](https://hackmd.io/vnXeF_LCS9CxYbE6mV8_mw)

This project implements Short-Time Fourier Transform (STFT) using CUDA for effecient computation on GPUs. 
It includes one CUDA module and a test python script.

## Features
* Implement STFT in CUDA.
* Compare Python and CUDA version time consumption.
* Burn performance on different parameters.

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
3. Install Python Dependencies

    ```bash
    pip install -r requirements.txt
    ```
4. Build CUDA Extension *(with specified compiler)*

    ```bash
    cmake -B build \ 
        -DCMAKE_C_COMPILER=gcc-11 \
        -DCMAKE_CXX_COMPILER=g++-11 .
    ```
    or
    ```
    cmake --build build 
    ```

## Usage
Python Setups
```python
import numpy as np
import matplotlib.pyplot as plt
from build import stft_cuda

l = 1024    #signal length
window_size = 256
hop = 128
N = 256     #fft size
Fs = 1000   #sampling rate
```
## Benchmark
Run scripts.
```bash
python test_cuda.py
python test_hop.py
python test_win.py
```
### Performance
Performance compare on Python and different CUDA implementations:
- CUDA: Original design.
- CUDA Align: Shared memory on window function. 
- CUDA Async: Strean design.

**Length Burning**
![CUDA vs Python Performance Benchmark](https://wdnmd-nft.infura-ipfs.io/ipfs/QmdYqn19guY4ZpG8hQkpnnXnLGDfidvNFeYG6K9iBcbQpj)

- Testing large signal length at `hop_size = 32` and `win_size = 256`
- CUDA is 10 times faster.

**Burning Hop Size**
![](https://wdnmd-nft.infura-ipfs.io/ipfs/Qmb1pKNk3QMwHpMk4idNcYee8ikKZZMKAJsFrs8KhNbeZN)
- Testing smaller hop size at fixed length and window size.
- CUDA is in greater advantage for even smaller hop size.

**Burning Window** 


## Equality Check
Comparison of STFT results between CUDA and Python implementations:

![STFT Result Comparison](https://wdnmd-nft.infura-ipfs.io/ipfs/QmSrSaUvpgPUEJkopTqVHTycXs1izC4SiLRLhixyH5vECD)

Both implementations produce identical results, validating the correctness of the CUDA implementation.

## Additional Resource
For more information visit 
