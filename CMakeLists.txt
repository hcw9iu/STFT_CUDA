cmake_minimum_required(VERSION 3.10)
project(stft_cuda LANGUAGES C CXX CUDA)

find_package(CUDAToolkit REQUIRED)

#add_library(stft_cuda SHARED stft_cuda.cu)
#target_link_libraries(stft_cuda CUDA::cufft)
#set(CMAKE_PREFIX_PATH "/home/hcw/anaconda3/envs/v38/lib/python3.8/site-packages/pybind11/share/cmake/pybind11")

set(CMAKE_C_COMPILER /usr/bin/gcc-11)
set(CMAKE_CXX_COMPILER /usr/bin/g++-11)
set(CUDA_HOST_COMPILER /usr/bin/gcc-11)

execute_process(COMMAND python3 -m pybind11 --cmakedir OUTPUT_VARIABLE pybind11_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
set(CMAKE_PREFIX_PATH "${pybind11_DIR}")
find_package(pybind11 REQUIRED)

pybind11_add_module(stft_cuda stft_cuda.cu)

target_link_libraries(stft_cuda PRIVATE CUDA::cufft pybind11::module)


