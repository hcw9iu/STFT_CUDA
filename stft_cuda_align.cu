// stft_cufft_hanning.cu
#include <cuda_runtime.h>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>

#define SHARED_SIZE 1024

namespace py = pybind11;

__global__ void process_segments(float* input, float* output,
                               float* hanning_w, 
                               int sig_len, int win_size, 
                               int hop_size,
                               int segs) {
    __shared__ float hanning_w_shared[SHARED_SIZE];
    int m = blockIdx.x;
    int tid = threadIdx.x;

    if (tid < win_size) {
        hanning_w_shared[tid] = hanning_w[tid];
    }
    __syncthreads();
    
    if (m < segs && tid < win_size) {
        int input_idx = m * hop_size + tid;
        if (input_idx < sig_len) {
            float sample = __ldg(&input[input_idx]) * hanning_w_shared[tid];
            output[m * win_size + tid] = sample;
        }
    }
}

__global__ void normalize_fft(cufftComplex* data, int size, float Fs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float norm = Fs;
        data[idx].x /= norm;
        data[idx].y /= norm;
    }
}

__global__ void transpose(cuFloatComplex* d_output, cuFloatComplex* d_output_T, int segs, int fft_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < segs && j < (fft_size/2 + 1)) {
        d_output_T[j * segs + i] = d_output[i * (fft_size/2 + 1) + j];
    }
}

void performTranspose(cuFloatComplex* d_segs, cuFloatComplex* d_output_T, int segs, int fft_size) {
    dim3 blockSize(16, 16);
    dim3 gridSize((segs + blockSize.x - 1) / blockSize.x, (fft_size/2 + 1 + blockSize.y - 1) / blockSize.y);

    transpose<<<gridSize, blockSize>>>(d_segs, d_output_T, segs, fft_size);
    cudaDeviceSynchronize();
}

void stft_cufft_hanning(float* input, cuFloatComplex* output, int sig_len, int win_size, 
int hop_size, int fft_size, int Fs) {
    int segs = 1 + (sig_len - win_size) / hop_size;
    
    // device memory
    float *d_input, *d_hanning_win, *d_output_B;
    cufftComplex *d_output, *d_output_T;
    
    // set output size
    size_t output_samples = segs * (fft_size/2+1) * sizeof(cufftComplex);  
    size_t samples = segs * fft_size * sizeof(cufftComplex); 
    size_t samples_B = segs * win_size * sizeof(float); 

    // allocate device memory
    cudaMalloc(&d_input, sig_len * sizeof(float));  
    cudaMalloc(&d_output_B, samples_B);  // input segment handler buffer
    cudaMalloc(&d_output, samples);  // FFT output buffer
    cudaMalloc(&d_output_T, output_samples);  // transposed final output buffer
    cudaMalloc(&d_hanning_win, win_size * sizeof(float));

    cudaError_t cudaErr = cudaGetLastError();
    if (cudaErr != cudaSuccess) {
        printf("CUDA error before FFT plan: %s\n", cudaGetErrorString(cudaErr));
        return;
    }
    
    // set init value = 0
    cudaMemset(d_input, 0, sig_len * sizeof(float));
    cudaMemset(d_output, 0, samples);
    cudaMemset(d_output_B, 0, samples_B);
    //cudaMemset(d_segs, 0, fft_buffer_size);
    
    // create and init hanning window
    float* h_hanning = (float*)malloc(win_size * sizeof(float));
    for(int i = 0; i < win_size; i++) {
        h_hanning[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (win_size - 1)));
    }
    cudaMemcpy(d_hanning_win, h_hanning, win_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    free(h_hanning);
    
    // copy input data
    cudaMemcpy(d_input, input, sig_len * sizeof(float), cudaMemcpyHostToDevice);
    
    // set grid and block dimension
    dim3 block(win_size);
    dim3 grid(segs);
    
    // launch kernel
    process_segments<<<grid, block>>>(d_input, d_output_B, d_hanning_win,
                                    sig_len, win_size, hop_size, segs);
    
    // set and execute FFT
    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, fft_size, CUFFT_R2C, segs);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT error: Plan creation failed\n");
    }

    result = cufftExecR2C(plan, d_output_B, d_output);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT error: ExecR2C failed\n");
    }
    
    // FFT normalization
    int total_elements = segs * (fft_size/2 + 1);
    dim3 normalize_block(256);
    dim3 normalize_grid((total_elements + normalize_block.x - 1) / normalize_block.x);
    normalize_fft<<<normalize_grid, normalize_block>>>(d_output, total_elements, (float)Fs);
    
    // ensure normalization completed
    cudaDeviceSynchronize();

    performTranspose(d_output, d_output_T, segs, fft_size);

    cudaMemcpy(output, d_output_T, segs * (fft_size/2+1) * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
    

    // release memory
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_output_B);
    cudaFree(d_hanning_win);
    cudaFree(d_output_T);
    
    // ensure all CUDA operations completed
    cudaDeviceSynchronize();

}

// pybind11 wrapper
void stft_pybind(py::array_t<float> inputArray, py::array_t<std::complex<float>> outputArray, int win_size, int hop_size, int fft_size, int Fs) {
    auto inputBuf = inputArray.request();
    auto outputBuf = outputArray.request();

    float* input = static_cast<float*>(inputBuf.ptr);
    std::complex<float>* output = static_cast<std::complex<float>*>(outputBuf.ptr);

    int sig_len = inputBuf.shape[0];

    stft_cufft_hanning(input, reinterpret_cast<cufftComplex*>(output), sig_len, win_size, hop_size, fft_size, Fs);
}

PYBIND11_MODULE(stft_cuda_align, m) {
    m.def("stft", &stft_pybind, "Short Time Fourier Transform using cuFFT with Hanning window");
}