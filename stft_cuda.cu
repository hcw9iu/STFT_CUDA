// stft_cufft_hanning.cu
#include <cuda_runtime.h>
#include <cufft.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>

namespace py = pybind11;

__global__ void process_segments(float* input, cuFloatComplex* output,
                               float* hanning_window, 
                               int signal_length, int window_size, 
                               int hop_size, int fft_size,
                               int num_segments) {
    int m = blockIdx.x;
    int tid = threadIdx.x;
    
    if (m < num_segments && tid < window_size) {
        int input_idx = m * hop_size + tid;
        if (input_idx < signal_length) {
            float sample = input[input_idx] * hanning_window[tid];
            output[m * fft_size + tid].x = sample;
            output[m * fft_size + tid].y = 0.0f;
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

__global__ void transpose(cuFloatComplex* d_output, cuFloatComplex* temp, int num_segments, int fft_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_segments && j < (fft_size/2 + 1)) {
        temp[j * num_segments + i] = d_output[i * (fft_size/2 + 1) + j];
    }
}

void performTranspose(cuFloatComplex* d_output, cuFloatComplex* temp, int num_segments, int fft_size) {
    dim3 blockSize(16, 16);
    dim3 gridSize((num_segments + blockSize.x - 1) / blockSize.x, (fft_size/2 + 1 + blockSize.y - 1) / blockSize.y);

    transpose<<<gridSize, blockSize>>>(d_output, temp, num_segments, fft_size);
    cudaDeviceSynchronize();
}

void stft_cufft_hanning(float* input, cuFloatComplex* output, int signal_length, int window_size, 
int hop_size, int fft_size, int Fs) {
    int num_segments = 1 + (signal_length - window_size) / window_size;
    
    // 分配設備內存
    float *d_input, *d_hanning_window;
    cuFloatComplex *d_output;
    
    cudaMalloc(&d_input, signal_length * sizeof(float));
    cudaMalloc(&d_output, num_segments * fft_size * sizeof(cuFloatComplex));
    cudaMalloc(&d_hanning_window, window_size * sizeof(float));
    
    // [!] 創建並初始化 hanning window
    float* h_hanning = (float*)malloc(window_size * sizeof(float));
    for(int i = 0; i < window_size; i++) {
        h_hanning[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (window_size - 1)));
    }
    cudaMemcpy(d_hanning_window, h_hanning, window_size * sizeof(float), 
               cudaMemcpyHostToDevice);
    free(h_hanning);
    
    // 複製輸入數據
    cudaMemcpy(d_input, input, signal_length * sizeof(float), 
               cudaMemcpyHostToDevice);
    
    // 設置 grid 和 block 維度
    dim3 block(256);
    dim3 grid((num_segments + block.x - 1) / block.x);
    
    // 啟動 kernel
    process_segments<<<grid, block>>>(d_input, d_output, d_hanning_window,
                                    signal_length, window_size, hop_size,
                                    fft_size, num_segments);
    
    // 設置並執行 FFT
    cufftHandle plan;
    cufftResult result = cufftPlan1d(&plan, fft_size, CUFFT_R2C, num_segments);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT error: Plan creation failed\n");
    }

    result = cufftExecR2C(plan, (cufftReal*)d_input, (cufftComplex*)d_output);
    if (result != CUFFT_SUCCESS) {
        printf("CUFFT error: ExecR2C failed\n");
    }
    
    // [!] FFT 後正規化
    int total_elements = num_segments * (fft_size/2 + 1);
    dim3 normalize_block(256);
    dim3 normalize_grid((total_elements + normalize_block.x - 1) / normalize_block.x);
    normalize_fft<<<normalize_grid, normalize_block>>>((cufftComplex*)d_output, total_elements, (float)Fs);
    
    // [!] 確保正規化完成
    cudaDeviceSynchronize();
    
    //size_t src_pitch = num_segments * sizeof(cuFloatComplex);
    //size_t dst_pitch = (fft_size/2 + 1) * sizeof(cuFloatComplex);
    //size_t width = sizeof(cuFloatComplex);  // 每次複製一個複數值
    //size_t height = num_segments;  // 行數，即時間段的數量

    //cudaMemcpy2D(output, dst_pitch, d_output, src_pitch, width, height, cudaMemcpyDeviceToHost);


    cuFloatComplex* temp;
    cudaMalloc(&temp, num_segments * (fft_size/2+1) * sizeof(cuFloatComplex));

    performTranspose(d_output, temp, num_segments, fft_size);

    cudaMemcpy(output, temp, num_segments * (fft_size/2+1) * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost);
    

    // [!] 釋放所有內存
    cufftDestroy(plan);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_hanning_window);
    
    // [!] 確保所有 CUDA 操作完成
    cudaDeviceSynchronize();

}

// pybind11 包裝函數
void stft_pybind(py::array_t<float> inputArray, py::array_t<std::complex<float>> outputArray, int window_size, int hop_size, int fft_size, int Fs) {
    auto inputBuf = inputArray.request();
    auto outputBuf = outputArray.request();

    float* input = static_cast<float*>(inputBuf.ptr);
    //cuFloatComplex* output = static_cast<cuFloatComplex*>(outputBuf.ptr);
    std::complex<float>* output = static_cast<std::complex<float>*>(outputBuf.ptr);

    int signal_length = inputBuf.shape[0];

    stft_cufft_hanning(input, reinterpret_cast<cufftComplex*>(output), signal_length, window_size, hop_size, fft_size, Fs);
}

PYBIND11_MODULE(stft_cuda, m) {
    m.def("stft", &stft_pybind, "Short Time Fourier Transform using cuFFT with Hanning window");
}
