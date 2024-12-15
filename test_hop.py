import numpy as np
import matplotlib.pyplot as plt
from build import stft_cuda, stft_cudaM1, stft_cuda_align
import perfplot

# 基本參數設置
l = 1000000  # 固定信號長度
window_size = 256
N = 256
Fs = 1000

def setup(hop_size):
    # 生成測試信號
    s = np.random.rand(l).astype(np.float32)
    t = np.arange(l)/Fs
    c = np.sin(2*np.pi*200*t)
    x = c*s
    return x

def stft(x, w, h):
    L = len(x)
    N = len(w)
    M = 1+(L-N)//h
    X = np.zeros((N,M), dtype=np.complex64)
    for m in range(M):
        x_win = x[m*h:m*h+N] * w
        X_win = np.fft.fft(x_win)/Fs
        X[...,m] = X_win
    K = 1+N//2
    X = X[:K,...]
    return X

# 包裝函數，接受 hop_size 作為參數
def wrapper_cuda(hop_size):
    x = setup(hop_size)
    out = np.empty((1+N//2,1+(l-window_size)//hop_size), dtype=np.complex64)
    stft_cuda.stft(x, out, window_size, hop_size, N, Fs)
    return out

def wrapper_cudaM1(hop_size):
    x = setup(hop_size)
    out = np.empty((1+N//2,1+(l-window_size)//hop_size), dtype=np.complex64)
    stft_cudaM1.stft(x, out, window_size, hop_size, N, Fs)
    return out

def wrapper_cuda_align(hop_size):
    x = setup(hop_size)
    out = np.empty((1+N//2,1+(l-window_size)//hop_size), dtype=np.complex64)
    stft_cuda_align.stft(x, out, window_size, hop_size, N, Fs)
    return out

def wrapper_python(hop_size):
    x = setup(hop_size)
    hanning_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))).astype(np.float32)
    return stft(x, hanning_window, hop_size)

# 執行性能測試
out = perfplot.bench(
    setup=lambda n: n,  # 直接傳遞 hop_size
    kernels=[
        wrapper_cuda,
        wrapper_cudaM1,
        wrapper_cuda_align,
        wrapper_python,
    ],
    labels=['CUDA', 'CUDA Async', 'CUDA Align', "Python"],
    n_range=[16, 32, 64, 128, 256],  # 不同的 hop size
    xlabel='Hop Size',
    equality_check=None
)

# 打印詳細時間
print("\nDetailed timing (seconds):")
print("Hop Size     |", end=" ")
for label in out.labels:
    print(f"{label:12}", end=" | ")
print()

for i, n in enumerate(out.n_range):
    print(f"{n:12} |", end=" ")
    for t in out.timings_s[:, i]:
        print(f"{t:12.6f}", end=" | ")
    print()

# 繪製性能圖
plt.figure(figsize=(10, 6))
out.plot()
plt.title("STFT Performance with Different Hop Sizes")
plt.ylabel("Time [s]")
plt.xscale("log", base=2)
plt.grid(True)
plt.xticks(out.n_range, out.n_range)
plt.show()
