import numpy as np
import matplotlib.pyplot as plt
from build import stft_cuda


l = 1024
window_size = 256
hop = 128
N = 256
Fs = 1000
batch_size = 128

hanning_window = 0.5 * (1 - np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))).astype(np.float32)


def setup(l):
    s = np.random.rand(l).astype(np.float32)
    t = np.arange(l)/Fs
    c = np.sin(2*np.pi*200*t)
    x = c*s
    return x

def stft(x,w,h):
    L = len(x)
    N = len(w)
    M = 1+(L-N)//N
    X = np.zeros((N,M), dtype=np.complex64)
    for m in range(M):
        x_win = x[m*h:m*h+N] * w
        X_win = np.fft.fft(x_win)/Fs
        X[...,m] = X_win
    K = 1+N//2
    X = X[:K,...]
    return X


seg = 1 + (l - window_size) // window_size

x = setup(l)

out_shape = (N,seg)
out = np.empty(out_shape, dtype=np.complex64)
stft_cuda.stft(x,out,window_size,hop,N,Fs)

out = out[:1+N//2,...]

X = stft(x,hanning_window,hop)
X_cuda = out
print("STFT cuda output shape:", X_cuda.shape)

import perfplot 
def wrapper_cuda(y):
    out2 = np.empty((1+N//2,1+(len(y)-window_size)//window_size), dtype=np.complex64)
    stft_cuda.stft(y,out2,window_size,hop,N,Fs)
    return out2

def wrapper_python(y):
    return stft(y,hanning_window,hop)

perfplot.plot(
    setup=lambda l: setup(l),  # 準備數據
    kernels=[
        wrapper_cuda,
        wrapper_python,
    ],
    labels=['CUDA', 'Python'],
    n_range=[1000,10000,100000],  # 不同輸入大小
    xlabel='Signal Length',
    equality_check=None
)
plt.title("STFT Performance")
plt.show()

T = np.arange(X.shape[1]) * hop / Fs
F = np.arange(X.shape[0]) * Fs / N

left = min(T)
right = max(T) + N /Fs
up = max(F)
down = min(F)

"""
plt.imshow(np.abs(X),aspect="auto",origin="lower",cmap="viridis",extent=[left,right,down,up])
plt.xlabel("Time Frame")
plt.ylabel("Frequence Bin")
plt.colorbar(label="Magnitude")
plt.show()
"""

plt.figure(figsize=(12,5))

plt.subplot(121)
plt.imshow(np.abs(X),aspect="auto",origin="lower",cmap="viridis",extent=[left,right,down,up])
plt.xlabel("Time Frame")
plt.ylabel("Frequence Bin")
plt.title("STFT")
plt.colorbar(label="Magnitude")

plt.subplot(122)
plt.imshow(np.abs(X_cuda),aspect="auto",origin="lower",cmap="viridis",extent=[left,right,down,up])
plt.xlabel("Time Frame")
plt.ylabel("Frequence Bin")
plt.title("CUDA STFT")
plt.colorbar(label="Magnitude")

plt.tight_layout()
plt.show()

