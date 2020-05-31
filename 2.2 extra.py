import matplotlib.pyplot as plt
import math
from random import *
import numpy as np
from multiprocessing import Pool

# Var 11
counts = 256
frequency = 1500
harmonics = 10



def freg_gen(frequency, harmonics):
    Freg = []
    step = frequency / harmonics
    for i in range(harmonics):
        Freg.append(frequency - step * i)
    return Freg


def expectation(x, counts):
    mx = 0.0
    for i in range(counts):
        mx += x[i]
    return mx / counts


def dispersion(x, counts, mx):
    dx = 0.0
    for i in range(counts):
        dx += math.pow(x[i] - mx, 2)
    return dx / (counts - 1)


def xt_gen(counts, harmonics, Arraray_new, Freg, alpha):
    x = [0] * counts
    for j in range(counts):
        for i in range(harmonics):
            x[j] += Arraray_new[i] * math.sin(Freg[i] * j + alpha[i])
    return x


def arr_gen(harmonics, min, max):
    arr = [0] * harmonics
    for i in range(harmonics):
        arr[i] = randint(min, max)
    return arr


def dpf(signal):
    harmonics = len(signal)
    p = np.arange(harmonics)
    k = p.reshape((harmonics, 1))
    frequency = np.exp(-2j * np.pi * p * k / harmonics)
    return np.dot(frequency, signal)


def fft(signal):
    signal = np.asanyarray(signal, dtype=float)
    counts = len(signal)
    if counts <= 2:
        return dpf(signal)
    else:
        signal_even = fft(signal[::2])
        signal_odd = fft(signal[1::2])
        terms = np.exp(-2j * np.pi * np.arange(counts) / counts)
        return np.concatenate([signal_even + terms[:counts // 2] * signal_odd,
                               signal_even + terms[counts // 2:] * signal_odd])

def fft_parallel(signal):
    p = Pool(2)
    signal = np.asanyarray(signal, dtype=float)
    counts = len(signal)
    if counts <= 2:
        return dpf(signal)
    else:
        signal_even_odd = (signal[::2], signal[1::2])
        signal_even, signal_odd = p.map(fft, signal_even_odd)
        terms = np.exp(-2j * np.pi * np.arange(counts) / counts)
        return np.concatenate([signal_even + terms[:counts // 2] * signal_odd,
                               signal_even + terms[counts // 2:] * signal_odd])

# 2.1
Arraray_new = arr_gen(harmonics, 0, 5)
alpha = arr_gen(harmonics, 0, 5)
Freg = freg_gen(frequency, harmonics)
x = xt_gen(counts, harmonics, Arraray_new, Freg, alpha)
x_dpf = dpf(x)
x_fft = fft(x)

# 2.2
t = np.linspace(0, 10, counts)
x_dpf_real = x_dpf.real
x_dpf_img = x_dpf.imag
x_fft_real = x_fft.real
x_fft_img = x_fft.imag


def draw_dpf():
    plt.title("ДИСКРЕТНОГО ПЕРЕТВОРЕННЯ ФУР'Є")
    plt.plot(t, x_dpf_real, 'b', t, x_dpf_img, 'r')
    plt.show()


def draw_fft(x_fft_real, x_fft_img):
    plt.title("ШВИДКЕ ПЕРЕТВОРЕННЯ ФУР'Є З ЧАСОМ")
    plt.plot(t, x_fft_real, 'b', t, x_fft_img, 'r')
    plt.show()

x_fft = fft_parallel(x)
x_fft_real2 = x_fft.real
x_fft_img2 = x_fft.imag

draw_fft(x_fft_real2, x_fft_img2)