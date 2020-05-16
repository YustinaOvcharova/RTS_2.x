import matplotlib.pyplot as plt
import math
from random import *
import numpy as np

# Var 11
counts = 256
frequency = 1500
harmonics = 10

# Var 9
# counts = 256
# frequency = 2000
# harmonics = 8


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


def draw_fft():
    plt.title("ШВИДКЕ ПЕРЕТВОРЕННЯ ФУР'Є З ЧАСОМ")
    plt.plot(t, x_fft_real, 'b', t, x_fft_img, 'r')
    plt.show()


draw_dpf()
draw_fft()
