#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 21:45:15 2020

@author: owlthekasra
"""
fs = 256;
data_epoch = []

import numpy as np
import matplotlib.pyplot as plt

def nextpow2(i):
    """
    Find the next power of 2 for number i
    """
    n = 1
    while n < i:
        n *= 2
    return n
# def compute_band_powers(data_epoch, fs):
"""Extract the features (band powers) from the EEG.

Args:
    data_epoch (numpy.ndarray): array of dimension [number of samples,
            number of channels]
    fs (float): sampling frequency of data_epoch

Returns:
    (numpy.ndarray): feature matrix of shape [number of feature points,
        number of different features]
"""
# 1. Compute the PSD
winSampleLength, nbCh = data_epoch_2.shape

# Apply Hamming window
w = np.hamming(winSampleLength)
dataWinCentered = data_epoch - np.mean(data_epoch, axis=0)  # Remove offset
dataWinCenteredHam = (dataWinCentered.T * w).T

NFFT = nextpow2(winSampleLength)
Y = np.fft.fft(data_epoch_2, n=512, axis=0) / 512
PSD = 2 * np.abs(Y[0:int(512 / 2), :])
f = fs / 2 * np.linspace(0, 1, int(NFFT / 2))
Y*winSampleLength
# SPECTRAL FEATURES
# Average of band powers
# Delta <4
ind_delta, = np.where(f < 4)
meanDelta = np.mean(PSD[ind_delta, :], axis=0)
# Theta 4-8
ind_theta, = np.where((f >= 4) & (f <= 8))
meanTheta = np.mean(PSD[ind_theta, :], axis=0)
# Alpha 8-12
ind_alpha, = np.where((f >= 8) & (f <= 12))
meanAlpha = np.mean(PSD[ind_alpha, :], axis=0)
# Beta 12-30
ind_beta, = np.where((f >= 12) & (f < 30))
meanBeta = np.mean(PSD[ind_beta, :], axis=0)

feature_vector = np.concatenate((meanDelta, meanTheta, meanAlpha,
                                 meanBeta), axis=0)

feature_vector = np.log10(feature_vector)

plt.plot(Y[1:512//2, :])
data_epoch_2= blinkTrain[1].iloc[:, 1].reset_index().iloc[:, 1:]

np.linspace(0, 127, 128)

import audiofile as af

signal, sampling_rate = af.read('gigpip.wav')

plt.plot(signal[0])

from scipy.fft import fft, ifft
# Number of sample points0
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, 1, 256)
y = np.sin(2.0*np.pi*x)
yf = fft(y)
yff = np.abs(yf[0:256//2])
plt.plot(yff)
xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
import matplotlib.pyplot as plt
plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
plt.grid()
plt.show()

linspace = np.linspace(0, 1.0/800.0*600, 600)


from scipy import signal

fc = 0.01
b = 0.08
N = int(np.ceil((4 / b)))
if not N % 2: N += 1
n = np.arange(N)

sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1)) + 0.08 * np.cos(4 * np.pi * n / (N - 1))
sinc_func = sinc_func * window
sinc_func = sinc_func / np.sum(sinc_func)
s = np.array(data_epoch_2).flatten()
new_signal = np.convolve(s, sinc_func)
# plt.plot(s)
plt.plot(new_signal)


