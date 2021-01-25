#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 00:18:26 2020

@author: owlthekasra
"""

import pandas as pd
import numpy as np
from lfilter import butter_bandpass_filter
from scipy.signal import lfilter, butter
import matplotlib.pyplot as plt
from CSP import CSP
import covariance as co


path = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/blink_experiment/blink_results'
nonblinkDf = pd.read_csv(path + '/nonBlinkTraining.csv').iloc[:,1:]
blinkDf = pd.read_csv(path + '/blinkTraining.csv').iloc[:,1:]

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def filter_and_append(df, size, trials, fs=256, low=4, high=32):
    nyq = fs/2
    low = low/nyq
    high = high/nyq
    b, a = butter(5, [low, high], btype='band')
    filt_list = []
    for i in range(0, trials):
        temp = lfilter(b, a, df[i*size:(i+1)*size]).T
        filt_list.append(temp)
    return filt_list

blink_filtered_4_32 = filter_and_append(blinkDf, 512, 68)
nonBlink_filtered_4_32 = filter_and_append(nonblinkDf, 512, 68)


csps = CSP(blink_filtered_4_32, nonBlink_filtered_4_32)

ti4 = []
ti4.append(blinkDf[0:512].T)
ti5 = []
ti5.append(nonblinkDf[0:512].T)

# csps2 = CSP(ti4,ti5)

# from scipy import linalg as la

# vec, val = la.eig(csps[0])

# blinkDf1Trial = co.get_mean_center(blinkDf[0:512].T)
# nonBlinkDf1Trial = co.get_mean_center(nonblinkDf[0:512].T)
# covMatBlink = co.get_covariance_matrix(blinkDf1Trial, 4)
# covMatNonBlink = co.get_covariance_matrix(nonBlinkDf1Trial, 4)

# covmat1 = covMatBlink.to_numpy()
# covmat2 = covMatNonBlink.to_numpy()
# covmat1 = np.array(covmat1, dtype=float)
# covmat2 = np.array(covmat2, dtype=float)

# v,r = la.eig(covmat1, covmat1+covmat2)

import matplotlib.pyplot as plt

arr = np.hstack(blink_filtered_4_32)
arr2 = np.hstack(nonBlink_filtered_4_32)
spatialfilteredblink = np.dot(csps[0], arr)
spatialfilterednonblink = np.dot(csps[1], arr2)
arrR = arr[:, 512:1024]
arr2R = arr2[:, 512:1024]
s1 = spatialfilteredblink[:, 512:1024]
s2 = spatialfilterednonblink[:, 512:1024]
plt.scatter(arrR[1], arrR[2])
plt.scatter(s1[1], s1[2])
plt.scatter(arr2R[1], arr2R[2])
plt.scatter(s2[1], s2[2])
plt.ylim([-7, 5])


plt.scatter(s1[0],s2[1])
arr = np.hstack(transpose)
plt.plot(spatialfilteredblink)
spatialfilterednonblink = np.dot(csps[0], plt.plot(blink_filtered_4_32))

