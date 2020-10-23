#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 01:30:17 2020

@author: owlthekasra
"""
from scipy.fft import fft
from matplotlib.pyplot import plot 
import numpy as np
import pandas as pd

xrows = 2
df = pd.DataFrame()
df_FFT = pd.DataFrame()
# y = fft(df.iloc[x,:].to_numpy())

for x in range(0, 6):
    temp = pd.DataFrame(big_ass_training_set.iloc[:,1:].iloc[[x]])
    df = pd.concat([df, temp])
    temp2 = pd.DataFrame(fft(big_ass_training_set.iloc[:,1:].iloc[[x]]))
    df_FFT = pd.concat([df_FFT, temp2])

df = df.reset_index().iloc[:, 1:]
df_FFT = df_FFT.reset_index().iloc[:, 2:]

fourier = 256/15
newf = 0
for fa in range(1, 15):
    zed = fa*17 
    for item in fa:
    print(zed)
    newf = newf + small_frame_1.iloc[fa, item] 


y = fft(df.iloc[1,1:])
s=len(df_FFT.iloc[1])

x = np.linspace(2, 2000, len(df_FFT.iloc[1]))

plot(x,df_FFT.iloc[0])