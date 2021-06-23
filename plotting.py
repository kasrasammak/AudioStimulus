#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 01:25:07 2021

@author: owlthekasra
"""

%matplotlib auto

import matplotlib.pyplot as plt
import soundfile as sf
from sklearn.preprocessing import normalize


plt.plot(big_train_c1[2, 55, 1:])
data, samplerate = sf.read('sine_bass.wav')

# %%
brain_wave_no = big_train_no[1, 12, 1:]
norm_wave_no = brain_wave_no/np.linalg.norm(brain_wave_no)

brain_wave = big_train_c1[1, 15, 1:]
norm_wave = brain_wave/np.linalg.norm(brain_wave)

brain_wave_imag_c1 = big_train_imag_c1[1, 15, 1:]
norm_wave_imag_c1 = brain_wave_imag_c1/np.linalg.norm(brain_wave_imag_c1)

plt.clf()
plt.close()
fig, axs = plt.subplots(4, sharey='col')
fig.set_size_inches(28.5, 20.5)
fig.suptitle('sine wave and brain wave')
axs[0].set_title('sound')
axs[0].plot(data/8)
axs[1].set_title('listening')
axs[1].plot(norm_wave)
axs[2].set_title('imagining')
axs[2].plot(norm_wave_imag_c1)
axs[3].set_title('no sound')
axs[3].plot(norm_wave_no)
# fig.xlim([-1, 1])
