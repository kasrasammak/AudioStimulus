#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 18:19:36 2020

@author: owlthekasra
"""

import pandas as pd
import numpy as np
from numpy import linalg as LA
from scipy import linalg as la
import matplotlib.pyplot as plt
import covariance as co

path = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/blink_experiment/blink_results'
blinkDf = pd.read_csv(path + '/nonBlinkTraining.csv')
blinkDf = blinkDf.iloc[:,1:]

# easier way to get covariance matrix
blinkCovMat1 = co.get_cov_mat_ERP(blinkDf, 68, 4)
blinkErp = co.get_ERP(blinkDf, 68, 4).T.reset_index().iloc[:,1:].T

covMat = blinkCovMat1.to_numpy()
covMat = np.array(covMat, dtype=float)

evals, evecs = LA.eig(covMat)

biggger = np.dot(covMat, evecs)

idx = evals.argsort()[::-1]   
eigenValues = evals[idx]
eigenVectors = evecs[:,idx]

comp_eig_vecs = np.dot(eigenVectors.T, blinkErp)

comp1 = eigenVectors.T

plt.plot(comp_eig_vecs.T)
plt.plot(blinkErp.iloc[:,3])
fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(blinkErp.iloc[:,0])
axs[1].plot(blinkErp.iloc[:,1])
axs[2].plot(blinkErp.iloc[:,2])
axs[3].plot(blinkErp.iloc[:,3])

plt.scatter(blinkErp.iloc[:,0], blinkErp.iloc[:,1])

array = np.array([[3,1],[1,2]], dtype=float)
e, v = LA.eig(array)