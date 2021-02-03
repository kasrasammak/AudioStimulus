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
from CSP import CSP



ntrials = 68
pnts = 512
nbchan = 4
path = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/blink_experiment/blink_results'
nonblinkDf = np.array(pd.read_csv(path + '/nonBlinkTraining.csv').iloc[:,1:])
blinkDf = np.array(pd.read_csv(path + '/blinkTraining.csv').iloc[:,1:])
blinkTestDf = np.array(pd.read_csv(path + '/blinkTesting.csv').iloc[:,1:])
nonblinkTestDf = np.array(pd.read_csv(path + '/nonBlinkTesting.csv').iloc[:,1:])

def add_dimension(df, pnts, ntrials, nbchan):
    reshaped = np.reshape(df, (ntrials, pnts, nbchan))
    return reshaped

def subtract_dimension(df, nbchan):
    reshaped = df.transpose(2,0,1).reshape(nbchan, -1)
    return reshaped

def get_covmat(df, pnts, ntrials, nbchan):
    reshapedDf = np.reshape(df, (ntrials, pnts, nbchan))
    covmat = np.array(nbchan)
    for i in range(1,ntrials):
        trialDf = reshapedDf[i,:,:]
        means = np.mean(trialDf, axis = 0)
        j = trialDf - means
        covmat = covmat + np.dot(j.T, j)/len(j)
    covmat = covmat/ntrials
    return covmat

def get_eigs(covmat):
    evals, evecs = LA.eig(covmat)
    idx = evals.argsort()[::-1]
    eigVal = evals[idx]
    eigVec = evecs[:,idx]
    return (eigVal, eigVec)

def get_covmat_eigs(df, pnts, trials, nbchan):
    covmat = get_covmat(df, pnts, trials, nbchan)
    eigVal, eigVec = get_eigs(covmat)
    return (eigVal, eigVec)

def get_percent(eigVal):
    evals_percent = np.array(100*eigVal/sum(eigVal));
    return evals_percent

def get_thin_index(eigVal, threshhold=0):
    percent = get_percent(eigVal)
    ind = np.nonzero(percent>threshhold)[0]
    return (ind, percent)

def get_eig_and_comp(df, pnts, ntrials, nbchan, threshhold=0):
    reshapedDf = np.reshape(df, (ntrials, pnts, nbchan))
    eigVal, eigVec = get_covmat_eigs(df, ntrials, pnts, nbchan)
    ind, percent = get_thin_index(eigVal, threshhold)
    comp_vecs = np.dot(eigVec[:,ind].T, subtract_dimension(reshapedDf, nbchan))
    return (comp_vecs, (eigVal, eigVec), percent)

def reshape_to_list(df, pnts):
    nbchan = len(df)
    ntrials = np.int(np.shape(df)[1]/pnts)
    reshape = add_dimension(df.T, pnts, ntrials, nbchan)
    list1 = []
    for i in range(1,ntrials):
        list1.append(reshape[i,:,:].T)
    return list1

comp_eig_vecs_blink, eig_blink, blink_percent = get_eig_and_comp(blinkDf, 512, 68, 4, 5)
comp_eig_vecs_nonblink, eig_nonblink,nonblink_percent = get_eig_and_comp(nonblinkDf, 512, 69, 4, 7)
comp_eig_vecs_blink_test, eig_blink_test, blink_test_percent = get_eig_and_comp(blinkTestDf, 512, 68, 4, 5)
comp_eig_vecs_nonblink_test, eig_nonblink_test,nonblink_test_percent = get_eig_and_comp(nonblinkTestDf, 512, 68, 4, 10)

list1 = reshape_to_list(comp_eig_vecs_blink, 512)
list2 = reshape_to_list(comp_eig_vecs_nonblink, 512)
list3 = reshape_to_list(comp_eig_vecs_blink_test, 512)
list4 = reshape_to_list(comp_eig_vecs_nonblink_test, 512)

csps_training = CSP(list1, list2)
csps_testing = CSP(list3, list4)

comp_blink_train = np.dot(csps_training[0], comp_eig_vecs_blink)
comp_nonblink_train = np.dot(csps_training[1], comp_eig_vecs_nonblink)
comp_blink_test = np.dot(csps_testing[0], comp_eig_vecs_blink_test)
comp_nonblink_test = np.dot(csps_testing[1], comp_eig_vecs_nonblink_test)


# ---------EXTRA OPTIONS----------------------
# --------------------------------------------
# --------------------------------------------
# ---------EXTRA OPTIONS----------------------

# ERP covariance matrix
blinkCovMat = co.get_cov_mat_ERP(blinkDf, 68, 4)
blinkErp = co.get_ERP(blinkDf, 68, 4).T.reset_index().iloc[:,1:].T

blinkCovMatErp = blinkCovMat.to_numpy()
blinkCovMatErp = np.array(blinkCovMatErp, dtype=float)

# for erp
list1 = []
list2 =[]
list1.append(comp_eig_vecs_blink)
list2.append(comp_eig_vecs_nonblink)


# ------ some plotting 
plt.plot(comp_eig_vecs_blink.T)
plt.plot(blinkErp.iloc[:,3])
fig, axs = plt.subplots(4)
fig.suptitle('Vertically stacked subplots')
axs[0].plot(blinkErp.iloc[:,0])
axs[1].plot(blinkErp.iloc[:,1])
axs[2].plot(blinkErp.iloc[:,2])
axs[3].plot(blinkErp.iloc[:,3])

plt.scatter(blinkErp.iloc[:,0], blinkErp.iloc[:,1])