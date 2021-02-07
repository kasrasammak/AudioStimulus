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
import methods as md
import statistics as stat
import methods as md


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

def get_comp(df, V, ind):
    comp_vecs = np.dot(V[:, ind].T, df)
    return comp_vecs

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
def add_label(df, label):
    dfL = np.empty([len(df), 1])
    dfL[:,:] = label
    df = np.append(df, dfL, axis=1)
    return df

