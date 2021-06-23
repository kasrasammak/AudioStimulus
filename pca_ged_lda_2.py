#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 02:20:13 2021

@author: owlthekasra
"""

import numpy as np
import pandas as pd
import pca
from CSP import CSP
import methods as md
import add_label as al

ntrials = 120
pnts = 512
nbchan = 4
directory = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus'

# =============================================================================
# read CSVs from thought experiment into an MxN dataframe,
# where M is datapoints of all trials in succession
# and N is number of channels
# =============================================================================

# =============================================================================
# prepare CSVs from 3 sounded experiment
# for this pipeline
# =============================================================================

# training data
big_train_no = np.reshape(big_train_3d[big_train_3d[:,:,0]==0], (4,int(len(big_train_3d[big_train_3d[:,:,0]==0])/4),519))
big_train_c1 = np.reshape(big_train_3d[big_train_3d[:,:,0]==1], (4,int(len(big_train_3d[big_train_3d[:,:,0]==1])/4),519))
big_train_d1 = np.reshape(big_train_3d[big_train_3d[:,:,0]==2], (4,int(len(big_train_3d[big_train_3d[:,:,0]==2])/4),519))
big_train_f1 = np.reshape(big_train_3d[big_train_3d[:,:,0]==3], (4,int(len(big_train_3d[big_train_3d[:,:,0]==3])/4),519))
big_train_imag_c1 = np.reshape(big_train_3d[big_train_3d[:,:,0]==4], (4,int(len(big_train_3d[big_train_3d[:,:,0]==4])/4),519))
big_train_imag_d1 = np.reshape(big_train_3d[big_train_3d[:,:,0]==5], (4,int(len(big_train_3d[big_train_3d[:,:,0]==5])/4),519))
big_train_imag_f1 = np.reshape(big_train_3d[big_train_3d[:,:,0]==6], (4,int(len(big_train_3d[big_train_3d[:,:,0]==6])/4),519))

# validation data 
big_valid_no = np.reshape(big_valid_3d[big_valid_3d[:,:,0]==0], (4,int(len(big_valid_3d[big_valid_3d[:,:,0]==0])/4),519))
big_valid_c1 = np.reshape(big_valid_3d[big_valid_3d[:,:,0]==1], (4,int(len(big_valid_3d[big_valid_3d[:,:,0]==1])/4),519))
big_valid_d1 = np.reshape(big_valid_3d[big_valid_3d[:,:,0]==2], (4,int(len(big_valid_3d[big_valid_3d[:,:,0]==2])/4),519))
big_valid_f1 = np.reshape(big_valid_3d[big_valid_3d[:,:,0]==3], (4,int(len(big_valid_3d[big_valid_3d[:,:,0]==3])/4),519))
big_valid_imag_c1 = np.reshape(big_valid_3d[big_valid_3d[:,:,0]==4], (4,int(len(big_valid_3d[big_valid_3d[:,:,0]==4])/4),519))
big_valid_imag_d1 = np.reshape(big_valid_3d[big_valid_3d[:,:,0]==5], (4,int(len(big_valid_3d[big_valid_3d[:,:,0]==5])/4),519))
big_valid_imag_f1 = np.reshape(big_valid_3d[big_valid_3d[:,:,0]==6], (4,int(len(big_valid_3d[big_valid_3d[:,:,0]==6])/4),519))


import pca 
c1_train = big_train_c1[:,:,1:].reshape(4, -1).T
d1_train = big_train_imag_c1[:,:,1:].reshape(4,-1).T
c1_valid = big_valid_c1[:,:,1:].reshape(4,-1).T
d1_valid = big_valid_imag_c1[:,:,1:].reshape(4,-1).T

# =============================================================================
# Common Spatial Patterns (CSP)
# Generalized Eigendecomposition from raw data
# =============================================================================

# compute covariance matrices
covA = np.cov(c1_train.T)
covE = np.cov(d1_train.T)

# compute GED and sort eigenvalues in descending order
E, V = np.linalg.eig(np.dot(np.linalg.inv(covA + covE), covA))
idx = E.argsort()[::-1]

comp_blink_train = np.dot(V[:, idx].T, c1_train.T)
comp_nonblink_train = np.dot(V[:, idx].T, d1_train.T)
comp_blink_test = np.dot(V[:, idx].T, c1_valid.T)
comp_nonblink_test = np.dot(V[:, idx].T, d1_valid.T)

# =============================================================================
# Principal Component Analysis into Common Spatial Patterns
# Perform PCA on raw data
# Perform CSP on component vectors
# =============================================================================

# # set threshold and compute PCA
# thresh1 = 0
# thresh2 = 0
# comp_eig_vecs_blink, eig_blink, blink_percent = pca.get_eig_and_comp(blinkDf, 512, 68, 4, thresh1)
# comp_eig_vecs_nonblink, eig_nonblink,nonblink_percent = pca.get_eig_and_comp(nonblinkDf, 512, 69, 4, thresh2)

# # Compute component vectors of validation data 
# # using the eigenvectors of training data
# comp_blink_test_pca = pca.get_comp(blinkTestDf.T,eig_blink[1], [0,1,2,3])
# comp_nonblink_test_pca = pca.get_comp(nonblinkTestDf.T, eig_nonblink[1], [0,1,2,3])

# # create list where each element of list represents an MxN trial
# # where M is the number of time points and N is number of channels
# list1 = pca.reshape_to_list(comp_eig_vecs_blink, 512)
# list2 = pca.reshape_to_list(comp_eig_vecs_nonblink, 512)

# # Compute Common Spatial Patterns of PCA component vectors
# # and compute new CSP component vectors
# csps_training = CSP(list1, list2)
# comp_blink_train = np.dot(csps_training[0], comp_eig_vecs_blink)
# comp_nonblink_train = np.dot(csps_training[0], comp_eig_vecs_nonblink)
# comp_blink_test = np.dot(csps_training[0], comp_blink_test_pca)
# comp_nonblink_test = np.dot(csps_training[0], comp_nonblink_test_pca)

# reshape array to dimension of leftover components
dim = 2
# for multiple sounds experiment
pnts = 518
f_blink_tr = np.reshape(comp_blink_train[:dim, :], (int(len(c1_train)/pnts), pnts, dim))
f_nonblink_tr = np.reshape(comp_nonblink_train[:dim, :], (int(len(d1_train)/pnts), pnts, dim))
f_blink_t = np.reshape(comp_blink_test[:dim, :], (int(len(c1_valid)/pnts), pnts, dim))
f_nonblink_t = np.reshape(comp_nonblink_test[:dim, :], (int(len(d1_valid)/pnts), pnts, dim))

# f_blink_tr = pca.add_dimension(comp_blink_train[:dim, :], pnts, ntrials, dim)
# f_nonblink_tr = pca.add_dimension(comp_nonblink_train[:dim, :], pnts, ntrials, dim)
# f_blink_t = pca.add_dimension(comp_blink_test[:dim, :], pnts, ntrials, dim)
# f_nonblink_t = pca.add_dimension(comp_nonblink_test[:dim, :], pnts, ntrials, dim)

# compute the log of the variance of all the data points in each trial
f_x_b_tr = np.log(np.var(f_blink_tr,axis = 1, ddof=1))
# f_x_nb_tr = np.log(np.var(f_nonblink_tr,axis = 1, ddof=1))[:ntrials,:]
f_x_nb_tr = np.log(np.var(f_nonblink_tr,axis = 1, ddof=1))
f_x_b_t = np.log(np.var(f_blink_t,axis = 1, ddof=1))
f_x_nb_t = np.log(np.var(f_nonblink_t,axis = 1, ddof=1))

# add a label to distinguish classes
f_x_b_tr = pca.add_label(f_x_b_tr, 1)
f_x_nb_tr = pca.add_label(f_x_nb_tr, 0)
f_x_b_t = pca.add_label(f_x_b_t, 1)
f_x_nb_t = pca.add_label(f_x_nb_t, 0)

# append different classes into one array, and separate labels into a separate array
# for both training set and validation set
x_tr = np.append(f_x_b_tr[:,:dim], f_x_nb_tr[:,:dim], axis = 0)
y_tr = np.append(f_x_b_tr[:,dim], f_x_nb_tr[:,dim])

x_t = np.append(f_x_b_t[:,:dim], f_x_nb_t[:,:dim], axis = 0)
y_t = np.append(f_x_b_t[:,dim], f_x_nb_t[:,dim])

# input train-test-split data for both training and validation set 
# into an LDA classifier
mod, acc, pred, y_test = md.fitPredictValSet(x_tr, y_tr, x_t, y_t, "lda")

# def var(sample):
#     return sum(np.square(sample - np.mean(sample)))/(len(sample)-1)



# Calculate ERPs
# comp_blink_train_erp = np.mean(add_dimension(comp_blink_train, pnts, ntrials, dim), axis = 0)
# comp_nonblink_train_erp = np.mean(add_dimension(comp_nonblink_train, pnts, 69, dim), axis = 0)
# comp_blink_test_erp = np.mean(add_dimension(comp_blink_train, pnts, ntrials, dim), axis = 0)
# comp_nonblink_test_erp = np.mean(add_dimension(comp_nonblink_train, pnts, ntrials, dim), axis = 0)
