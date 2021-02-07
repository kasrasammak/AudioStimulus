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

ntrials = 68
pnts = 512
nbchan = 4
directory = '/Users/owlthekasra/Documents/Code/Python'
folder = '/AudioStimulus/blink_experiment/blink_results'
path = directory + folder
nonblinkDf = np.array(pd.read_csv(path + '/nonBlinkTraining.csv').iloc[:,1:])
blinkDf = np.array(pd.read_csv(path + '/blinkTraining.csv').iloc[:,1:])
blinkTestDf = np.array(pd.read_csv(path + '/blinkTesting.csv').iloc[:,1:])
nonblinkTestDf = np.array(pd.read_csv(path + '/nonBlinkTesting.csv').iloc[:,1:])

## Quick CSP
covA = np.cov(blinkDf.T)
covE = np.cov(nonblinkDf.T)

E, V = np.linalg.eig(np.dot(np.linalg.inv(covA + covE), covA))

comp_blink_train = np.dot(V.T, blinkDf.T)
comp_nonblink_train = np.dot(V.T, nonblinkDf.T)
comp_blink_test = np.dot(V.T, blinkTestDf.T)
comp_nonblink_test = np.dot(V.T, nonblinkTestDf.T)

## perform PCA and then CSP
# thresh1 = 0
# thresh2 = 0
# comp_eig_vecs_blink, eig_blink, blink_percent = pca.get_eig_and_comp(blinkDf, 512, 68, 4, thresh1)
# comp_eig_vecs_nonblink, eig_nonblink,nonblink_percent = pca.get_eig_and_comp(nonblinkDf, 512, 69, 4, thresh2)

# comp_blink_test_pca = pca.get_comp(blinkTestDf.T,eig_blink[1], [0,1,2,3])
# comp_nonblink_test_pca = pca.get_comp(nonblinkTestDf.T, eig_nonblink[1], [0,1,2,3])

# list1 = pca.reshape_to_list(comp_eig_vecs_blink, 512)
# list2 = pca.reshape_to_list(comp_eig_vecs_nonblink, 512)

# csps_training = CSP(list1, list2)

# comp_blink_train = np.dot(csps_training[0], comp_eig_vecs_blink)
# comp_nonblink_train = np.dot(csps_training[0], comp_eig_vecs_nonblink)

# comp_blink_test = np.dot(csps_training[0], comp_blink_test_pca)
# comp_nonblink_test = np.dot(csps_training[0], comp_nonblink_test_pca)

dim = 1

f_blink_tr = pca.add_dimension(comp_blink_train[:dim, :], pnts, 68, dim)
f_nonblink_tr = pca.add_dimension(comp_nonblink_train[:dim, :], pnts, 69, dim)
f_blink_t = pca.add_dimension(comp_blink_test[:dim, :], pnts, 68, dim)
f_nonblink_t = pca.add_dimension(comp_nonblink_test[:dim, :], pnts, 68, dim)

f_x_b_tr = np.log(np.var(f_blink_tr,axis = 1, ddof=1))
f_x_nb_tr = np.log(np.var(f_nonblink_tr,axis = 1, ddof=1))[:68,:]
f_x_b_t = np.log(np.var(f_blink_t,axis = 1, ddof=1))
f_x_nb_t = np.log(np.var(f_nonblink_t,axis = 1, ddof=1))


f_x_b_tr = pca.add_label(f_x_b_tr, 1)
f_x_nb_tr = pca.add_label(f_x_nb_tr, 0)
f_x_b_t = pca.add_label(f_x_b_t, 1)
f_x_nb_t = pca.add_label(f_x_nb_t, 0)

x_tr = np.append(f_x_b_tr[:,:dim], f_x_nb_tr[:,:dim], axis = 0)
y_tr = np.append(f_x_b_tr[:,dim], f_x_nb_tr[:,dim])

x_t = np.append(f_x_b_t[:,:dim], f_x_nb_t[:,:dim], axis = 0)
y_t = np.append(f_x_b_t[:,dim], f_x_nb_t[:,dim])



mod, acc, pred, y_test = md.fitPredictValSet(x_tr, y_tr, x_t, y_t, "lda")

# def var(sample):
#     return sum(np.square(sample - np.mean(sample)))/(len(sample)-1)



# Calculate ERPs
# comp_blink_train_erp = np.mean(add_dimension(comp_blink_train, pnts, ntrials, dim), axis = 0)
# comp_nonblink_train_erp = np.mean(add_dimension(comp_nonblink_train, pnts, 69, dim), axis = 0)
# comp_blink_test_erp = np.mean(add_dimension(comp_blink_train, pnts, ntrials, dim), axis = 0)
# comp_nonblink_test_erp = np.mean(add_dimension(comp_nonblink_train, pnts, ntrials, dim), axis = 0)


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