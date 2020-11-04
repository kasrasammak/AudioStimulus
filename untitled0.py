#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:01:26 2020

@author: owlthekasra
"""

import numpy as np
import pandas as pd
import math

def get_covariance(arr1, arr2):
    n = len(arr1)
    cov = 1/(n-1)*sum((arr1-np.mean(arr1))*(arr2-np.mean(arr2)))
    return cov

def get_correlation(arr1, arr2):
    num = sum((arr1-np.mean(arr1))*(arr2-np.mean(arr2)))
    denom1 = math.sqrt(sum((arr1-np.mean(arr1))**2))
    denom2 = math.sqrt(sum((arr2-np.mean(arr2))**2))
    corr = num/(denom1*denom2)
    return corr

def get_ERP(df, trials, M):
    l1 = []
    l2 = pd.DataFrame()
    for i in range(M):
        l1.append(pd.DataFrame(np.array_split(np.array(df.iloc[:, i]), trials)))
        l2 = pd.concat([l2, np.mean(l1[i])], axis = 1)
    return l2

def get_covariance_matrix(df, M):
    covmat1 = pd.DataFrame(index = range(M), columns=range(M))
    for i in range(M):
        for j in range(M):
            subi = df.iloc[:, i] - np.mean(df.iloc[:, i])
            subj = df.iloc[:, j]-np.mean(df.iloc[:, j])
            covmat1.iloc[i, j] = sum(subi*subj)/(len(df)-1)
    return covmat1

def get_correlation_matrix(df, M):
    corrmat1 = pd.DataFrame(index = range(M), columns=range(M))
    for i in range(M):
        for j in range(M):
            subi = df.iloc[:, i] - np.mean(df.iloc[:, i])
            subj = df.iloc[:, j]-np.mean(df.iloc[:, j])
            corrmat1.iloc[i, j] = sum(subi*subj)/(math.sqrt(sum(subi**2))*math.sqrt(sum(subj**2)))
    return corrmat1

def get_cov_mat_inds(listofdfs, trials, M):
    covmat1 = pd.DataFrame(0, index = range(M), columns=range(M))
    for i in range(0, trials):
        covmat1 = covmat1 + get_covariance_matrix(listofdfs[i], M)
    covmat1 = covmat1/trials
    return covmat1

def get_corr_mat_inds(listofdfs, trials, M):
    corrmat = pd.DataFrame(0, index = range(M), columns=range(M))
    for i in range(0, trials):
        corrmat = corrmat + get_correlation_matrix(listofdfs[i], M)
    corrmat = corrmat/trials
    return corrmat

def get_cov_mat_ERP(df, trials, M):
    erp = get_ERP(df, trials, M)
    cov = get_covariance_matrix(erp, M)
    return cov

def get_corr_mat_ERP(df, trials, M):
    erp = get_ERP(df, trials, M)
    corr = get_correlation_matrix(erp, M)
    return corr

def get_mean_center(df):
    dfM = pd.DataFrame()
    for col in range(0,len(df.columns)):
        dfM = dfM.append(df.iloc[:,col] - np.mean(df.iloc[:,col]))
    return dfM

blinkCovMat1 = get_cov_mat_ERP(blinkDf, 68, 4)
blinkErp = get_ERP(blinkDf, 68, 4)
blinkErpM = get_mean_center(blinkErp)
blinkCovMat2 = blinkErpM.dot(blinkErpM.T) / (len(blinkErp) - 1)

yo3 = get_corr_mat_ERP(blinkDf, 68, 4)

feet_erp = get_ERP(feetDf, 68, 22)
feet_covERP = get_cov_mat_ERP(feetDf, 68, 22)
feet_corrERP = get_corr_mat_ERP(feetDf, 68, 22)
feet_corrinds = get_corr_mat_inds(feetlist, 68, 22)
feet_covinds = get_cov_mat_inds(feetlist, 68, 22)

blinkDf = pd.read_csv('/Users/k-owl/Documents/Code/DataScience/Python/AudioStimulus/blinkTesting.csv')
feetDf = pd.read_csv('/Users/k-owl/Documents/Code/DataScience/Python/AudioStimulus/feet-training.csv')



#feetlist = [feetDf[i:i + 750] for i in range(0, len(feetDf), 750)]


shortDf = blinkDf.iloc[:512, :]
mainsplit = np.array_split(blinkDf, 68)
feetsplit = np.array_split(feetDf, 47)
covmat2 = get_covariance_matrix_individuals(mainsplit, 68, 4)
corrmatind = get_correlation_matrix_individuals(mainsplit, 68, 4)
corrmat = get_correlation_matrix()
l3 = get_ERP(blinkDf, 68, 4)

arr11 = shortDf.iloc[:, 2]
arr22 = shortDf.iloc[:, 1]
covariance = get_covariance(arr11, arr22)
correlation = get_correlation(arr11, arr22)

dat = np.array([shortDf.iloc[:, 0],shortDf.iloc[:, 1],shortDf.iloc[:, 2], shortDf.iloc[:, 3]])
a = np.cov(dat)
M = 4;
#covmat1 = [[0 for x in range(M)] for y in range(M)] 
covmat1 = pd.DataFrame(index = range(M), columns=range(M))

yo = get_correlation_matrix(l2, 4)
yoyo = get_covariance_matrix(l2, 4)
yo2 = get_correlation_matrix(shortDf, 4)
yoyo2 = get_covariance_matrix(shortDf, 4)

data = get_covariance_matrix(shortDf, 0, 1)
    # columnData1 = []
    # for (columnName, columnData) in blinkDf.iteritems():
    #    columnData1.append(columnData.values)