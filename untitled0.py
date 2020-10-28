#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 23:01:26 2020

@author: owlthekasra
"""

import numpy as np
import pandas as pd
import math

blinkDf = pd.read_csv('/Users/owlthekasra/Documents/Code/Python/AudioStimulus/blinkTesting.csv')
shortDf = blinkDf.iloc[:512, :]

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
arr11 = shortDf.iloc[:, 2]
arr22 = shortDf.iloc[:, 1]
covariance = get_covariance(arr11, arr22)
correlation = get_correlation(arr11, arr22)

    return columnData1

np.mean(shortDf.iloc[:, 0])

data = get_covariance_matrix(shortDf, 0, 1)
    # columnData1 = []
    # for (columnName, columnData) in blinkDf.iteritems():
    #    columnData1.append(columnData.values)