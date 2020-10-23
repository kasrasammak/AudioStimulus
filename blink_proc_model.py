#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:39:51 2020

@author: owlthekasra
"""

import pandas as pd
import methods as md
def t_label(df, label):
    df = df.reset_index().iloc[:, 1:].T
    df['label'] = label
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    return df

def getDF(df_list, label):
    bigdf = pd.DataFrame()
    for frame in df_list: 
        print(t_label(frame, label))
        bigdf = bigdf.append(t_label(frame, label))
    return bigdf.reset_index().iloc[:, 1:]

blinkTrainDF = getDF(blinkTrain, 1)
blinkTestDF = getDF(blinkTest, 1)
nonBlinkTrainDF = getDF(nonBlinkTrain, 0)
nonBlinkTestDF = getDF(nonBlinkTest, 0)

training = blinkTrainDF.append(nonBlinkTrainDF)
bg1 = training.iloc[::4, :]
bg2 = training.iloc[1::4, :]
bg3 = training.iloc[2::4, :]
bg4 = training.iloc[3::4, :]
main = bg1.sample(frac=1)
main2 = bg2.sample(frac=1)
main3 = bg3.sample(frac=1)
main4 = bg4.sample(frac=1)
val = blinkTestDF.append(nonBlinkTestDF)
bg1val = training.iloc[::4, :]
bg2val = training.iloc[1::4, :]
bg3val = training.iloc[2::4, :]
bg4val = training.iloc[3::4, :]
mainval = bg1val.sample(frac=1)
main2val = bg2val.sample(frac=1)
main3val = bg3val.sample(frac=1)
main4val = bg4val.sample(frac=1)
X_train = main3.iloc[:, 1:]
y_train = main3['label']
X_val = main3val.iloc[:, 1:]
y_val = main3val['label']

model, acc, pred, y_test = md.fitPredictValSet(X_train, y_train, X_val, y_val, 'svm')