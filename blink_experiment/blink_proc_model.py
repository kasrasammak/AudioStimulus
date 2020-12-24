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
        bigdf = bigdf.append(t_label(frame, label))
    return bigdf.reset_index().iloc[:, 1:]

blinkTrainDF = getDF(blinkTrain, 1)
blinkTestDF = getDF(blinkTest, 1)
nonBlinkTrainDF = getDF(nonBlinkTrain, 0)
nonBlinkTestDF = getDF(nonBlinkTest, 0)

training = blinkTrainDF.append(nonBlinkTrainDF)

trainingy = pd.DataFrame(training['label'])
newtraining = pd.DataFrame()
for i in range(len(training)):
    mean_cent_row = training.iloc[i,1:] - np.mean(training.iloc[i,1:])
    newtraining = newtraining.append(mean_cent_row)
newtraining = pd.concat([trainingy, newtraining], axis=1)

bg1 = newtraining.iloc[::4, :]
bg2 = newtraining.iloc[1::4, :]
bg3 = newtraining.iloc[2::4, :]
bg4 = newtraining.iloc[3::4, :]

val = blinkTestDF.append(nonBlinkTestDF)

valy = pd.DataFrame(val['label'])
newval = pd.DataFrame()
for i in range(len(val)):
    mean_cent_row = val.iloc[i,1:] - np.mean(val.iloc[i,1:])
    newval = newval.append(mean_cent_row)
newval = pd.concat([valy, newval], axis=1)

bg1val = newval.iloc[::4, :]
bg2val = newval.iloc[1::4, :]
bg3val = newval.iloc[2::4, :]
bg4val = newval.iloc[3::4, :]

main = bg1.sample(frac=1)
main2 = bg2.sample(frac=1)
main3 = bg3.sample(frac=1)
main4 = bg4.sample(frac=1)
mainval = bg1val.sample(frac=1)
main2val = bg2val.sample(frac=1)
main3val = bg3val.sample(frac=1)
main4val = bg4val.sample(frac=1)
X_train = main4.iloc[:, 1:]
y_train = main4['label']
X_val = main4val.iloc[:, 1:]
y_val = main4val['label']

model, acc, pred, y_test = md.fitPredictValSet(X_train, y_train, X_val, y_val, 'forest')


BUFFER_LENGTH = 512
EPOCH_LENGTH = 256 
OVERLAP_LENGTH = 256-64
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH 
n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) / SHIFT_LENGTH + 1))
a1train = bg1.reset_index().iloc[:,2:]
a1val = bg1val.reset_index().iloc[:,2:]
a1block = pd.DataFrame()
for j in range(0, len(a1train)):
    for i in range(0, n_win_test):
        temp = a1train.iloc[j, i:i+EPOCH_LENGTH].T.reset_index().iloc[:,1:].T
        if j < 68:
            temp['label'] = 1
        else:
            temp['label'] =  0
        a1block = a1block.append(temp)
a2block = pd.DataFrame()
for j in range(0, len(a1val)):
    for i in range(0, n_win_test):
        temp = a1val.iloc[j, i:i+EPOCH_LENGTH].T.reset_index().iloc[:,1:].T
        if j < 68:
            temp['label'] = 1
        else:
            temp['label'] =  0
        a2block = a2block.append(temp)
cols = list(a1block.columns)
cols = [cols[-1]] + cols[:-1]
a1block = a1block[cols].reset_index().iloc[:,1:]
a2block = a2block[cols].reset_index().iloc[:,1:]
a1main = a1block.sample(frac=1)
a2main = a2block.sample(frac=1)
X_train = a1main.iloc[:, 1:]
y_train = a1main['label']
X_val = a2main.iloc[:, 1:]
y_val = a2main['label']

model, acc, pred, y_test = md.fitPredictValSet(X_train, y_train, X_val, y_val, 'knn')
