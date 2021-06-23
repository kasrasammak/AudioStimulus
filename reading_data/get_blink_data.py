#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 02:06:28 2020

@author: owlthekasra
"""

import pandas as pd
import glob
import numpy as np

path = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/blink_experiment/'
    
# find nearest timestamp and get indices for before and after that timestamp
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

def getIndices(df, val):
    Timestamp = df.iloc[:,0].to_list()
    # TimeStampVal = lisst
    valInd = find_nearest(Timestamp, val)
    indMiddle = int(df[df['Timestamp']==valInd].index[0])
    indStart = indMiddle - 256
    indStop = indMiddle + 256
    return (indStart, indStop)

# get csvs into list of dataframes with eeg data for each set
blinkTrainingList = glob.glob(path + 'blinkTraining/*.csv')
blinkTestingList = glob.glob(path + 'blinkTesting/*.csv')
nonBlinkTrainingList = glob.glob(path + 'nonBlinkTraining/*.csv')
nonBlinkTestingList = glob.glob(path + 'nonBlinkTesting/*.csv')
blinkTrainList = []
blinkTestList = []
nonBlinkTrainList = []
nonBlinkTestList = []
for member in blinkTrainingList:
    blinkTrainList.append(pd.read_csv(member).iloc[:,1:])
for member in blinkTestingList:
    blinkTestList.append(pd.read_csv(member).iloc[:,1:])
for member in nonBlinkTrainingList:
    nonBlinkTrainList.append(pd.read_csv(member).iloc[:,1:])
for member in nonBlinkTestingList:
    nonBlinkTestList.append(pd.read_csv(member).iloc[:,1:])
    
# get csvs into lists of information about the timestamps where the blinks did and didn't happen
blinkTrainingData = glob.glob(path + 'blinkTraining/blinkTrainingData/*.csv')
blinkTestingData = glob.glob(path + 'blinkTesting/blinkTestingData/*.csv')
nonBlinkTrainingData = glob.glob(path + 'nonBlinkTraining/nonBlinkTrainingData/*.csv')
nonBlinkTestingData = glob.glob(path + 'nonBlinkTesting/nonBlinkTestingData/*.csv')
blinkTrainData = []
blinkTestData = []
nonBlinkTrainData = []
nonBlinkTestData = []
for member in blinkTrainingData:
    blinkTrainData.append(pd.read_csv(member).iloc[0,1])
for member in blinkTestingData:
    blinkTestData.append(pd.read_csv(member).iloc[0,1])
for member in nonBlinkTrainingData:
    nonBlinkTrainData.append(pd.read_csv(member).iloc[0,1])
for member in nonBlinkTestingData:
    nonBlinkTestData.append(pd.read_csv(member).iloc[0,1])
    


# find correct indices in order to cut the arrays around the area where the blink 
    # did or did not happen, cut the dataframes so they are all the same size,
# and then stack them upon one another into a larger dataframe
blinkTrain = []
blinkTrainDf = pd.DataFrame()
for indx, timestamp in enumerate(blinkTrainData):
    indStart, indStop = getIndices(blinkTrainList[indx], timestamp)
    blinkTrain.append(blinkTrainList[indx].iloc[indStart:indStop, 1:])
    blinkTrainDf = blinkTrainDf.append(blinkTrainList[indx].iloc[indStart:indStop, 1:])
blinkTest = []
blinkTestDf = pd.DataFrame()
for indx, timestamp in enumerate(blinkTestData):
    indStart, indStop = getIndices(blinkTestList[indx], timestamp)
    blinkTest.append(blinkTestList[indx].iloc[indStart:indStop, 1:])
    blinkTestDf = blinkTestDf.append(blinkTestList[indx].iloc[indStart:indStop, 1:])
nonBlinkTrain = []
nonBlinkTrainDf = pd.DataFrame()
for indx, timestamp in enumerate(nonBlinkTrainData):
    indStart, indStop = getIndices(nonBlinkTrainList[indx], timestamp)
    nonBlinkTrain.append(nonBlinkTrainList[indx].iloc[indStart:indStop, 1:])
    nonBlinkTrainDf = nonBlinkTrainDf.append(nonBlinkTrainList[indx].iloc[indStart:indStop, 1:])
nonBlinkTest = []
nonBlinkTestDf = pd.DataFrame()
for indx, timestamp in enumerate(nonBlinkTestData):
    indStart, indStop = getIndices(nonBlinkTestList[indx], timestamp)
    nonBlinkTest.append(nonBlinkTestList[indx].iloc[indStart:indStop, 1:])
    nonBlinkTestDf= nonBlinkTestDf.append(nonBlinkTestList[indx].iloc[indStart:indStop, 1:])
    
# reset index
blinkTrainDf = blinkTrainDf.reset_index().iloc[:,1:]

blinkTestDf = blinkTestDf.reset_index().iloc[:,1:]

nonBlinkTrainDf = nonBlinkTrainDf.reset_index().iloc[:,1:]

nonBlinkTestDf = nonBlinkTestDf.reset_index().iloc[:,1:]


# save to csvs
blinkTrainDf.to_csv(path + 'blink_results/blinkTraining.csv')
blinkTestDf.to_csv(path + 'blink_results/blinkTesting.csv')
nonBlinkTrainDf.to_csv(path + 'blink_results/nonBlinkTraining.csv')
nonBlinkTestDf.to_csv(path + 'blink_results/nonBlinkTesting.csv')



# one run for each class
# same code as above but separated by class rather than by process
blinkTrainingList = glob.glob(path + 'blinkTraining/*.csv')
blinkTrainList = []
for member in blinkTrainingList:
    blinkTrainList.append(pd.read_csv(member).iloc[:,1:])
blinkTrainingData = glob.glob(path + 'blinkTraining/blinkTrainingData/*.csv')
blinkTrainData = []
for member in blinkTrainingData:
    blinkTrainData.append(pd.read_csv(member).iloc[0,1])
blinkTrain = []
blinkTrainDf = pd.DataFrame()
for indx, timestamp in enumerate(blinkTrainData):
    indStart, indStop = getIndices(blinkTrainList[indx], timestamp)
    blinkTrain.append(blinkTrainList[indx].iloc[indStart:indStop, 1:])
    blinkTrainDf = blinkTrainDf.append(blinkTrainList[indx].iloc[indStart:indStop, 1:])
blinkTrainDf = blinkTrainDf.reset_index().iloc[:,1:]
blinkTrainDf.to_csv(path + 'blink_results/blinkTraining.csv')


blinkTestingList = glob.glob(path + 'blinkTesting/*.csv')
blinkTestList = []
for member in blinkTestingList:
    blinkTestList.append(pd.read_csv(member).iloc[:,1:])
blinkTestingData = glob.glob(path + 'blinkTesting/blinkTestingData/*.csv')
blinkTestData = []
for member in blinkTestingData:
    blinkTestData.append(pd.read_csv(member).iloc[0,1])
blinkTest = []
blinkTestDf = pd.DataFrame()
for indx, timestamp in enumerate(blinkTestData):
    indStart, indStop = getIndices(blinkTestList[indx], timestamp)
    blinkTest.append(blinkTestList[indx].iloc[indStart:indStop, 1:])
    blinkTestDf = blinkTestDf.append(blinkTestList[indx].iloc[indStart:indStop, 1:])
blinkTestDf = blinkTestDf.reset_index().iloc[:,1:]
blinkTestDf.to_csv(path + 'blink_results/blinkTesting.csv')



nonBlinkTrainingList = glob.glob(path + 'nonBlinkTraining/*.csv')
nonBlinkTrainList = []
for member in nonBlinkTrainingList:
    nonBlinkTrainList.append(pd.read_csv(member).iloc[:,1:])
nonBlinkTrainingData = glob.glob(path + 'nonBlinkTraining/nonBlinkTrainingData/*.csv')
nonBlinkTrainData = []
for member in nonBlinkTrainingData:
    nonBlinkTrainData.append(pd.read_csv(member).iloc[0,1])   
nonBlinkTrain = []
nonBlinkTrainDf = pd.DataFrame()
for indx, timestamp in enumerate(nonBlinkTrainData):
    indStart, indStop = getIndices(nonBlinkTrainList[indx], timestamp)
    nonBlinkTrain.append(nonBlinkTrainList[indx].iloc[indStart:indStop, 1:])
    nonBlinkTrainDf = nonBlinkTrainDf.append(nonBlinkTrainList[indx].iloc[indStart:indStop, 1:])
nonBlinkTrainDf = nonBlinkTrainDf.reset_index().iloc[:,1:]
nonBlinkTrainDf.to_csv(path + 'blink_results/nonBlinkTraining.csv')





nonBlinkTestingList = glob.glob(path + 'nonBlinkTesting/*.csv')
nonBlinkTestList = []
for member in nonBlinkTestingList:
    nonBlinkTestList.append(pd.read_csv(member).iloc[:,1:])
nonBlinkTestingData = glob.glob(path + 'nonBlinkTesting/nonBlinkTestingData/*.csv')
nonBlinkTestData = []
for member in nonBlinkTestingData:
    nonBlinkTestData.append(pd.read_csv(member).iloc[0,1])
nonBlinkTest = []
nonBlinkTestDf = pd.DataFrame()
for indx, timestamp in enumerate(nonBlinkTestData):
    indStart, indStop = getIndices(nonBlinkTestList[indx], timestamp)
    nonBlinkTest.append(nonBlinkTestList[indx].iloc[indStart:indStop, 1:])
    nonBlinkTestDf= nonBlinkTestDf.append(nonBlinkTestList[indx].iloc[indStart:indStop, 1:])    
nonBlinkTestDf = nonBlinkTestDf.reset_index().iloc[:,1:]
nonBlinkTestDf.to_csv(path + 'blink_results/nonBlinkTesting.csv')
