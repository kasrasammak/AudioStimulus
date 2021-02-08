#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 03:28:03 2020

@author: owlthekasra
"""


import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import Grouper
import matplotlib.pyplot as plt
from matplotlib import pyplot
import datetime
import seaborn as sns

import os
import glob

def tpose_add_label_1x1(df, recording, label):
    df = df.iloc[:,0:5]
    dfT = df.T
    cols = pd.Series(range(0,len(df)))
    dfT['label'] = label
    
    rec = recording
    suff = '_' + rec
    
    l_cols = cols.values.tolist() 
    # l_cols_str = list(map(str, l_cols))
    label = ['label']
    for lab in l_cols:
        label.append(lab)
        
    dfT = dfT[label]
    dfT = dfT.set_index(dfT.index + suff)
    return dfT
# big_frame.iloc[1:]



def get_all_dataframes(rootdir, label):
    filenames = glob.glob(rootdir + "/*.csv")
    names = []
    timesigs = []
    dfs = []
    for filename in filenames:
        name = os.path.basename(filename)
        name = name.split('.')
        names.append(name[0])
        temp = [int(s) for s in name[0].split('_') if s.isdigit()]
        timesigs.append(str(temp[0]))
        dfs.append(pd.read_csv(filename))
    
    dfT = []
    for (df, sig) in zip(dfs, timesigs): 
        dfT.append(tpose_add_label_1x1(df, sig, label))
          
    dftlabels = []
    for (name, df) in zip(names, dfT):
        dftlabels.append((name, df))
        
    dftnew = []
    for df in dfT:
        dftnew.append(df.iloc[1:])
       
    big_frame = pd.concat(dftnew, ignore_index=True)
    
    return (dftlabels, big_frame)

def get_long_dataframe(rootdir):
    filenames = glob.glob(rootdir + "/*.csv")
    names = []
    timesigs = []
    dfs = []
    for filename in filenames:
        name = os.path.basename(filename)
        name = name.split('.')
        names.append(name[0])
        temp = [int(s) for s in name[0].split('_') if s.isdigit()]
        timesigs.append(str(temp[0]))
        dfs.append(pd.read_csv(filename))
    
    dfT = []
    for df in dfs: 
        dfT.append(df.iloc[:512, 1:5])
          
    dftlabels = []
    for (name, df) in zip(names, dfT):
        dftlabels.append((name, df))
        
    dftnew = []
    for df in dfT:
        dftnew.append(df)
       
    big_frame = pd.concat(dftnew, ignore_index=True)
    
    return big_frame


def add_label_train_test_split(df, label, ntrials):
    nbchan = len(df)
    df = df.T
    comp = np.empty([len(df), 1])
    comp[:,:] = label
    comp2 = np.append(df, comp, axis = 1)
    comp2 = add_dimension(comp2, pnts, ntrials, nbchan+1)
    compX = comp2[:,:,:nbchan]
    compY = comp2[:,:,nbchan]
    return (compX, compY)








# x_train_1, y_train_1 = add_label_train_test_split(comp_blink_train, 1, 68)
# x_train_2, y_train_2 = add_label_train_test_split(comp_nonblink_train, 0, 69)

# x_tr = np.append(x_train_1, x_train_2[:68,:,:], axis=1)
# y_tr = np.append(y_train_1, y_train_2[:68,:], axis=1)

# x_test_1, y_test_1 = add_label_train_test_split(comp_blink_test_ged, 1, 68)
# x_test_2, y_test_2 = add_label_train_test_split(comp_nonblink_test_ged, 0, 68)

# x_t = np.append(x_test_1, x_test_2[:,:,:], axis=1)
# y_t = np.append(y_test_1, y_test_2[:,:], axis=1)
    
# import methods as md

# mod, acc = md.fitPredictValSet(x_tr, y_tr, x_t, y_t, "lda", n_components=2)





# dft_labels, big_frame = get_all_dataframes(rootdir, 1)




# for subdir, dirs, files in os.walk(rootdir):
#     for file in files:
#         names.append(file)


# dfTry = pd.read_csv('/Users/owlthekasra/Downloads/Raw_Recording_1588092203049.csv')
# tpTry = tpose_add_label_1x1(dfTry, '1588092203049', 0)

# tpTry2 = tpTry.iloc[:,1:]
# tpTrpy = tpTry2.T
# tpTrpy.columns.values[1]



# # Use seaborn style defaults and set the default figure size
# sns.set(rc={'figure.figsize':(11, 4)})

# tpTrpy[tpTrpy.columns.values[0]].plot(linewidth=0.5)
# x = tpTrpy[tpTrpy.columns.values[0]]
# y = tpTrpy[tpTrpy.columns.values[1]]
# plt.plot(x,y)
# plt.show()

# groups = tpTry2.groupby(Grouper(freq='A'))
# years = DataFrame()
# for name, group in groups:
# 	years[name.year] = group.values
# years.plot(subplots=True, legend=False)
# pyplot.show()

# ['Consumption'].plot(linewidth=0.5);

# tpTrpy['datetime']=pd.to_datetime(tpTrpy[tpTrpy.columns.values[0]])


# df['v4'] = df['v2'].apply(myFunction.classify)

# tpTrpy.plot(kind='scatter',x=tpTrpy.columns.values[0],y=tpTrpy.columns.values[1],color='red')
# plt.show()

