#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 03:41:18 2020

@author: owlthekasra
"""

import methods as md
import add_label as al
import numpy as np
import pandas as pd
import random

sb_rd_1 = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/data/sine_bass/trials_2'
sb_rd_2 = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/data/sine_bass/trials_3'
sb_rd_3 = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/data/sine_bass/extra_deleted_metadata'
ns_rd_1 = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/data/no_sound/trials_1'
ns_rd_2 = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/data/no_sound/trials_2'
sbt_rd_1 = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/data/sine_bass_thought/trials_1'

_, df_sine_bass_extra = al.get_all_dataframes(sb_rd_3, 1)
_, df_sine_bass_trials_2 = al.get_all_dataframes(sb_rd_1, 1)
_, df_sine_bass_trials_3 = al.get_all_dataframes(sb_rd_2, 1)
_, df_no_sound_trials_1 = al.get_all_dataframes(ns_rd_1, 0)
_, df_no_sound_trials_2 = al.get_all_dataframes(ns_rd_2, 0)
_, df_sine_bass_thought_trials_1 = al.get_all_dataframes(sbt_rd_1, 2)


def get_X_and_y(df, start=1):
    y = df[['label']]
    X = df[:, start:]
    return (X, y)

def subtract_moving_average(df, n=50):
    k = n
    bgnorm = pd.DataFrame(np.zeros((len(df), len(df.columns))))
    for j in range(0, len(df)):
        for i in range(0, len(df.columns)):
            #define indices
            indices = range(max(1, i-k), min(i+k, len(df.columns)));
            avg = df.iloc[j, :]
            avg = avg.iloc[indices].mean()
            newnum = df.iloc[j, i] - avg
            print(newnum)
            bgnorm.iloc[j, i] = newnum
    return bgnorm

# #preprocess thought sine wave only
# y_values_thought = df_sine_bass_thought_trials_1.iloc[:, 0]
# X_values_thought = df_sine_bass_thought_trials_1.iloc[:, 132:660]
# df_thought = pd.concat([y_values_thought, X_values_thought], axis=1, ignore_index=True)

diff_labels = [df_sine_bass_thought_trials_1, df_sine_bass_extra, df_sine_bass_trials_2, df_sine_bass_trials_3, df_no_sound_trials_1, df_no_sound_trials_2]
big_frame = pd.concat(diff_labels, ignore_index=True)
bg = big_frame.drop(big_frame.iloc[:, 517:], axis=1)

#separate channels into different dataframes
bg1 = bg.iloc[::4, :]
bg2 = bg.iloc[1::4, :]
bg3 = bg.iloc[2::4, :]
bg4 = bg.iloc[3::4, :]

bigX = bg.iloc[:, 1:]
bigy = bg.iloc[:,0]

#subtracting average of each row
bigX = lab.iloc[:, 1:]
len(bigX.columns)
len(bigX)
bgnorm = subtract_moving_average(bigX)


bgnormlab = pd.concat([bigy, bgnorm], axis=1)

bgnormlab.to_csv('bgnormalized3600x517.csv')

bgnormlab = pd.read_csv('/Users/owlthekasra/Documents/Code/Python/AudioStimulus/data/csv/bgnormalized3600x517.csv')

j3 = pd.DataFrame()
def get_mean_down_df(df, nchan=4 ):
    avg_df = pd.DataFrame()
    for i in range(1, len(df)+1):
        if ((i % nchan) == 0):
            j5 = range(i-nchan,i)
            j1 = df.iloc[j5, :]
            k1 = j1.mean()
            avg_df = avg_df.append(k1, ignore_index=True)
    return avg_df

j5 = range(0,8)
j1 = bigX.iloc[j5, :]
    
bgnormavg = get_mean_down_df(bgnormlab)
lab = bgnormavg.iloc[:,-1]
lab = lab.append(bgnormavg.iloc[:,:-1])


indices = range(397, 417)
j1 = lab.iloc[1, :].iloc[1:]
j2 = j1.iloc[indices]
j3 = j2.mean()

random.seed(100)
# bssss = bgnormavg.drop(columns=['Unnamed: 0'])
main = bg1.sample(frac=1)
main = main.reset_index()
main = main.iloc[:, 1:]

train = main.iloc[:650]
val = main.iloc[650:]

# main2 = main.sample(frac=1)

X_train = main.iloc[:, 1:]
y_train = main['label']
X_val = val.iloc[:, 1:]
y_val = val['label']

model, acc, pred, y_test = md.fitPredictValSet(X_train, y_train, X_val, y_val, 'tree')

print(indices)