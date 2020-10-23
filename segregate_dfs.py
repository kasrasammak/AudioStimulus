#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 05:21:14 2020

@author: owlthekasra
"""

import pandas as pd
import numpy as np
import os      
from getcsv import path_leaf                                                                                                       

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]
def transpose_add_label(df, label):
    dfT = df.T
    int1 = int(df.index[0])
    int2 = int(df.index[-1])
    cols = pd.Series(range(int1,int2))
    dfT['label'] = label
    l_cols = cols.values.tolist() 
    label = ['label']
    for lab in l_cols:
        label.append(lab)
    dfT = dfT[label]
    return dfT
def segregate_dfs(df, stamps, length, label):
    newDF = []
    for i in range(0,len(stamps)):
        array = df['Timestamp']
        value = stamps.iloc[i,0]
        valueIndexStart = find_nearest(array, value)
        valueIndexStop = find_nearest(array, value + length)
        if (valueIndexStart != valueIndexStop):
            indStart = int(df[df['Timestamp']==valueIndexStart].index[0])
            indStop = int(df[df['Timestamp']==valueIndexStop].index[0])
            endDF = df.iloc[indStart:indStop].reset_index().iloc[:, 1:]
            time = endDF['Timestamp']
            endDF = endDF.loc[:, df.columns != 'Timestamp']
            endDF.insert(loc=0, column='Timestamp', value=time)
            endDFT = transpose_add_label(endDF, label)
            newDF.append(endDFT)
    return newDF
def combine_seg_dfs(seg_df):
    concat_df = pd.DataFrame()
    for i in range(0, len(seg_df)):
        if(len(seg_df[i].columns) > 518):
            newshit = seg_df[i].iloc[1:, :]
            concat_df = pd.concat([concat_df, newshit.iloc[:, :519]], axis = 0)
    return concat_df

def list_files(dir):                                                                                                  
    r = []
    subdirs = [x[0] for x in os.walk(dir)]                                                                            
    for subdir in subdirs:    
        subr = []                                                                       
        files = os.walk(subdir).__next__()[2]                                                                             
        if (len(files) > 0):                                                                                          
            for file in files:
                if (file!='.DS_Store'):                                                                                     
                    subr.append(subdir + "/" + file)
        if (subr):
            r.append(subr)                                                                   
    return r 

def select_filename_with(set_of_names, df_starts_with):
    for filename in set_of_names:
        tail = path_leaf(filename)
        if (tail.startswith(df_starts_with)):
            return filename

def get_it_girl(list_of_files, name):
    main_list = []
    boolean = False
    for element in list_of_files:
        for filename in element:
            if (path_leaf(filename).startswith(name)):
                boolean = True
        if (boolean):
            df = pd.read_csv(select_filename_with(element, 'df'))
            df = df.iloc[:,1:]
            stamps = pd.read_csv(select_filename_with(element, name))
            stamps = stamps.iloc[:,1:]
            main_list.append((df, stamps))
            boolean = False
    return main_list

def get_medium_df_from_folder(tuple_list, name, label, length):
    newDF = segregate_dfs(tuple_list[0], tuple_list[1], length, label)
    first = combine_seg_dfs(newDF)
    return first
def get_big_df_from_folders(main_list, name, label, length):
    big =pd.DataFrame()
    for element in main_list:
        first = get_medium_df_from_folder(element, name, label, length)
        big = big.append(first)
    return big
def get_big_ass_df(list_of_files, names, length):
    big_ass_df = pd.DataFrame()
    x = 0
    for name in names:
        newDF = get_big_df_from_folders(get_it_girl(list_of_files, name), name, x, length)
        big_ass_df = big_ass_df.append(newDF)
        x = x+1
    return big_ass_df

def get_separate_channel_dfs(big_ass_df):
    reset = big_ass_df.reset_index()
    newdf = reset[reset['index']=='0'].iloc[:, 1:]
    newdf1 = reset[reset['index']=='1'].iloc[:, 1:]
    newdf2 = reset[reset['index']=='2'].iloc[:, 1:]
    newdf3 = reset[reset['index']=='3'].iloc[:, 1:]
    return (newdf, newdf1, newdf2, newdf3)

names = ['nothing', 'sine_c1', 'sine_d1', 'sine_f#1', 'imagine_sine_c1', 'imagine_sine_d1', 'imagine_sine_f#1']
length = 2.0287
directory = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/sine_bass_trials/'
training_list = list_files(directory + 'training_set')
validation_list = list_files(directory + 'validation_set')

big_ass_training_set = get_big_ass_df(training_list, names, length)
big_ass_validation_set = get_big_ass_df(validation_list, names, length)

training_chans = get_separate_channel_dfs(big_ass_training_set)
validation_chans = get_separate_channel_dfs(big_ass_validation_set)



