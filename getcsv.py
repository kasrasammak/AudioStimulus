#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 02:15:19 2020

@author: owlthekasra
"""

import ntpath
import glob
import os, shutil
import datetime
import re
import pandas as pd
from config import *

#ntpath.basename(latest_file)


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

def path_leaf_head(path):
    head, tail = ntpath.split(path)
    return head or ntpath.basename(head)

def rename(old, new_name):
    head = path_leaf_head(old)
    tail = path_leaf(old)
    os.rename(r'' + head + "/" + tail,r'' + head + "/" + new_name)

def next_file(type_name, trials):
    global latest_file, num, src, dst
    moveto =  dst + type_name + "/" + 'trials_' + trials + "/"
    print(latest_file)
    list_of_files = glob.glob(src + '*.csv') 
    previous_file = latest_file
    latest_file = max(list_of_files, key=os.path.getctime)
    #latest_file_time = time.ctime(os.path.getctime(latest_file))
    if (previous_file != latest_file):
        num += 1
        temp = re.findall(r'\d+', latest_file) 
        sec = int(temp[0])
        strsec = str(sec)
        trial = 'trial_{}'.format(num)
        new_name = '{}'.format(type_name)+ "_" + strsec +'_{}.csv'.format(trial)
        rename(latest_file, new_name)
        shutil.move(src + new_name, moveto + new_name )
    return (new_name, sec)


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


#try:
#    os.mkdir(path)
#except OSError:
#    print ("Creation of the directory %s failed" % path)
#else:
#    print ("Successfully created the directory %s " % path)

#if (previous_file != latest_file):
#    f = pd.read_csv(latest_file)
#        
#
#print(f)
#os.path.getctime
#
#print(time.ctime(os.path.getctime(latest_file)))
#latest_file
#



