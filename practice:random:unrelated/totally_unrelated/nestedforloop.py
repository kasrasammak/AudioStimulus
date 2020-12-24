#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:20:11 2020

@author: owlthekasra
"""

import numpy as np

for i in range(0, 8):
    print(i**3)
    
    
a =  np.full((5, 5), np.nan)

for i in range(0, len(a)):
    for j in range(0, len(a)):
        if (i <= j):
            a[i][j] = j - i
        elif (i > j):
            a[i][j] = i - j
            