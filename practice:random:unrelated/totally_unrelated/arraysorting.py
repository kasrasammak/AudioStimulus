#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 16:56:59 2020

@author: owlthekasra
"""


array = [2,3,9,5,2000,3,6,10,4]

newarr = []
last = None
for element in array:
    if (last != None):
        if (element > last):
            newarr.append(element)   
    last = element

def get_highest_2(arr):
    for i in range(len(array)):
        if (i == 0):
            newHighest = array[i]
        else:
            if array[i] > newHighest:
                oldHighest = newHighest
                newHighest = array[i]
    newarr = [newHighest, oldHighest]
    return newarr

newarr = get_highest_2(array)

