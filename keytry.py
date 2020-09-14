#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  5 09:14:46 2020

@author: owlthekasra
"""

import keyboard

while True:
    try:
        if keyboard.is_pressed('p'):
            print ("Wee")
            break
    except:
        break
    