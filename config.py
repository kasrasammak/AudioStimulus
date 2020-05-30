#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 20:03:14 2020

@author: owlthekasra
"""

src = '/Users/owlthekasra/Downloads/'
dst = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus/data/'
proj_path = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus'

import pandas as pd

num = 0
latest_file = ""

def run_config():
    global src, dst, proj_path, num, latest_file
    src = src
    dst = dst
    proj_path = proj_path
    num = num
    latest_file = latest_file