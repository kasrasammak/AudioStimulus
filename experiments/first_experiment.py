#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 18:48:30 2020

First experiment, collecting data using EEGedu.com

@author: owlthekasra
"""

import playwave as pw
import getcsv as nf
from pynput.mouse import Button, Controller
import time
from config import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import add_label


#variables
sine_bass = 'sine_bass'
no_sound = 'no_sound'
sine_bass_thought = 'sine_bass_thought'
aud_name = sine_bass+'.wav'
column_names = ["name", "duration", "start_rec_s", "start_sound_s", "end_sound_s", "fin_rec_ms", "sound_lag", "end_sound_lag", "dur_whole", "sound_to_fin"]
meta_df = pd.DataFrame(columns = column_names)

column_names_2 = ["name", "start_rec_s", "fin_rec_ms", "dur_whole"]

meta_df_2 = pd.DataFrame(columns = column_names_2)
meta_df_3 = pd.DataFrame(columns = column_names_2)


# num = num - 4
#latest_file = ""

mouse = Controller()
mouse.position
#Start Experiment

trials = 1



mouse.position =  (620.01171875, 1122.9765625)
mouse.click(Button.left, 2)
click_time = time.time()
(start, duration) = pw.play_sound(aud_name, proj_path, 0)


(new_name, fin) = nf.next_file(sine_bass, str(trials))
row = list([new_name, duration, click_time, start, start+duration, fin, start - click_time, (start - click_time + duration), fin/1000 - click_time, fin/1000 - start])
meta_df.loc[len(meta_df)] = row

#delete last row
# meta_df = meta_df[:-1]

#new_path = dst + sine_bass + "/" + new_name
meta_df.to_csv(sine_bass + '_trials_{}'.format(trials) + '_meta_data.csv')


#no sound

mouse.position =  (620.01171875, 1122.9765625)
mouse.click(Button.left, 2)
click_time = time.time()

(new_name, fin) = nf.next_file(no_sound, str(trials))
row_2 = list([new_name, click_time, fin, fin/1000 - click_time])
meta_df_2.loc[len(meta_df_2)] = row_2


meta_df_2.to_csv(no_sound + '_trials_{}'.format(trials) + '_meta_data_2.csv')




#thought

(start, duration) = pw.play_sound(aud_name, proj_path, 0)

mouse.position =  (685.10546875, 1134.05078125)
mouse.click(Button.left, 2)
click_time = time.time()

(new_name, fin) = nf.next_file(sine_bass_thought, str(trials))
row_3 = list([new_name, click_time, fin, fin/1000 - click_time])
meta_df_3.loc[len(meta_df_3)] = row_3


meta_df_3.to_csv(sine_bass_thought + '_trials_{}'.format(trials) + '_meta_data_3.csv')
