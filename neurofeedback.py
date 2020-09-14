#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 04:27:12 2020

@author: owlthekasra
"""

# -*- coding: utf-8 -*-

from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import pandas as pd
import playwave as pw
import keyboard
import time
# from muselsl import view
def create_lists(df):
    data = [item for sublist in df for item in sublist[0]]
    timestamp = [item for sublist in df for item in sublist[1]]
    return(data, timestamp)

proj_path = '/Users/owlthekasra/Documents/Code/Python/AudioStimulus'
aud_name = 'sine_c1.wav'
aud_name_2 =  'sine_d1.wav'
aud_name_3 = 'sine_f#1.wav'

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1
# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8
# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Empty dataframes
eeg = []
eeg2 = []
soundstamps = []
soundstamps_2 = []
soundstamps_3 = []
imaginestamps = []
imaginestamps_2 = []
imaginestamps_3 = []
nothingstamps = []
length_sound = 2.0287

c1 = 0
d1 = 0
f1 = 0
ic1 = 0
id1 = 0
if1 = 0
n1 = 0

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    # view(version=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    # Get the stream info and description
    info = inlet.info()
    description = info.desc()
    # Get the sampling frequency
    fs = int(info.nominal_srate())

    """ 3. GET DATA """
    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    switch = True;
    print('Press Ctrl-C in the console to break the while loop.')
    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
            
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=64)
            eeg.append([eeg_data, timestamp])  

            try:
                if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                    c1 = c1 + 1
                    print("Listen!", c1)
                    start, duration = pw.play_sound(aud_name, proj_path, 0)
                    soundstamps.append(start)
                    print('Done!')
                if keyboard.is_pressed('e'):
                    d1 = d1 + 1
                    print("Listen!", d1)
                    start, duration = pw.play_sound(aud_name_2, proj_path, 0)
                    soundstamps_2.append(start)
                    print('Done!')
                if keyboard.is_pressed('t'):
                    f1 = f1 + 1
                    print("Listen!", f1)
                    start, duration = pw.play_sound(aud_name_3, proj_path, 0)
                    soundstamps_3.append(start)
                    print('Done!') 
                if keyboard.is_pressed('p'):
                    ic1 = ic1 + 1
                    print("Imagine C1!", ic1)
                    start = time.time()
                    time.sleep(length_sound)
                    print("Finished!")
                    imaginestamps.append(start)
                if keyboard.is_pressed('i'):
                    id1 = id1 + 1
                    print("Imagine D1!", id1)
                    start = time.time()
                    time.sleep(length_sound)
                    print("Finished!")
                    imaginestamps_2.append(start)
                if keyboard.is_pressed('y'):
                    if1 = if1 + 1
                    print("Imagine F#1!", if1)
                    start = time.time()
                    time.sleep(length_sound)
                    print("Finished!")
                    imaginestamps_3.append(start)
                if keyboard.is_pressed('n'):
                    n1 = n1 + 1 
                    print("Do nothing!", n1)
                    start = time.time()
                    time.sleep(length_sound)
                    print("Finished!")
                    nothingstamps.append(start)
            except:
                print("You broke out of the loop")
            
    except KeyboardInterrupt:
        data, timestamps = create_lists(eeg)
        df = pd.DataFrame(data)
        df = df.iloc[:, :-1]
        df.insert(0, "Timestamp", timestamps)
        df.loc[df['Timestamp'] < start-.001, 'new?'] = 'False' 
        df.loc[df['Timestamp'] > start -.001, 'new?'] = 'True' 
        df.loc[df['Timestamp'] > start+length_sound, 'new?'] = 'False' 
        dfTest = df[df['new?'] == "True"]
        dfTest = dfTest.iloc[:, :-1]
        df = df.iloc[:, :-1]
        dfT = dfTest.T
        dfT['label'] = 1
        print(len(df))
        print('Closing!')
        
if not df.empty:
    df.to_csv("df_" + str(start) + ".csv")
if soundstamps: 
    soundstamps_df = pd.DataFrame(soundstamps)
    soundstamps_df.to_csv("sine_c1_" + str(soundstamps[0]) + ".csv")
if soundstamps_2:
    soundstamps_2_df = pd.DataFrame(soundstamps_2)
    soundstamps_2_df.to_csv("sine_d1_" + str(soundstamps_2[0]) + ".csv")
if soundstamps_3:
    soundstamps_3_df = pd.DataFrame(soundstamps_3)
    soundstamps_3_df.to_csv("sine_f#1_" + str(soundstamps_3[0]) + ".csv")
if imaginestamps:
    imaginestamps_df = pd.DataFrame(imaginestamps)
    imaginestamps_df.to_csv("imagine_sine_c1_" + str(imaginestamps[0]) + ".csv")
if imaginestamps_2:
    imaginestamps_2_df = pd.DataFrame(imaginestamps_2)
    imaginestamps_2_df.to_csv("imagine_sine_d1_" + str(imaginestamps_2[0]) + ".csv")
if imaginestamps_3:
    imaginestamps_3_df = pd.DataFrame(imaginestamps_3)
    imaginestamps_3_df.to_csv("imagine_sine_f#1_" + str(imaginestamps_3[0]) + ".csv")
if nothingstamps:
    nothingstamps_df = pd.DataFrame(nothingstamps)
    nothingstamps_df.to_csv("nothing_" + str(nothingstamps[0]) + ".csv")