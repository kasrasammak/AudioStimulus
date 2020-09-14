#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 17:09:50 2020

@author: owlthekasra
"""

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
    except KeyboardInterrupt:
        print("closing!")
        
        