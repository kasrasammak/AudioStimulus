#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 23:07:44 2020

@author: owlthekasra
"""

import wave
import pyaudio
import time
import os


def download_wait(path_to_downloads):
    seconds = 0
    dl_wait = True
    while dl_wait and seconds < 7:
        time.sleep(1)
        dl_wait = False
        for fname in os.listdir(path_to_downloads):
            if fname.endswith('.crdownload'):
                dl_wait = True
        seconds += 1
    return seconds


def play_sound(audio, path, wait):
    time.sleep(wait)
    os.chdir(path)
    sound_stim = wave.open(audio, 'rb')
    sample_rate = sound_stim.getframerate()
    num_samples = sound_stim.getnframes()
    duration = round(num_samples/sample_rate, 4)
    
    p = pyaudio.PyAudio()
    
    
    def callback(in_data, frame_count, time_info, status):
        data =  sound_stim.readframes(frame_count)
        return (data, pyaudio.paContinue)
    
    stream = p.open(
            format=p.get_format_from_width(sound_stim.getsampwidth()),
            channels=sound_stim.getnchannels(),
            rate=sample_rate,
            output=True,
            stream_callback = callback)
    
    
    before = time.time()
    stream.start_stream()
    after = time.time()
    delta = round((after*1000) - (before*1000), 3)
    print(delta)
    while stream.is_active():
        time.sleep(0.1)
        
    stream.stop_stream()
    stream.close()
    
    p.terminate()
    return (after, duration)


#n = 1024
#count = 0
#now = time.time()
#data = sound_stim.readframes(n)
#while data:
#    before = time.time()
#    stream.write(data)
#    then = now
#    now = time.time()
#    delta = round((now*1000) - (before*1000), 3)
#    count += 1
#    print("Current Position: " + 
#          str(sound_stim.tell()) + 
#          " | " + str(time.time()) + 
#          " | " + str(count) +
#          " | " + str(delta))
#    data = sound_stim.readframes(n)
    

    
#while data:
#    before = datetime.now()
#    stream.write(data)
#    after = datetime.now()

