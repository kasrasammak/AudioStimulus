#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 08:27:00 2020

@author: owlthekasra
"""
import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import pandas as pd
import utils
import cv2
from matplotlib import pyplot as plt
from numba import cuda


from __future__ import absolute_import, print_function
import numpy as np
import pyopencl as cl

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)

ctx = cl.create_some_context()
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags
a_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a_np)
b_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b_np)

prg = cl.Program(ctx, """
__kernel void sum(
    __global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid = get_global_id(0);
  res_g[gid] = a_g[gid] + b_g[gid];
}
""").build()

res_g = cl.Buffer(ctx, mf.WRITE_ONLY, a_np.nbytes)
prg.sum(queue, a_np.shape, None, a_g, b_g, res_g)

res_np = np.empty_like(a_np)
cl.enqueue_copy(queue, res_np, res_g)

# Check on CPU with Numpy:
print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))
assert np.allclose(res_np, a_np + b_np)

def create_lists(df):
    data = [item for sublist in df for item in sublist[0]]
    timestamp = [item for sublist in df for item in sublist[1]]
    return(data, timestamp)
def gammaCorrect2(im , gamma ):
    # create a blank output Image
    outImg = np.zeros(im.shape,im.dtype)
    rows,cols,rgbs = im.shape
    # n = int(rows*cols/fs)
    # y=1
    # a=0
    for i in range(rows):
        for j in range(cols):
            # if (y % n == 0):
            #     a += 1   
            for x in range(rgbs):        
                gammaValue = (im.item(i,j,x) + (gamma*600))
                outImg.itemset((i,j,x), gammaValue)
                # y +=1

    return (outImg, gammaValue)

img = cv2.imread('/Users/owlthekasra/Documents/Code/Python/AudioStimulus/vikyni.bmp')  #load image

class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """

BUFFER_LENGTH = 5

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1
# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8
# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH
# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

eeg = []
eeg2 = pd.Series([])
d = {}

if __name__ == "__main__":

    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())
    
    """ 2. INITIALIZE BUFFERS """
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter
    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))
    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))
    print('Press Ctrl-C in the console to break the while loop.')
    x = 0
    cap = cv2.VideoCapture(0)
    e = 1
    try:
        while True:   
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=64)
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer,
                                                 np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])

            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            # These metrics could also be used to drive brain-computer interfaces
            alpha_metric = smooth_band_powers[Band.Alpha] / \
                smooth_band_powers[Band.Delta]
                
            print('Alpha Relaxation: ', alpha_metric)
            # beta_metric = smooth_band_powers[Band.Beta] / \
            #     smooth_band_powers[Band.Theta]
            # print('Beta Concentration: ', beta_metric)
            # theta_metric = smooth_band_powers[Band.Theta] / \
            #     smooth_band_powers[Band.Alpha]
            # print('Theta Relaxation: ', theta_metric)
            # if (x%2 ==0):
            #     gbaby, gwave = gammaCorrect2(img, abs(alpha_metric))
            #     plt.imshow(gbaby)
            #     plt.show()
            eeg.append([eeg_data, timestamp])
            eeg_64 = pd.DataFrame(eeg_data).iloc[:,:-1]
            eeg_64.insert(len(eeg_64.columns), "Timestamp", timestamp)
            eeg2 = eeg2.append(eeg_64)
            if(len(eeg2) == 256):
                eeg2 = eeg2.reset_index().iloc[:,1:]
                d["{0}".format(x)] = eeg2
                eeg2 = pd.Series([])
                x = x + 1
            ret, frame = cap.read()
            gbaby, gwave = gammaCorrect2(frame, abs(alpha_metric))
            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, 1)
            # e +=1
            # Display the resulting frame
            cv2.imshow('frame',gbaby)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything done, release the capture

            # print(eeg_data[1])
            
    except KeyboardInterrupt:
        data, timestamps = create_lists(eeg)
        df = pd.DataFrame(data)
        df = df.iloc[:, :-1]
        df.insert(len(df.columns), "Timestamp", timestamps)
        eeg2 = eeg2.reset_index().iloc[:,1:]
        print('Closing!')

cap.release()
cv2.destroyAllWindows()

dp = d["0"][0]

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, 5)

    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np


def mouseRGB(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN: #checks mouse left button down condition
        colorsB = frame[y,x,0]
        colorsG = frame[y,x,1]
        colorsR = frame[y,x,2]
        colors = frame[y,x]
        print("Red: ",colorsR)
        print("Green: ",colorsG)
        print("Blue: ",colorsB)
        print("BRG Format: ",colors)
        print("Coordinates of pixel: X: ",x,"Y: ",y)


cv2.namedWindow('mouseRGB')
cv2.setMouseCallback('mouseRGB',mouseRGB)

capture = cv2.VideoCapture(0)

while(True):

    ret, frame = capture.read()

    cv2.imshow('mouseRGB', frame)

    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()

from PIL import Image

myImage = Image.open('/Users/owlthekasra/Documents/Code/Python/AudioStimulus/vikyni.bmp')

pixels = list(myImage.getdata())
OUTPUT_IMAGE_SIZE = myImage.size

n = int(len(pixels)/fs)



y=0  
list_of_pixels = []  
while (y<len(pixels)):
    pixels[0]
    for i in range(0,n):
        newple = tuple(int(dp[y/n]+x) for x in pixels[i])
        newple = tuple([0 if i < 0 else i for i in newple])
        newple = tuple([255 if i > 255 else i for i in newple])
        list_of_pixels.append(newple)
    y = y+n

image = Image.new('RGB', OUTPUT_IMAGE_SIZE)
image.putdata(list_of_pixels)
image.save("filename.bmp")
image.show()
myImage.size

import numpy as np
import time
import cv2
from matplotlib import pyplot as plt

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
def gammaCorrect1(im , gamma  ):
    # create a blank output Image
    outImg = np.zeros(im.shape,im.dtype)
    rows,cols, rgbs = im.shape

    for i in range(rows):
        for j in range(cols):
            outImg[i][j] = ( (im[i][j]/255.0) ** (1/gamma) )*255

    return outImg


            
gbaby, gammaValue = gammaCorrect2(img, dp[1])
plt.imshow(gbaby)
plt.show()
# cv2.imshow("image",gbaby)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
img.dtype
0 % 256