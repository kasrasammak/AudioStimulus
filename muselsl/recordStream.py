#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 04:30:47 2020

@author: owlthekasra
"""


"""
Record from a Stream
This example shows how to record data from an existing Muse LSL stream
"""
from muselsl import record

if __name__ == "__main__":

    # Note: an existing Muse LSL stream is required
    record(60)

    # Note: Recording is synchronous, so code here will not execute until the stream has been closed
    print('Recording has ended')