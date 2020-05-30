#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 04:31:41 2020

@author: owlthekasra
"""


"""
Starting a Stream
This example shows how to search for available Muses and
create a new stream
"""
from muselsl import stream, list_muses

if __name__ == "__main__":

    muses = list_muses()

    if not muses:
        print('No Muses found')
    else:
        stream(muses[0]['address'])

        # Note: Streaming is synchronous, so code here will not execute until the stream has been closed
        print('Stream has ended')