__name__ = '__main__'

from Visuals import *
from DataStr import *
import multiprocessing as mp
from network_sim import Network
import os
from covid_simulator_upd import Node
import random
import numpy as np
import csv
import copy
import time

def threads(callbackFunc, running):
    datastream = DataStream(callbackFunc=callbackFunc, running=running) #initialize
    datastream.start()

def main():

    # Set global flag
    event = threading.Event() # event used to communicate between threads
    event.set() # set to True

    webVisual = Visual(callbackFunc=threads, running=event) #start Bokeh web document
    threads(callbackFunc=webVisual, running=event)

if __name__ == '__main__':
    try:
        print('[INFO] Start the program.')
        main()
    except (KeyboardInterrupt):
        print('[INFO] Exiting the program. ')
        exit
