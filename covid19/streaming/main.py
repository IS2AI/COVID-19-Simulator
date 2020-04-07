__name__ = '__main__'

from visual_show import *
from data_stream import *

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
    print('[INFO] Start the program.')
    main()
