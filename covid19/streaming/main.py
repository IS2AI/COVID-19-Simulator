__name__ = '__main__'

from visual_show import *
from data_stream import *

def threads(callbackFunc, running):

    # set up the thread for reading the data
    datastream = DataStream(callbackFunc=callbackFunc, running=running)
    datastream.start()

def main():

    print('[INFO] Start the program.')

    # start the thread
    event = threading.Event()
    event.set()

    # open the web interface
    webVisual = Visual(callbackFunc=threads, running=event)
    threads(callbackFunc=webVisual, running=event)


if __name__ == '__main__':
    main()
