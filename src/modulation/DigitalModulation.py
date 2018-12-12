import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wavfile


def writeWav(fileName, rate, data):
    wavfile.write(fileName, rate, data)

def ASK(signal, fs, bitRate, nFigura, title):
    t=np.arange(0, 1/bitRate, 1 / fs)
    A=20
    B=10
    fc=fs/5
    
    carrier1 = A*np.cos(2*np.pi*fc*t)
    carrier2 = B*np.cos(2*np.pi*fc*t)
    y=[]
    for bit in signal:
        if (bit==1):
            y.extend(carrier1)
        else:
            y.extend(carrier2)
    plt.figure(nFigura)
    plt.title("ASK Carrier 1")    
    plt.plot(carrier1)
    plt.figure(nFigura+1)
    plt.title("ASK Carrier 2")    
    plt.plot(carrier2)
    plt.figure(nFigura+2)
    plt.title(title)
    plt.plot(y)
    return np.array(y)

def FSK(signal, fs, bitRate, nFigura, title):
    A=10
    f1=fs/5 #Hz
    f2=fs/6 #Hz
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)
    y=[]
    for bit in signal:
        if (bit==1):
            y.extend(carrier1)
        else:
            y.extend(carrier2)
    plt.figure(nFigura)
    plt.title("FSK Carrier 1")    
    plt.plot(carrier1)
    plt.figure(nFigura+1)
    plt.title("FSK Carrier 2")    
    plt.plot(carrier2)
    plt.figure(nFigura+2)
    plt.title(title)
    plt.plot(y)
    return np.array(y)


def mainDigitalModulation():
    test=[0,1,0,1,0,1]
    test2=[1,0]*100
    bitRate=10 #Bit por segundo
    titleFSK="FSK"
    titleASK="ASK"
    fs = 1000 #Frecuencia de muestreo en Hz

    #ASK
    yASK=ASK(test, fs, bitRate, nFigura=1, title=titleASK+" test1")
    yASK2=ASK(test2, fs, bitRate, nFigura=4, title=titleASK+ "test2")

    #FSK
    #y=FSK(test, fs, bitRate, nFigura=7, title=titleFSK+" test1")
    #y2=FSK(test2, fs, bitRate, nFigura=10, title=titleFSK+" test2")

    #writeWav("test1FSK.wav", fs, y)
    #writeWav("test2FSK.wav", fs, y2)
    writeWav("test1ASK.wav", fs, yASK)
    writeWav("test2ASK.wav", fs, yASK2)
    plt.show()