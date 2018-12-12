import numpy as np
import sys
from matplotlib import pyplot as plt
import scipy.io.wavfile as wavfile
sys.path.insert(0, '../SoundInterface')
import SoundOut as sin


def ASK(signal, fs, bitRate, title):
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
   #Portadoras
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.title("ASK Carrier "+str(A)+" [db]")    
    plt.plot(carrier1)
    plt.subplot(2,1,2)    
    plt.title("ASK Carrier 2 "+str(B)+" [db]")    
    plt.plot(carrier2)
    plt.subplot(2,1,2)  
    plt.subplots_adjust(hspace = 1)
    #Señal
    plt.figure(2)
    dCurve=genDigitalCurve(signal, fs, bitRate)
    plt.title("Señal Digital")
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.title(title)    
    plt.subplot(2,1,2)
    plt.plot(dCurve)
    plt.subplots_adjust(hspace = 1)
    return np.array(y)
    
def genDigitalCurve(signal, fs, bitRate):
    t=np.arange(0, 1/bitRate, 1 / fs)
    
    carrier1 = [1]*len(t)
    carrier2 = [0]*len(t)
    y=[]
    for bit in signal:
        if (bit==1):
            y.extend(carrier1)
        else:
            y.extend(carrier2)
    
    return np.array(y)

def FSK(signal, fs, bitRate, title):
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
    #Portadoras
    plt.figure(1)
    plt.subplot(2,1,1)
    plt.title("FSK Carrier "+str(f1)+" [Hz]")    
    plt.plot(carrier1)
    plt.subplot(2,1,2)    
    plt.title("FSK Carrier 2 "+str(f2)+" [Hz]")    
    plt.plot(carrier2)
    plt.subplot(2,1,2)  
    plt.subplots_adjust(hspace = 1)
    #Señal
    plt.figure(2)
    dCurve=genDigitalCurve(signal, fs, bitRate)
    plt.title("Señal Digital")
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.title(title)    
    plt.subplot(2,1,2)
    plt.plot(dCurve)
    plt.subplots_adjust(hspace = 1)
    return np.array(y)


def mainDigitalModulation(modType,flag,fileName):
    test=[0,1]*5
    bitRate=10 #Bit por segundo
    fs = 1000 #Frecuencia de muestreo en Hz
    #ASK
    y=np.array([])
    
    if(modType=="ASK"):
        #Funcion que modula
        y=ASK(test, fs, bitRate, title="ASK "+fileName)

    #FSK
    if(modType=="FSK"):
        #Funcion que modula
        y=FSK(test, fs, bitRate, title="FSK "+fileName)
    #Escribir archivo .wav
    sin.writeWav(fileName+modType, fs, y)
    if(flag==1):
        plt.show()
