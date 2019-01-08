import math
import sys
import wave
sys.path.insert(0, '../')
import Plot as oPlot
sys.path.insert(0, './')
import DigitalDemodulation as demod


import numpy as np
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
from numpy.ma.core import concatenate


sys.path.insert(0, '../SoundInterface')
import SoundOut as sin
sys.path.insert(0, '../Files InOut')
import fileDirector as FD
# import DigitalDemodulation as DDemodulation
import warnings
from scipy import signal as sg
warnings.filterwarnings("ignore")
from bitstring import Bits




def ASK(signal, fs, bitRate, title):
    t=np.arange(0, 1/bitRate, 1 / fs)
    A=2000
    B=50
    fc=fs/5
    
    carrier1 = A*np.cos(2*np.pi*fc*t)
    carrier2 = B*np.cos(2*np.pi*fc*t)
    y=[]
    for sample in signal:
        sample = "{0:08b}".format(sample)
        for bit in sample:
            if(bit == "1"):
                y.extend(carrier1)
            else:
                y.extend(carrier2)
    
    #Se agrega ruido gausiano a la señal modulada
    mean = 0
    std = 1
    noise = np.random.normal(0.0, 2000, len(y))

    #Demodulacion
    
    # ask_demodulation(y,carrier1,carrier2,t,fs,bitRate)

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

def getSignalTime(fs_rate, signal):
    signal_len = float(len(signal))
    tAux = float(signal_len) / float(fs_rate)
    t = np.linspace(0, tAux, signal_len)
    return t


def FSK(signal, fs, bitRate, title):
    
    plot = False
    A=10
    f1= 15000
    f2= 2000
    fs = 5*f1
    t=np.arange(0, 1/bitRate, 1/fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)
    valToPrint = []
    for i in range(0,10):
        valToPrint.extend("{0:08b}".format(signal[i]))
    print(str(valToPrint))
    signalModulated=[]
    for sample in signal:
        sample = "{0:08b}".format(sample)
        for bit in sample:
            if(bit == "1"):
                signalModulated.extend(carrier1)
            else:
                signalModulated.extend(carrier2)
    
    if(plot):
        #Grafica de las portadoras
        plt.figure(1)
        plt.subplot(3,1,1)
        oPlot.plotSignalTime(carrier1,t,"Portadora (1)",False)
        plt.subplot(3,1,2)
        oPlot.plotSignalTime(carrier2,t,"Portadora (0) ",False)
        #Grafica de la señal modulada
        plt.subplot(3,1,3)
        signalTime = getSignalTime(fs,signalModulated)
        oPlot.plotSignalTime(signalModulated,signalTime,"Señal modulada",False)
        
    noise = np.random.normal(0.0, 5, len(signalModulated))
    signalModulated = signalModulated + noise
    return np.array(signalModulated)


def mainDigitalModulation(modType,flag,fileName):
    flagTest=True
    test=[]
    if(flagTest):
        fs, signal = FD.openDigitalWav(fileName)
        waveData = wave.open("../../Wav/"+fileName+".wav","rb")
        test = waveData.readframes(waveData.getnframes())
    else:
        test=[0,1,0,0,0,1]*1000
        print(test)
        plt.figure(4)
        plt.subplot(3,1,1)
        plt.plot(test)
        plt.title("test")
        #plt.subplot(3,1,2)
        #plt.plot(t,carrier_2)
        #plt.title("Carrier 2")
        #plt.subplots_adjust(hspace = 1)
        fs = 40000 #Frecuencia de muestreo en Hz
    
    bitRate=1000 #Bit por segundo

    
    modulatedSignal=np.array([])    
    if(modType=="ASK"):
        #Funcion que modula
        modulatedSignal=ASK(test, fs, bitRate, title="ASK "+fileName)

    #FSK
    if(modType=="FSK"):
        #Funcion que modula
        # print("Binary Signal => "+str(test))
        modulatedSignal=FSK(test, fs, bitRate, title="FSK "+fileName)
        print("He modulado")
        demodulatedSignal = demod.fsk_demodulation(modulatedSignal,15000,2000,5*15000,bitRate)
        print("He demodulado")
        print("Tamanos senales\n")
        print("signal=> "+str(len(test)) +"  demod=>"+str(len(demodulatedSignal)))

        result = demod.depureMachine(test,demodulatedSignal)
        print("tasa de error: "+str(float(result/len(test))*100) + "%")
        print("He finalizado")
    #Escribir archivo .wav    
    sin.writeWav(fileName+modType, fs, modulatedSignal)
    plt.show()
    
mainDigitalModulation("FSK",1,"ook")