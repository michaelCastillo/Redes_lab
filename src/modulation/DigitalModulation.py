import math
import sys
import wave
sys.path.insert(0, '../')
import Plot as oPlot
sys.path.insert(0, './')
import DigitalDemodulation as demod


import numpy as np
import sys
import math
from matplotlib import pyplot as plt
import scipy.io.wavfile as wavfile
sys.path.insert(0, '../SoundInterface')
import SoundOut as sin
sys.path.insert(0, '../Files InOut')
import fileDirector as FD
# import DigitalDemodulation as DDemodulation
import warnings
from scipy import signal as sg
warnings.filterwarnings("ignore")
from bitstring import Bits
import wave as w
import threading



def ASK(signal, fs, bitRate, threads, title):

    t=np.arange(0, 1/bitRate, 1 / fs)
    A=2000
    B=50
    fc=fs/5
    
    carrier1 = A*np.cos(2*np.pi*fc*t)
    carrier2 = B*np.cos(2*np.pi*fc*t)
    samplesPerThreads = len(signal)//threads
    start = 0
    end = samplesPerThreads
    i = 0
    # Lista de hebras
    threadList = []
    while i<threads:
        # Se inicializa la hebra
        mythread = threading.Thread(target=signalOperator, args=(start, end, signal, t, fs, i, title, carrier1, carrier2)) 
        # Se agrega a la lista
        threadList.append(mythread)
        # Se inicia
        mythread.start()
        # El inicio aumenta en samplesPerThreads    
        start = start + samplesPerThreads
        # El final aumenta en samplesPerThreads
        end = end + samplesPerThreads
        i = i+1
    y = []
   #Portadoras
    # plt.figure(1)
    """
    plt.subplot(2,1,1)
    plt.title("ASK Carrier "+str(A)+" [db]")    
    plt.plot(carrier1)
    plt.subplot(2,1,2)    
    plt.title("ASK Carrier 2 "+str(B)+" [db]")    
    plt.plot(carrier2)
    plt.subplot(2,1,2)  
    plt.subplots_adjust(hspace = 1)
    #Señal
    """
    
    plt.figure(2)
    dCurve=genDigitalCurve(signal, fs, bitRate)
    plt.title("Señal Digital")
    plt.subplot(2,1,1)
    plt.plot(y)
    plt.title(title)    
    plt.subplot(2,1,2)
    plt.plot(dCurve)
    plt.subplots_adjust(hspace = 1)

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

def signalOperator(start, end, signal, t ,fs, i, fileName, carrier1, carrier2):
    y=[]
    # Se divide la senial de acuerdo al inicio y al final
    signal = signal[start:end]
    
    for sample in signal:
        sample="{0:08b}".format(sample)
        for bit in sample:
            if (bit=="1"):
                y.extend(carrier1)
            else:
                y.extend(carrier2)
    #Se agrega ruido gausiano a la señal modulada
    mean = 0
    std = 1
    #noise = np.random.normal(0.0, A, len(y))
    #y = y + noise
    # Se escribe en un archivo de salida la seccion leida
    sin.writeWav(fileName+str(i+1), fs, np.array(y))
    print("Hebra ", i ,"termino")

def FSK(signal, fs, bitRate, threads, title):
    
    plot = False
    signalModulated=[]
    A=10    
    y = []
    f1= 15000
    f2= 2000
    fs = 5*f1
    t=np.arange(0, 1/bitRate, 1 / fs)
    carrier1 = A*np.cos(2*np.pi*f1*t)
    carrier2 = A*np.cos(2*np.pi*f2*t)
    samplesPerThreads = len(signal)//threads
    start = 0
    end = samplesPerThreads
    i = 0
    # Lista de hebras
    threadList = []
    while i<threads:
        # Se inicializa la hebra
        mythread = threading.Thread(target=signalOperator, args=(start, end, signal, t, fs, i, title, carrier1, carrier2)) 
        # Se agrega a la lista
        threadList.append(mythread)
        # Se inicia
        mythread.start()
        # El inicio aumenta en samplesPerThreads    
        start = start + samplesPerThreads
        # El final aumenta en samplesPerThreads
        end = end + samplesPerThreads
        i = i+1

    if(plot):
        #Grafica de las portadoras
        plt.figure(1)
        plt.subplot(3,1,1)
        oPlot.plotSignalTime(carrier1,t,"Portadora (1)",False)
        plt.subplot(3,1,2)
        oPlot.plotSignalTime(carrier2,t,"Portadora (0) ",False)
        #Grafica de la señal modulada


def mainDigitalModulation(modType,flag,fileName):
    flagTest=False
    test=[]
    binarySignal=[]
    if(flagTest):  
        """
        #signal=signal[0:1000]
        print("Original: ", signal)
        print(len(signal))
        binaryFunc = np.vectorize(toBinary)
        binarySignalAux=binaryFunc(signal)
        maxBinaryLenght=len("{0:b}".format((max(signal))))
        
        for index, value in enumerate(binarySignalAux):
            i=len(str(value))
            while i<maxBinaryLenght:
                binarySignal.append(0)
                i=i+1
            for bit in value:
                binarySignal.append(int(bit))
        binarySignal=np.array(binarySignal)
        """
        fs, signal = wavfile.read("../../Wav/"+fileName+".wav")
        test=w.open("../../Wav/"+fileName+".wav", "rb")
        test=test.readframes(test.getnframes())
    else:
        test=[0,1,0,0,0,1]*1000
        plt.figure(4)
        plt.subplot(3,1,1)
        plt.plot(test)
        plt.title("test")
        #plt.subplot(3,1,2)
        #plt.plot(t,carrier_2)
        #plt.title("Carrier 2")
        #plt.subplots_adjust(hspace = 1)
        fs = 1000 #Frecuencia de muestreo en Hz
    fs=10000
    bitRate=100 #Bit por segundo
    threads = int(input("Ingrese cantidad de hebras: "))
    y=np.array([])    
    if(modType=="ASK"):
        #Funcion que modula
        y=ASK(test, fs, bitRate,threads,  title="ASK "+fileName)

    #FSK
    if(modType=="FSK"):
        #Funcion que modula
        print("COMENZANDO MODULACION")
        y=FSK(test, fs, bitRate, threads, title="FSK "+fileName)
    #sin.writeWav(fileName+modType, fs, y)
    plt.show()
    #Plot de correlacionadores
    #demodulacion FSK
    #DDemodulation.mainDigitalDemodulation(flag, bitRate, fileName+modType)

mainDigitalModulation("ASK",1,"handel")
